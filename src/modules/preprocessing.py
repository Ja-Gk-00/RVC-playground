# src/modules/preprocessing.py
"""
Preprocessing for RVC training.

Uses the SAME feature extraction as inference to ensure consistency:
- HuBERT: HuggingFace transformers model, layer 12
- RMVPE: For F0 extraction at hop 160, downsampled to hop 320
"""
from __future__ import annotations

import os
os.environ["TORCHAUDIO_USE_TORCHCODEC"] = "0"

from pathlib import Path
import numpy as np
import torch
import torchaudio

from src.models.index_creator import IndexCreator
from src.models.preprocessor import Preprocessor
from src.data_models.data_models import UnprocessedTrainingData


def preprocess_for_training(
    data_dir: str,
    out_dir: str,
    hubert_path: str | None = None,  # Ignored - using HuggingFace model for consistency
    rmvpe_path: str | None = None,
    target_sr: int = 48000,
    segment_duration: float = 4.0,
    segment_overlap: float = 0.3,
    quiet: bool = False,
) -> None:

    from src.modules.inference_rvc import load_hubert_model, extract_hubert_features
    from src.models.rmvpe import RMVPE
    from src.constants import DEFAULT_RMVPE_PATH

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    content_sr = 16000
    hop_length = 320  # HuBERT outputs at ~50Hz (320 samples at 16kHz)
    samples_per_frame = int(round(target_sr * hop_length / float(content_sr)))
    target_T = int(round(segment_duration * content_sr / hop_length))
    target_N = int(target_T * samples_per_frame)

    pre = Preprocessor(
        target_sample_rate=content_sr,
        segment_duration=segment_duration,
        segment_overlap=segment_overlap,
    )
    segmented = pre.preprocess(UnprocessedTrainingData(audio_dir_path=data_dir), quiet=quiet)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not quiet:
        print(f"Loading HuBERT model (HuggingFace transformers)...")
    hubert = load_hubert_model(device, is_half=False)

    if not quiet:
        print(f"Loading RMVPE model...")
    rmvpe_model_path = rmvpe_path if rmvpe_path else DEFAULT_RMVPE_PATH
    rmvpe = RMVPE(rmvpe_model_path, is_half=False, device=device)

    all_content: list[np.ndarray] = []

    for idx, seg in enumerate(segmented.segments):
        wav16 = seg.audio.astype(np.float32)          # 16k audio (from Preprocessor)
        wav16_np = wav16.copy()  # For RMVPE (numpy)

        wav_t = torch.from_numpy(wav16).unsqueeze(0)  # [1, n] for HuBERT

        with torch.no_grad():
            content_t = extract_hubert_features(hubert, wav_t, device, is_half=False)
            content = content_t.squeeze(0).cpu().numpy().astype(np.float32)  # [T, 768]

        f0_raw = rmvpe.infer_from_audio(wav16_np, thred=0.03)  # At hop 160
        f0 = f0_raw[::2].astype(np.float32)  # [T]

        if content.shape[0] < target_T:
            pad = target_T - content.shape[0]
            content = np.pad(content, ((0, pad), (0, 0)), mode="constant")
            f0 = np.pad(f0, (0, pad), mode="constant")
        else:
            content = content[:target_T]
            f0 = f0[:target_T]

        wav48_t = torchaudio.functional.resample(torch.from_numpy(wav16).unsqueeze(0), content_sr, target_sr)
        wav48 = wav48_t.squeeze(0).cpu().numpy().astype(np.float32)

        if wav48.shape[0] < target_N:
            wav48 = np.pad(wav48, (0, target_N - wav48.shape[0]), mode="constant")
        else:
            wav48 = wav48[:target_N]

        peak = float(np.max(np.abs(wav48))) if wav48.size else 0.0
        if peak > 0:
            wav48 = (wav48 / peak).astype(np.float32)

        file_id = f"seg_{idx:06d}"
        np.save(out_path / f"{file_id}_units.npy", content)
        np.save(out_path / f"{file_id}_f0.npy", f0)
        np.save(out_path / f"{file_id}_audio.npy", wav48)

        all_content.append(content)
        if not quiet:
            print(f"Saved {file_id}: units={content.shape}, f0={f0.shape}, audio={wav48.shape}")

    if not all_content:
        print("No features extracted, index creation skipped.")
        return

    combined = np.concatenate(all_content, axis=0).astype(np.float32)
    np.save(out_path / "content_vectors.npy", combined)

    index = IndexCreator(dimension=combined.shape[1])
    index.add(combined)
    index.save(str(out_path / "faiss_index.index"))

    print(f"Preprocess complete. Segments: {len(all_content)}. Index vectors: {combined.shape[0]}.")
