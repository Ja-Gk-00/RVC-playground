from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np
import torch
import torchaudio

from src.models.audio_encoder import AudioEncoder
from src.models.index_creator import IndexCreator
from src.models.preprocessor import Preprocessor

CONTENT_SR: Final[int] = 16000
HOP_LENGTH: Final[int] = 320
AUDIO_EXTS: Final[set[str]] = {".wav", ".flac", ".mp3"}


def _to_mono(wav: torch.Tensor) -> torch.Tensor:
    if wav.dim() != 2:
        raise ValueError("Expected waveform shape [channels, n_samples].")
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav[:1]


def _remove_dc_and_peak_norm(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)
    x = x.astype(np.float32, copy=False)
    x = x - float(x.mean())
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0.0:
        x = x / peak
    return x.astype(np.float32, copy=False)


def preprocess_for_training(
    data_dir: str,
    out_dir: str,
    hubert_path: str | None = None,
    rmvpe_path: str | None = None,
    target_sr: int = 48000,
    quiet: bool = False,
) -> None:
    """
    Extract (units, f0, aligned waveform target) per segment and build FAISS index.

    Saved files:
      - seg_XXXXXX_units.npy  : [T, C]
      - seg_XXXXXX_f0.npy     : [T]
      - seg_XXXXXX_audio.npy  : [T*samples_per_frame]  (target_sr)
      - faiss_index.index
      - content_vectors.npy
      - meta.json
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Use Preprocessor for chunking/denoise, but do resampling via torchaudio.functional for speed.
    pre = Preprocessor(target_sample_rate=target_sr)
    encoder = AudioEncoder(hubert_path=hubert_path, rmvpe_path=rmvpe_path)

    samples_per_frame = int(round(target_sr * HOP_LENGTH / float(CONTENT_SR)))
    all_content: list[np.ndarray] = []

    audio_paths = sorted([p for p in Path(data_dir).rglob("*") if p.suffix.lower() in AUDIO_EXTS])
    if not audio_paths:
        print(f"No audio files found in {data_dir}")
        return

    seg_idx = 0
    for ap in audio_paths:
        try:
            wav, sr = torchaudio.load(str(ap))
            wav = _to_mono(wav)

            wav_np = wav.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
            wav_np = pre._denoise_audio(wav_np, int(sr))
            splits = pre._split_on_silence(wav_np, int(sr))

            for split_audio in splits:
                for chunk in pre._create_chunks(split_audio, int(sr)):
                    chunk = pre._normalize_volume(chunk)
                    chunk = _remove_dc_and_peak_norm(chunk)

                    # 1) Units/F0 extracted at CONTENT_SR
                    chunk_t = torch.from_numpy(chunk).unsqueeze(0)  # [1, n]
                    if int(sr) != CONTENT_SR:
                        chunk_16 = torchaudio.functional.resample(chunk_t, int(sr), CONTENT_SR)
                    else:
                        chunk_16 = chunk_t

                    content_t, f0_t = encoder.encode_from_tensor(chunk_16, CONTENT_SR)
                    content = content_t.detach().cpu().numpy().astype(np.float32, copy=False)  # [T, C]
                    f0 = f0_t.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)  # [T]

                    T = int(content.shape[0])
                    if T <= 0:
                        continue

                    # 2) Waveform target at target_sr aligned to frames
                    if int(sr) != int(target_sr):
                        chunk_tr = torchaudio.functional.resample(chunk_t, int(sr), int(target_sr))
                    else:
                        chunk_tr = chunk_t
                    wavt = chunk_tr.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
                    wavt = _remove_dc_and_peak_norm(wavt)

                    target_len = int(T * samples_per_frame)
                    if wavt.shape[0] < target_len:
                        wavt = np.pad(wavt, (0, target_len - wavt.shape[0]))
                    else:
                        wavt = wavt[:target_len]

                    file_id = f"seg_{seg_idx:06d}"
                    np.save(out_path / f"{file_id}_units.npy", content)
                    np.save(out_path / f"{file_id}_f0.npy", f0)
                    np.save(out_path / f"{file_id}_audio.npy", wavt)

                    all_content.append(content)
                    seg_idx += 1

        except Exception as e:
            if not quiet:
                print(f"Warning: failed to process {ap.name}: {e}")

    if not all_content:
        print("No features extracted, index creation skipped.")
        return

    combined = np.concatenate(all_content, axis=0).astype(np.float32, copy=False)
    index = IndexCreator(dimension=int(combined.shape[1]))
    index.add(combined)
    index.save(str(out_path / "faiss_index.index"))
    np.save(out_path / "content_vectors.npy", combined)

    meta = {
        "content_sr": CONTENT_SR,
        "hop_length": HOP_LENGTH,
        "target_sr": int(target_sr),
        "samples_per_frame": int(samples_per_frame),
        "content_dim": int(combined.shape[1]),
        "segments": int(seg_idx),
    }
    (out_path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if not quiet:
        print(f"Preprocess complete. Segments: {seg_idx}. Index vectors: {combined.shape[0]}.")
