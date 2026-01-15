# src/modules/inference.py
"""Inference module for RVC voice conversion."""
from __future__ import annotations

import os
os.environ["TORCHAUDIO_USE_TORCHCODEC"] = "0"

from pathlib import Path
import numpy as np
import torch
import torchaudio

from src.models.audio_encoder import AudioEncoder
from src.models.generator import Generator
from src.models.index_creator import IndexCreator


def _apply_faiss_retrieval(
    content: np.ndarray,              # [T, D]
    index: IndexCreator,
    bank: np.ndarray,                 # [N, D]
    k: int,
    index_rate: float,
) -> np.ndarray:
    """Apply FAISS-based retrieval to blend content with training data."""
    if index_rate <= 0.0:
        return content

    sr = index.search(content.astype(np.float32), k=int(k))
    idx = sr.indices  # [T, k]
    dist = sr.distances.astype(np.float32)  # [T, k]

    # Guard against invalid indices
    idx = np.clip(idx, 0, bank.shape[0] - 1)
    neigh = bank[idx]  # [T, k, D]

    # weight by inverse distance
    w = 1.0 / (dist + 1e-6)  # [T, k]
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-6)
    retrieved = (neigh * w[:, :, None]).sum(axis=1)  # [T, D]

    return ((1.0 - index_rate) * content + index_rate * retrieved).astype(np.float32)


def run_inference(
    input_path: str,
    output_path: str,
    model_name: str,
    hubert_path: str | None = None,
    rmvpe_path: str | None = None,
    use_pitch: bool = True,  # Always True for NSF-based generation
    target_sr: int = 48000,
    device: str | None = None,
    index_path: str | None = None,
    index_data_path: str | None = None,
    index_rate: float = 0.5,
    k: int = 8,
    noise_scale: float = 0.66666,  # Match reference RVC coefficient
) -> None:
    """
    Run voice conversion inference.

    Args:
        input_path: Path to input audio file
        output_path: Path to save converted audio
        model_name: Name of the trained model (from Jar)
        hubert_path: Optional path to HuBERT model
        rmvpe_path: Optional path to RMVPE model
        use_pitch: Whether to use F0 (always True for NSF)
        target_sr: Target sample rate
        device: Device to run on
        index_path: Optional path to FAISS index for retrieval
        index_data_path: Optional path to content vectors for retrieval
        index_rate: Blend ratio for retrieval (0-1)
        k: Number of neighbors for retrieval
        noise_scale: Noise scale for sampling (0 = deterministic)
    """
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load generator (Jar object)
    g = Generator.load(model_name)
    g.synthesizer.to(dev).eval()

    # Load audio
    wav, sr = torchaudio.load(input_path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav[:1]

    # Encode to content/f0 at 16k
    enc = AudioEncoder(hubert_path=hubert_path, rmvpe_path=rmvpe_path, device=dev)
    content_t, f0_t = enc.encode_from_tensor(wav.to(torch.float32), int(sr))

    content = content_t.cpu().numpy().astype(np.float32)  # [T, 768]
    f0 = f0_t.cpu().numpy().astype(np.float32)            # [T]

    # Optional retrieval
    if index_path and index_rate > 0:
        idx_p = Path(index_path)
        if idx_p.exists():
            bank_p = Path(index_data_path) if index_data_path else (idx_p.parent / "content_vectors.npy")
            if bank_p.exists():
                bank = np.load(bank_p).astype(np.float32)
                faiss_index = IndexCreator(dimension=bank.shape[1])
                faiss_index.load(str(idx_p))
                content = _apply_faiss_retrieval(content, faiss_index, bank, k=k, index_rate=float(index_rate))
            else:
                print(f"Warning: index_data not found: {bank_p}. Skipping retrieval.")
        else:
            print(f"Warning: index not found: {idx_p}. Skipping retrieval.")

    # Convert to tensors
    content_tensor = torch.from_numpy(content).to(dev)  # [T, 768]
    f0_tensor = torch.from_numpy(f0).to(dev)            # [T]

    # Run inference using new synthesizer architecture
    with torch.inference_mode():
        # Transpose content to [1, 768, T] and add batch dim to f0
        content_input = content_tensor.unsqueeze(0).transpose(1, 2)  # [1, 768, T]
        f0_input = f0_tensor.unsqueeze(0)  # [1, T]

        # Generate audio using synthesizer inference (uses prior, no mel needed)
        y = g.synthesizer.infer(content_input, f0_input, noise_scale=noise_scale)
        y = y.squeeze(0).clamp(-1, 1).cpu()

    # Save as WAV
    out = y.unsqueeze(0)
    torchaudio.save(output_path, out, sample_rate=int(target_sr))
    print(f"Saved converted audio to: {output_path}")


def run_inference_from_features(
    content: np.ndarray,
    f0: np.ndarray,
    model_name: str,
    device: str | None = None,
    noise_scale: float = 0.66666,  # Match reference RVC coefficient
) -> np.ndarray:
    """
    Run inference directly from pre-extracted features.

    Args:
        content: [T, 768] HuBERT content features
        f0: [T] F0 in Hz
        model_name: Name of trained model
        device: Device to run on
        noise_scale: Noise scale for sampling

    Returns:
        [N] generated audio waveform
    """
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g = Generator.load(model_name)
    g.synthesizer.to(dev).eval()

    content_tensor = torch.from_numpy(content.astype(np.float32)).to(dev)
    f0_tensor = torch.from_numpy(f0.astype(np.float32)).to(dev)

    with torch.inference_mode():
        content_input = content_tensor.unsqueeze(0).transpose(1, 2)
        f0_input = f0_tensor.unsqueeze(0)
        y = g.synthesizer.infer(content_input, f0_input, noise_scale=noise_scale)
        return y.squeeze(0).cpu().numpy()
