# src/modules/inference_rvc.py
"""RVC inference matching reference implementation."""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


def load_hubert_model(device: torch.device, is_half: bool = False) -> torch.nn.Module:
    """Load HuBERT model for content extraction using HuggingFace transformers."""
    from transformers import HubertModel

    hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    hubert_model = hubert_model.to(device)
    if is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()


def extract_hubert_features(
    model: torch.nn.Module,
    audio: torch.Tensor,
    device: torch.device,
    is_half: bool = False,
) -> torch.Tensor:
    """Extract HuBERT content features from audio.

    Args:
        model: HuBERT model (HuggingFace transformers)
        audio: [1, T] audio at 16kHz
        device: Device to run on
        is_half: Use half precision

    Returns:
        [B, T', 768] content features
    """
    with torch.no_grad():
        feats = audio.to(device)
        if is_half:
            feats = feats.half()
        else:
            feats = feats.float()

        if feats.dim() == 2:  # stereo or [1, T]
            if feats.shape[0] > 1:
                feats = feats.mean(0, keepdim=True)
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)

        # Extract features using HuggingFace API (v2 uses layer 12)
        out = model(feats, output_hidden_states=True)
        # hidden_states is a tuple of 13 tensors (embedding + 12 layers)
        # layer 12 is index 12 (0 is embedding, 1-12 are transformer layers)
        feats = out.hidden_states[12]  # [B, T, 768]

        return feats


def extract_f0_rmvpe(
    audio: np.ndarray,
    sr: int,
    device: torch.device,
    f0_up_key: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract F0 using RMVPE.

    Args:
        audio: Audio samples
        sr: Sample rate (should be 16kHz)
        device: Device
        f0_up_key: Pitch shift in semitones

    Returns:
        f0_coarse: Quantized F0 for embedding (1-255)
        f0: Raw F0 in Hz
    """
    from src.models.rmvpe import RMVPE
    from src.constants import DEFAULT_RMVPE_PATH

    # Load RMVPE
    rmvpe = RMVPE(DEFAULT_RMVPE_PATH, is_half=False, device=device)

    # Extract F0
    f0 = rmvpe.infer_from_audio(audio, thred=0.03)

    # Apply pitch shift
    f0 *= pow(2, f0_up_key / 12)

    # Quantize to mel scale for embedding
    f0_min = 50
    f0_max = 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = np.rint(f0_mel).astype(np.int32)

    return f0_coarse, f0


def load_synthesizer(
    model_path: str,
    device: torch.device,
    is_half: bool = False,
) -> torch.nn.Module:
    """Load RVC synthesizer model matching reference implementation.

    Args:
        model_path: Path to .pth model file
        device: Device to load on
        is_half: Use half precision

    Returns:
        Synthesizer model
    """
    from src.models.synthesizer_rvc import SynthesizerTrnMs768NSFsid, SynthesizerTrnMs256NSFsid

    # Load checkpoint
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    # Get version and f0 flag (reference style)
    version = ckpt.get("version", "v2")
    if_f0 = ckpt.get("f0", 1)

    # Get state dict - reference uses "weight", pretrained uses "model"
    if "weight" in ckpt:
        state_dict = ckpt["weight"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    # Get config - reference uses list format that gets unpacked
    config = ckpt.get("config", None)

    if config is not None and isinstance(config, (list, tuple)):
        # Reference format: config is a list of positional args
        # [spec_channels, segment_size, inter_channels, hidden_channels,
        #  filter_channels, n_heads, n_layers, kernel_size, p_dropout,
        #  resblock, resblock_kernel_sizes, resblock_dilation_sizes,
        #  upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
        #  spk_embed_dim, gin_channels, sr]

        # Update spk_embed_dim from actual weights if available
        if "emb_g.weight" in state_dict:
            config = list(config)
            config[-3] = state_dict["emb_g.weight"].shape[0]  # spk_embed_dim

        # Select model class based on version
        if version == "v2":
            model = SynthesizerTrnMs768NSFsid(*config, is_half=is_half)
        else:
            model = SynthesizerTrnMs256NSFsid(*config, is_half=is_half)
    else:
        # Fallback: use defaults for v2 48k
        model = SynthesizerTrnMs768NSFsid(
            spec_channels=1025,
            segment_size=32,
            inter_channels=192,
            hidden_channels=192,
            filter_channels=768,
            n_heads=2,
            n_layers=6,
            kernel_size=3,
            p_dropout=0,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[12, 10, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[24, 20, 4, 4],
            spk_embed_dim=109,
            gin_channels=256,
            sr=48000,
            is_half=is_half,
        )

    # Delete enc_q (not needed for inference, matches reference)
    if hasattr(model, 'enc_q'):
        del model.enc_q

    # Load weights with strict=False (reference style)
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    model.to(device)

    if is_half:
        model.half()

    return model


def resample_audio(audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """High-quality audio resampling using Kaiser window.

    Better quality than default torchaudio.functional.resample for voice.
    """
    if orig_sr == target_sr:
        return audio

    # Use Kaiser window for better quality
    resampled = torchaudio.functional.resample(
        audio, orig_sr, target_sr,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )
    return resampled


def run_inference_rvc(
    input_path: str,
    model_path: str,
    output_path: str,
    speaker_id: int = 0,
    f0_up_key: int = 0,
    index_path: Optional[str] = None,
    index_rate: float = 0.5,
    device: Optional[str] = None,
    chunk_seconds: float = 10.0,
    protect: float = 0.33,
) -> None:
    """Run RVC inference matching reference implementation.

    Args:
        input_path: Path to input audio
        model_path: Path to RVC model (.pth)
        output_path: Path to save output
        speaker_id: Speaker ID for embedding
        f0_up_key: Pitch shift in semitones
        index_path: Path to FAISS index (optional)
        index_rate: Index mixing rate
        device: Device to use
        chunk_seconds: Process audio in chunks of this many seconds (saves memory)
        protect: Protect voiceless consonants (0-0.5, higher = more protection)
    """
    # Setup device
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    print(f"Using device: {dev}")

    # Load audio
    audio, sr = torchaudio.load(input_path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Resample to 16kHz for HuBERT using high-quality resampling
    if sr != 16000:
        audio = resample_audio(audio, sr, 16000)

    audio_np = audio.squeeze(0).numpy()
    total_samples = len(audio_np)
    duration = total_samples / 16000

    print(f"Audio duration: {duration:.1f}s")

    # Load models
    print("Loading HuBERT...")
    hubert = load_hubert_model(dev, is_half=False)

    print("Loading synthesizer...")
    synthesizer = load_synthesizer(model_path, dev)

    # Load RMVPE once
    from src.models.rmvpe import RMVPE
    from src.constants import DEFAULT_RMVPE_PATH
    rmvpe = RMVPE(DEFAULT_RMVPE_PATH, is_half=False, device=dev)

    # Determine chunk size (in 16kHz samples)
    chunk_samples = int(chunk_seconds * 16000)
    overlap_samples = int(0.5 * 16000)  # 0.5 second overlap

    # If audio is short enough, process in one go
    if total_samples <= chunk_samples:
        audio_out = _process_chunk(
            audio, audio_np, hubert, synthesizer, rmvpe, dev,
            speaker_id, f0_up_key, index_path, index_rate, protect
        )
    else:
        # Process in overlapping chunks
        print(f"Processing in {chunk_seconds}s chunks to save memory...")
        audio_chunks = []
        start = 0

        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            chunk_audio = audio[:, start:end]
            chunk_np = audio_np[start:end]

            print(f"  Processing chunk {start/16000:.1f}s - {end/16000:.1f}s...")

            chunk_out = _process_chunk(
                chunk_audio, chunk_np, hubert, synthesizer, rmvpe, dev,
                speaker_id, f0_up_key, index_path, index_rate, protect
            )
            audio_chunks.append(chunk_out)

            # Move to next chunk with overlap
            start = end - overlap_samples
            if start >= total_samples - overlap_samples:
                break

        # Crossfade chunks together
        audio_out = _crossfade_chunks(audio_chunks, overlap_samples * 3)  # 3x for 48kHz output

    # Normalize
    max_val = np.abs(audio_out).max()
    if max_val > 0:
        audio_out = audio_out / max_val * 0.95

    # Save as [1, T] tensor for torchaudio
    # Output is always mono at 48kHz (RVC native rate)
    audio_tensor = torch.from_numpy(audio_out).unsqueeze(0).float()
    torchaudio.save(output_path, audio_tensor, 48000)
    print(f"Saved to {output_path} (mono, 48kHz)")


def _process_chunk(
    audio: torch.Tensor,
    audio_np: np.ndarray,
    hubert: torch.nn.Module,
    synthesizer: torch.nn.Module,
    rmvpe,
    dev: torch.device,
    speaker_id: int,
    f0_up_key: int,
    index_path: Optional[str],
    index_rate: float,
    protect: float = 0.33,
) -> np.ndarray:
    """Process a single audio chunk.

    Args:
        protect: Protection ratio for voiceless consonants (0-0.5).
                Higher values preserve more of original timbre in unvoiced regions.
    """
    # Extract HuBERT features
    feats = extract_hubert_features(hubert, audio, dev, is_half=False)

    # Store original features before index mixing (for protect parameter)
    # Protect preserves original timbre in unvoiced regions
    feats_original = feats.clone()

    # Apply index if provided
    if index_path and Path(index_path).exists() and index_rate > 0:
        try:
            import faiss
            index = faiss.read_index(index_path)
            big_npy = index.reconstruct_n(0, index.ntotal)

            npy = feats[0].cpu().numpy().astype("float32")
            score, ix = index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(dev) * index_rate
                + (1 - index_rate) * feats
            )
        except Exception as e:
            warnings.warn(f"Failed to apply index: {e}")

    # Interpolate features by 2x (both original and processed)
    feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
    feats_original = F.interpolate(feats_original.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

    # Extract F0 using passed rmvpe
    f0 = rmvpe.infer_from_audio(audio_np, thred=0.03)
    f0 *= pow(2, f0_up_key / 12)

    # Apply median filter to smooth F0 (reduces crackling artifacts)
    from scipy.ndimage import median_filter
    voiced_mask = f0 > 0
    if np.sum(voiced_mask) > 5:
        f0_voiced = f0.copy()
        f0_voiced[~voiced_mask] = 0
        f0_smoothed = median_filter(f0_voiced, size=3)
        f0[voiced_mask] = f0_smoothed[voiced_mask]

    # Quantize F0
    f0_min, f0_max = 50, 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = np.rint(f0_mel).astype(np.int32)

    # Align lengths
    p_len = min(feats.shape[1], len(f0))
    feats = feats[:, :p_len, :]
    feats_original = feats_original[:, :p_len, :]
    f0_coarse = f0_coarse[:p_len]
    f0 = f0[:p_len]

    # Apply protect parameter - blend original features in unvoiced regions
    # This preserves consonants and breathiness
    if protect > 0 and protect < 0.5:
        # Create mask for unvoiced regions (where f0 is 0)
        pitchf_tensor = torch.from_numpy(f0).unsqueeze(0).unsqueeze(2).to(dev)  # [1, T, 1]

        # Protect masks original features where pitch is near zero
        # protect=0.5 means full protection (use original), protect=0 means no protection
        protect_mask = (pitchf_tensor < 0.001).float()  # [1, T, 1]

        # Blend: in unvoiced regions, use mix of original and processed features
        # Higher protect value = more original features preserved
        feats_original = feats_original.to(dev)
        feats = feats * (1 - protect_mask * protect * 2) + feats_original * (protect_mask * protect * 2)

    # Prepare tensors
    phone = feats.to(dev)
    phone_lengths = torch.tensor([p_len], device=dev).long()
    pitch = torch.from_numpy(f0_coarse).unsqueeze(0).to(dev).long()
    pitchf = torch.from_numpy(f0).unsqueeze(0).to(dev).float()
    sid = torch.tensor([speaker_id], device=dev).long()

    # Run inference
    with torch.no_grad():
        audio_out, _, _ = synthesizer.infer(phone, phone_lengths, pitch, pitchf, sid)

    # Clear GPU cache
    if dev.type == 'cuda':
        torch.cuda.empty_cache()

    return audio_out[0, 0].data.cpu().float().numpy()


def _crossfade_chunks(chunks: list, overlap_samples: int) -> np.ndarray:
    """Crossfade overlapping audio chunks."""
    if len(chunks) == 1:
        return chunks[0]

    # Create output buffer
    total_len = sum(len(c) for c in chunks) - overlap_samples * (len(chunks) - 1)
    output = np.zeros(total_len, dtype=np.float32)

    # Crossfade ramp
    fade_in = np.linspace(0, 1, overlap_samples, dtype=np.float32)
    fade_out = np.linspace(1, 0, overlap_samples, dtype=np.float32)

    pos = 0
    for i, chunk in enumerate(chunks):
        chunk_len = len(chunk)

        if i == 0:
            # First chunk: just copy, apply fade out at end
            output[:chunk_len - overlap_samples] = chunk[:-overlap_samples]
            output[chunk_len - overlap_samples:chunk_len] = chunk[-overlap_samples:] * fade_out
            pos = chunk_len - overlap_samples
        else:
            # Apply fade in at start, add to existing (which has fade out)
            output[pos:pos + overlap_samples] += chunk[:overlap_samples] * fade_in
            # Copy rest of chunk
            if i < len(chunks) - 1:
                output[pos + overlap_samples:pos + chunk_len - overlap_samples] = chunk[overlap_samples:-overlap_samples]
                output[pos + chunk_len - overlap_samples:pos + chunk_len] = chunk[-overlap_samples:] * fade_out
                pos += chunk_len - overlap_samples
            else:
                # Last chunk: no fade out at end
                output[pos + overlap_samples:pos + chunk_len] = chunk[overlap_samples:]

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input audio path")
    parser.add_argument("--model", required=True, help="RVC model path (.pth)")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--sid", type=int, default=0, help="Speaker ID")
    parser.add_argument("--f0_up_key", type=int, default=0, help="Pitch shift")
    parser.add_argument("--index", default=None, help="Index path")
    parser.add_argument("--index_rate", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--chunk_seconds", type=float, default=10.0,
                        help="Process in chunks of N seconds (lower = less memory)")
    parser.add_argument("--protect", type=float, default=0.33,
                        help="Protect voiceless consonants (0-0.5, higher = more protection)")

    args = parser.parse_args()

    run_inference_rvc(
        input_path=args.input,
        model_path=args.model,
        output_path=args.output,
        speaker_id=args.sid,
        f0_up_key=args.f0_up_key,
        index_path=args.index,
        index_rate=args.index_rate,
        device=args.device,
        chunk_seconds=args.chunk_seconds,
        protect=args.protect,
    )
