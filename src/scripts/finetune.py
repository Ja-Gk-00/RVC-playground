#!/usr/bin/env python3
"""
RVC Fine-tuning Script

Fine-tune a pretrained RVC model on your voice data.

Usage:
    python -m src.scripts.finetune --data_dir ./my_voice_data --pretrained ./pretrained.pth --output ./my_model.pth

Prepare your data:
    1. Create a folder with your voice recordings (WAV files, mono, any sample rate)
    2. Each file should be 3-15 seconds of clean speech
    3. Aim for 10-50 minutes total audio for best results
"""
import argparse
import os
from pathlib import Path
from typing import Optional
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler
import torchaudio
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceDataset(Dataset):
    """Dataset for fine-tuning RVC on voice data."""

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 48000,
        segment_length: int = 32 * 480,  # 32 frames * hop_length
        hubert_model=None,
        rmvpe_model=None,
        device: torch.device = torch.device("cpu"),
        features_dir: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.hubert = hubert_model
        self.rmvpe = rmvpe_model
        self.device = device
        self.features_dir = Path(features_dir) if features_dir else None

        # Find all audio files
        self.audio_files = list(self.data_dir.glob("*.wav")) + \
                          list(self.data_dir.glob("*.mp3")) + \
                          list(self.data_dir.glob("*.flac"))

        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {data_dir}")

        logger.info(f"Found {len(self.audio_files)} audio files")

        # Preprocess all files or load from cache
        self.samples = []
        self.all_features = []  # For FAISS index

        if self.features_dir and self._check_cached_features():
            self._load_cached_features()
        else:
            self._preprocess_all()
            if self.features_dir:
                self._save_cached_features()

    def _check_cached_features(self) -> bool:
        """Check if valid cached features exist."""
        if not self.features_dir or not self.features_dir.exists():
            return False

        cache_file = self.features_dir / "features_cache.pt"
        if not cache_file.exists():
            return False

        # Check if cache is newer than all audio files
        cache_mtime = cache_file.stat().st_mtime
        for audio_file in self.audio_files:
            if audio_file.stat().st_mtime > cache_mtime:
                logger.info(f"Audio file {audio_file.name} is newer than cache, will reprocess")
                return False

        logger.info(f"Found valid cached features at {cache_file}")
        return True

    def _load_cached_features(self):
        """Load preprocessed features from cache."""
        cache_file = self.features_dir / "features_cache.pt"
        logger.info(f"Loading cached features from {cache_file}")

        cache = torch.load(cache_file, map_location="cpu", weights_only=False)
        self.samples = cache["samples"]
        self.all_features = cache["all_features"]

        logger.info(f"Loaded {len(self.samples)} samples from cache")

    def _save_cached_features(self):
        """Save preprocessed features to cache."""
        self.features_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.features_dir / "features_cache.pt"

        cache = {
            "samples": self.samples,
            "all_features": self.all_features,
        }
        torch.save(cache, cache_file)
        logger.info(f"Saved features cache to {cache_file}")

    def _preprocess_all(self):
        """Preprocess all audio files and extract features."""
        from src.modules.inference_rvc import extract_hubert_features

        logger.info("Preprocessing audio files...")

        failed_files = []
        first_error = None

        for audio_path in tqdm(self.audio_files, desc="Processing"):
            try:
                # Load audio
                audio, sr = torchaudio.load(str(audio_path))
                if audio.shape[0] > 1:
                    audio = audio.mean(dim=0, keepdim=True)

                # Resample to target rate
                if sr != self.sample_rate:
                    audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

                # Resample to 16kHz for feature extraction
                audio_16k = torchaudio.functional.resample(audio, self.sample_rate, 16000)
                audio_16k_np = audio_16k.squeeze(0).numpy()

                # Extract HuBERT features
                with torch.no_grad():
                    feats = extract_hubert_features(self.hubert, audio_16k, self.device)
                    # Interpolate 2x
                    feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
                    feats = feats.squeeze(0).cpu()  # [T, 768]

                # Extract F0 using the passed RMVPE model
                f0_coarse, f0 = self._extract_f0(audio_16k_np)

                # Compute spectrogram
                spec = self._compute_spectrogram(audio.squeeze(0))  # [freq, T]

                # Align lengths
                min_len = min(feats.shape[0], len(f0), spec.shape[1])
                if min_len < 10:  # Skip very short segments
                    logger.warning(f"Skipping {audio_path}: too short ({min_len} frames)")
                    continue

                feats = feats[:min_len]
                f0_coarse = f0_coarse[:min_len]
                f0 = f0[:min_len]
                spec = spec[:, :min_len]
                audio_segment = audio[:, :min_len * 480]  # hop_length = 480

                # Store sample
                self.samples.append({
                    'phone': feats,  # [T, 768]
                    'pitch': torch.from_numpy(f0_coarse).long(),  # [T]
                    'pitchf': torch.from_numpy(f0).float(),  # [T]
                    'spec': spec,  # [freq, T]
                    'wave': audio_segment.squeeze(0),  # [T_audio]
                    'path': str(audio_path),
                })

                # Collect features for FAISS index
                self.all_features.append(feats.numpy())

            except Exception as e:
                failed_files.append((audio_path, str(e)))
                if first_error is None:
                    first_error = e
                    import traceback
                    first_traceback = traceback.format_exc()

        logger.info(f"Successfully preprocessed {len(self.samples)} files")
        logger.info(f"Collected {len(self.all_features)} feature arrays for FAISS index")

        if len(self.samples) == 0:
            # Print detailed error info
            print("\n" + "="*60)
            print("ERROR: All audio files failed to process!")
            print("="*60)
            print(f"\nFailed files ({len(failed_files)}):")
            for path, err in failed_files[:5]:  # Show first 5
                print(f"  - {path.name}: {err}")
            if len(failed_files) > 5:
                print(f"  ... and {len(failed_files) - 5} more")

            if first_error:
                print(f"\nFirst error traceback:")
                print(first_traceback)

            raise ValueError(
                f"No audio files were successfully processed from {self.data_dir}. "
                f"First error: {first_error}"
            )

    def create_faiss_index(self, output_path: str):
        """Create FAISS index from collected features."""
        try:
            import faiss
        except ImportError:
            logger.warning("FAISS not installed, skipping index creation. Install with: pip install faiss-cpu")
            return None

        if not self.all_features:
            logger.warning("No features collected for FAISS index")
            return None

        # Concatenate all features
        all_feats = np.concatenate(self.all_features, axis=0).astype(np.float32)
        logger.info(f"Creating FAISS index from {all_feats.shape[0]} vectors of dim {all_feats.shape[1]}")

        # Create index (IVF for speed if large, flat for small datasets)
        dim = all_feats.shape[1]  # 768
        if all_feats.shape[0] < 10000:
            # Small dataset: use flat index
            index = faiss.IndexFlatL2(dim)
        else:
            # Larger dataset: use IVF
            nlist = min(100, all_feats.shape[0] // 100)
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            index.train(all_feats)

        index.add(all_feats)

        # Save index
        faiss.write_index(index, output_path)
        logger.info(f"Saved FAISS index to {output_path}")
        return output_path

    def _extract_f0(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract F0 using RMVPE model with median filtering."""
        from scipy.ndimage import median_filter

        # Extract F0
        f0 = self.rmvpe.infer_from_audio(audio, thred=0.03)

        # Apply median filter to smooth F0 (reduces crackling)
        # Only filter voiced regions
        voiced_mask = f0 > 0
        if np.sum(voiced_mask) > 5:
            f0_voiced = f0.copy()
            f0_voiced[~voiced_mask] = 0
            f0_smoothed = median_filter(f0_voiced, size=3)
            f0[voiced_mask] = f0_smoothed[voiced_mask]

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

    def _compute_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute spectrogram for training."""
        n_fft = 2048
        hop_length = 480
        win_length = 2048

        # Ensure audio is 1D
        if audio.dim() > 1:
            audio = audio.squeeze()

        # Pad audio (need 2D for reflect mode, so add and remove batch dim)
        pad_amount = int((n_fft - hop_length) / 2)
        audio = audio.unsqueeze(0)  # [1, T]
        audio = F.pad(audio, (pad_amount, pad_amount), mode='reflect')
        audio = audio.squeeze(0)  # [T]

        # STFT
        spec = torch.stft(
            audio, n_fft, hop_length=hop_length, win_length=win_length,
            window=torch.hann_window(win_length, device=audio.device),
            return_complex=True,
        )
        spec = torch.abs(spec)  # [freq, T]

        return spec

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Random segment
        max_start = sample['phone'].shape[0] - (self.segment_length // 480)
        if max_start > 0:
            start = np.random.randint(0, max_start)
            end = start + (self.segment_length // 480)

            phone = sample['phone'][start:end]
            pitch = sample['pitch'][start:end]
            pitchf = sample['pitchf'][start:end]
            spec = sample['spec'][:, start:end]
            wave = sample['wave'][start * 480:end * 480]
        else:
            phone = sample['phone']
            pitch = sample['pitch']
            pitchf = sample['pitchf']
            spec = sample['spec']
            wave = sample['wave']

        return {
            'phone': phone,
            'phone_length': phone.shape[0],
            'pitch': pitch,
            'pitchf': pitchf,
            'spec': spec,
            'spec_length': spec.shape[1],
            'wave': wave,
            'sid': 0,  # Single speaker
        }


def collate_fn(batch):
    """Collate batch with padding."""
    # Find max lengths
    max_phone_len = max(b['phone_length'] for b in batch)
    max_spec_len = max(b['spec_length'] for b in batch)
    max_wave_len = max_spec_len * 480

    phones = []
    phone_lengths = []
    pitches = []
    pitchfs = []
    specs = []
    spec_lengths = []
    waves = []
    sids = []

    for b in batch:
        # Pad phone
        phone = b['phone']
        pad_len = max_phone_len - phone.shape[0]
        if pad_len > 0:
            phone = F.pad(phone, (0, 0, 0, pad_len))
        phones.append(phone)
        phone_lengths.append(b['phone_length'])

        # Pad pitch
        pitch = b['pitch']
        if pad_len > 0:
            pitch = F.pad(pitch, (0, pad_len))
        pitches.append(pitch)

        # Pad pitchf
        pitchf = b['pitchf']
        if pad_len > 0:
            pitchf = F.pad(pitchf, (0, pad_len))
        pitchfs.append(pitchf)

        # Pad spec
        spec = b['spec']
        spec_pad = max_spec_len - spec.shape[1]
        if spec_pad > 0:
            spec = F.pad(spec, (0, spec_pad))
        specs.append(spec)
        spec_lengths.append(b['spec_length'])

        # Pad wave
        wave = b['wave']
        wave_pad = max_wave_len - wave.shape[0]
        if wave_pad > 0:
            wave = F.pad(wave, (0, wave_pad))
        waves.append(wave)

        sids.append(b['sid'])

    return {
        'phone': torch.stack(phones),
        'phone_lengths': torch.tensor(phone_lengths),
        'pitch': torch.stack(pitches),
        'pitchf': torch.stack(pitchfs),
        'spec': torch.stack(specs),
        'spec_lengths': torch.tensor(spec_lengths),
        'wave': torch.stack(waves),
        'sid': torch.tensor(sids),
    }


class PreprocessedDataset(Dataset):
    """
    Dataset that loads from preprocess.py output directory.

    Expected files in preprocessed_dir:
        seg_XXXXXX_units.npy   [T, 768]  - HuBERT features at hop 320/16kHz
        seg_XXXXXX_f0.npy      [T]       - Raw F0 values
        seg_XXXXXX_audio.npy   [N]       - Audio at 48kHz
        faiss_index.index               - FAISS index (optional)
    """

    def __init__(
        self,
        preprocessed_dir: str,
        segment_frames: int = 32,  # Number of frames per segment (at 480 hop)
        hop_length: int = 480,     # Hop length at 48kHz
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.segment_frames = segment_frames
        self.hop_length = hop_length
        self.segment_samples = segment_frames * hop_length

        # Find all segment files
        self.segments = sorted(self.preprocessed_dir.glob("seg_*_units.npy"))
        if not self.segments:
            raise ValueError(f"No preprocessed segments found in {preprocessed_dir}")

        logger.info(f"Found {len(self.segments)} preprocessed segments")

        # Load and prepare all segments
        self.samples = []
        self._load_all_segments()

    def _quantize_f0(self, f0: np.ndarray) -> np.ndarray:
        """Quantize F0 to mel scale (1-255)."""
        f0_min, f0_max = 50, 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)

        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        return np.rint(f0_mel).astype(np.int32)

    def _compute_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute spectrogram for training."""
        n_fft = 2048
        hop_length = 480
        win_length = 2048

        if audio.dim() > 1:
            audio = audio.squeeze()

        pad_amount = int((n_fft - hop_length) / 2)
        audio = audio.unsqueeze(0)
        audio = F.pad(audio, (pad_amount, pad_amount), mode='reflect')
        audio = audio.squeeze(0)

        spec = torch.stft(
            audio, n_fft, hop_length=hop_length, win_length=win_length,
            window=torch.hann_window(win_length, device=audio.device),
            return_complex=True,
        )
        return torch.abs(spec)

    def _load_all_segments(self):
        """Load all preprocessed segments."""
        for units_path in tqdm(self.segments, desc="Loading preprocessed"):
            seg_id = units_path.stem.replace("_units", "")
            f0_path = self.preprocessed_dir / f"{seg_id}_f0.npy"
            audio_path = self.preprocessed_dir / f"{seg_id}_audio.npy"

            if not f0_path.exists() or not audio_path.exists():
                logger.warning(f"Skipping {seg_id}: missing f0 or audio")
                continue

            try:
                # Load features
                units = np.load(units_path)  # [T, 768] at hop 320/16kHz
                f0_raw = np.load(f0_path)    # [T]
                audio = np.load(audio_path)  # [N] at 48kHz

                # Interpolate units by 2x (hop 320 -> hop 160 at 16kHz = hop 480 at 48kHz)
                units_t = torch.from_numpy(units).unsqueeze(0).permute(0, 2, 1)  # [1, 768, T]
                units_t = F.interpolate(units_t, scale_factor=2, mode='linear', align_corners=False)
                units_t = units_t.permute(0, 2, 1).squeeze(0)  # [T*2, 768]

                # Interpolate F0 by 2x
                f0_t = torch.from_numpy(f0_raw).unsqueeze(0).unsqueeze(0)  # [1, 1, T]
                f0_t = F.interpolate(f0_t, scale_factor=2, mode='linear', align_corners=False)
                f0_interp = f0_t.squeeze().numpy()  # [T*2]

                # Quantize F0
                f0_coarse = self._quantize_f0(f0_interp)

                # Convert audio to tensor and compute spec
                audio_t = torch.from_numpy(audio).float()
                spec = self._compute_spectrogram(audio_t)  # [freq, T_spec]

                # Align lengths (spec frames should match interpolated feature frames)
                min_len = min(units_t.shape[0], len(f0_coarse), spec.shape[1])
                if min_len < 10:
                    logger.warning(f"Skipping {seg_id}: too short ({min_len} frames)")
                    continue

                units_t = units_t[:min_len]
                f0_coarse = f0_coarse[:min_len]
                f0_interp = f0_interp[:min_len]
                spec = spec[:, :min_len]
                audio_t = audio_t[:min_len * self.hop_length]

                self.samples.append({
                    'phone': units_t.float(),
                    'pitch': torch.from_numpy(f0_coarse).long(),
                    'pitchf': torch.from_numpy(f0_interp).float(),
                    'spec': spec,
                    'wave': audio_t,
                })

            except Exception as e:
                logger.warning(f"Error loading {seg_id}: {e}")
                continue

        logger.info(f"Loaded {len(self.samples)} valid segments")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Random segment within the sample
        max_start = sample['phone'].shape[0] - self.segment_frames
        if max_start > 0:
            start = np.random.randint(0, max_start)
            end = start + self.segment_frames

            phone = sample['phone'][start:end]
            pitch = sample['pitch'][start:end]
            pitchf = sample['pitchf'][start:end]
            spec = sample['spec'][:, start:end]
            wave = sample['wave'][start * self.hop_length:end * self.hop_length]
        else:
            phone = sample['phone']
            pitch = sample['pitch']
            pitchf = sample['pitchf']
            spec = sample['spec']
            wave = sample['wave']

        return {
            'phone': phone,
            'phone_length': phone.shape[0],
            'pitch': pitch,
            'pitchf': pitchf,
            'spec': spec,
            'spec_length': spec.shape[1],
            'wave': wave,
            'sid': 0,
        }

    def get_index_path(self) -> Optional[str]:
        """Return path to FAISS index if it exists."""
        index_path = self.preprocessed_dir / "faiss_index.index"
        if index_path.exists():
            return str(index_path)
        return None


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """KL divergence loss."""
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    loss = kl / torch.sum(z_mask)
    return loss


def mel_spectrogram(audio, n_fft=2048, hop_length=480, win_length=2048, n_mels=128, sr=48000):
    """Compute mel spectrogram."""
    # STFT
    audio = F.pad(audio, (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), mode='reflect')
    spec = torch.stft(
        audio, n_fft, hop_length=hop_length, win_length=win_length,
        window=torch.hann_window(win_length, device=audio.device),
        return_complex=True,
    )
    spec = torch.abs(spec)

    # Mel filterbank
    mel_basis = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=0,
        f_max=sr // 2,
        n_mels=n_mels,
        sample_rate=sr,
    ).T.to(audio.device)

    mel = torch.matmul(mel_basis, spec)
    mel = torch.log(torch.clamp(mel, min=1e-5))

    return mel


def train_step(model, batch, optimizer, scaler, device, c_mel=45, c_kl=0.1):
    """Single training step (without GAN - for compatibility)."""
    return train_step_gan(
        model, None, batch, optimizer, None, scaler, device,
        c_mel=c_mel, c_kl=c_kl, c_fm=0, c_adv=0, use_gan=False
    )


def train_step_gan(
    model, discriminator, batch, optim_g, optim_d, scaler, device,
    c_mel=45, c_kl=0.1, c_fm=2, c_adv=1, use_gan=True, use_fp16=False,
    compute_metrics=False,
):
    """
    Single training step with GAN losses.

    Args:
        model: Generator (synthesizer)
        discriminator: Combined MultiPeriodDiscriminatorV2
        batch: Training batch
        optim_g: Generator optimizer
        optim_d: Discriminator optimizer
        scaler: GradScaler for mixed precision
        device: Device
        c_mel: Mel loss coefficient
        c_kl: KL loss coefficient
        c_fm: Feature matching loss coefficient
        c_adv: Adversarial loss coefficient
        use_gan: Whether to use GAN training
        use_fp16: Use mixed precision (FP16)
        compute_metrics: Compute quality metrics (MCD, discriminator accuracy)
    """
    from src.models.commons import slice_segments

    # Move to device
    phone = batch['phone'].to(device)
    phone_lengths = batch['phone_lengths'].to(device)
    pitch = batch['pitch'].to(device)
    pitchf = batch['pitchf'].to(device)
    spec = batch['spec'].to(device)
    spec_lengths = batch['spec_lengths'].to(device)
    wave = batch['wave'].to(device)
    sid = batch['sid'].to(device)

    losses = {}

    # ========== Generator Forward ==========
    with torch.amp.autocast('cuda', enabled=use_fp16 and device.type == 'cuda'):
        y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = model(
            phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid
        )

        # Slice ground truth audio to match generated
        y = slice_segments(
            wave.unsqueeze(1),  # [B, 1, T]
            ids_slice * 480,  # Convert frame indices to sample indices
            model.segment_size * 480
        )

        # Ensure shapes match
        min_samples = min(y.shape[-1], y_hat.shape[-1])
        y = y[..., :min_samples]
        y_hat = y_hat[..., :min_samples]

        # Mel loss
        y_hat_mel = mel_spectrogram(y_hat.squeeze(1))
        y_mel = mel_spectrogram(y.squeeze(1))

        min_len = min(y_mel.shape[-1], y_hat_mel.shape[-1])
        y_mel = y_mel[..., :min_len]
        y_hat_mel = y_hat_mel[..., :min_len]

        loss_mel = F.l1_loss(y_mel, y_hat_mel) * c_mel

        # KL loss
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * c_kl

    losses['loss_mel'] = loss_mel.item()
    losses['loss_kl'] = loss_kl.item()

    # Compute MCD if metrics requested
    if compute_metrics:
        from src.utils.metrics import compute_mcd
        with torch.no_grad():
            mcd_value = compute_mcd(y_hat_mel, y_mel)
            losses['mcd'] = mcd_value

    # ========== GAN Training ==========
    if use_gan and discriminator is not None:
        from src.models.discriminator import (
            discriminator_loss, generator_loss, feature_matching_loss
        )

        # ---------- Train Discriminator ----------
        optim_d.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_fp16 and device.type == 'cuda'):
            # Combined discriminator (includes scale + period discriminators)
            y_d_rs, y_d_gs, _, _ = discriminator(y, y_hat.detach())
            loss_d, _, _ = discriminator_loss(y_d_rs, y_d_gs)

        scaler.scale(loss_d).backward()
        scaler.unscale_(optim_d)
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
        scaler.step(optim_d)

        losses['loss_d'] = loss_d.item()

        # Compute discriminator accuracy if metrics requested
        if compute_metrics:
            from src.utils.metrics import compute_discriminator_accuracy
            with torch.no_grad():
                d_acc = compute_discriminator_accuracy(y_d_rs, y_d_gs)
                losses['d_acc_real'] = d_acc['d_acc_real']
                losses['d_acc_fake'] = d_acc['d_acc_fake']

        # ---------- Train Generator with GAN losses ----------
        optim_g.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_fp16 and device.type == 'cuda'):
            # Recompute discriminator outputs (without detach for generator grads)
            y_d_rs, y_d_gs, fmap_rs, fmap_gs = discriminator(y, y_hat)

            # Generator adversarial loss
            loss_adv, _ = generator_loss(y_d_gs)
            loss_adv = loss_adv * c_adv

            # Feature matching loss
            loss_fm = feature_matching_loss(fmap_rs, fmap_gs) * c_fm

            # Total generator loss
            loss_g = loss_mel + loss_kl + loss_adv + loss_fm

        scaler.scale(loss_g).backward()
        scaler.unscale_(optim_g)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optim_g)
        scaler.update()

        losses['loss_adv'] = loss_adv.item()
        losses['loss_fm'] = loss_fm.item()
        losses['loss'] = loss_g.item()

    else:
        # No GAN - just mel + KL
        optim_g.zero_grad()
        loss = loss_mel + loss_kl
        scaler.scale(loss).backward()
        scaler.unscale_(optim_g)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optim_g)
        scaler.update()
        losses['loss'] = loss.item()

    return losses


def load_synthesizer_for_training(model_path: str, device: torch.device):
    """Load synthesizer with enc_q intact for training.

    Unlike inference loading, this preserves enc_q weights from the pretrained model.
    """
    from src.models.synthesizer_rvc import SynthesizerTrnMs768NSFsid, SynthesizerTrnMs256NSFsid

    # Load checkpoint
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    # Get state dict - reference uses "model", our saved models use "weight"
    if "weight" in ckpt:
        state_dict = ckpt["weight"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    # Check if enc_q is in the checkpoint
    has_enc_q = any(k.startswith("enc_q") for k in state_dict.keys())
    logger.info(f"Pretrained model {'includes' if has_enc_q else 'does not include'} enc_q weights")

    # Get version
    version = ckpt.get("version", "v2")

    # Get config
    config = ckpt.get("config", None)

    if config is not None and isinstance(config, (list, tuple)):
        # Reference format: config is a list of positional args
        if "emb_g.weight" in state_dict:
            config = list(config)
            config[-3] = state_dict["emb_g.weight"].shape[0]  # spk_embed_dim

        if version == "v2":
            model = SynthesizerTrnMs768NSFsid(*config, is_half=False)
        else:
            model = SynthesizerTrnMs256NSFsid(*config, is_half=False)
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
            is_half=False,
        )

    # Load weights - keep enc_q if present in checkpoint!
    # Don't delete enc_q like inference does
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        logger.warning(f"Missing keys in pretrained: {len(missing)}")
        # Only show first few
        for k in missing[:5]:
            logger.warning(f"  - {k}")
        if len(missing) > 5:
            logger.warning(f"  ... and {len(missing) - 5} more")

    if unexpected:
        logger.warning(f"Unexpected keys in pretrained: {len(unexpected)}")
        for k in unexpected[:5]:
            logger.warning(f"  - {k}")

    # CRITICAL: Check if enc_q was loaded - if not, the VAE won't train properly!
    enc_q_missing = [k for k in missing if k.startswith("enc_q")]
    if enc_q_missing:
        logger.warning("=" * 60)
        logger.warning("CRITICAL: enc_q weights not found in pretrained model!")
        logger.warning("The posterior encoder will be randomly initialized.")
        logger.warning("This is normal for inference-only models (like f0G48k.pth).")
        logger.warning("Training will take longer to converge.")
        logger.warning("=" * 60)

        # Initialize enc_q with sensible defaults for faster convergence
        # Use Xavier initialization for better gradient flow
        for name, param in model.enc_q.named_parameters():
            if 'weight' in name:
                if param.dim() >= 2:
                    torch.nn.init.xavier_uniform_(param)
                else:
                    torch.nn.init.zeros_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
        logger.info("Initialized enc_q with Xavier initialization")

    model.to(device)
    return model


def finetune(
    data_dir: Optional[str] = None,
    preprocessed_dir: Optional[str] = None,
    pretrained_path: str = "",
    output_path: str = "",
    epochs: int = 100,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    device: str = None,
    save_every: int = 50,
    use_gan: bool = True,
    features_dir: Optional[str] = None,
    use_fp16: bool = False,
    disc_warmup_epochs: int = 5,
    stats_csv: Optional[str] = None,
    compute_metrics: bool = False,
):
    """Fine-tune RVC model on voice data.

    Args:
        data_dir: Directory with voice recordings (use preprocess.py output with --preprocessed_dir instead)
        preprocessed_dir: Directory with preprocessed features from preprocess.py (recommended)
        pretrained_path: Path to pretrained model
        output_path: Output path for fine-tuned model
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device (cuda/cpu)
        save_every: Save checkpoint every N epochs
        use_gan: Use GAN training with discriminators (better quality but slower)
        features_dir: Directory to cache preprocessed features (only for data_dir mode)
        use_fp16: Use mixed precision (FP16) - faster but can be unstable
        disc_warmup_epochs: Number of epochs to train generator only before adding discriminator
        stats_csv: Path to CSV file for logging training statistics
        compute_metrics: Compute quality metrics (MCD, F0 RMSE, D accuracy) during training
    """
    from src.utils.stats_logger import StatsLogger
    from src.utils.metrics import compute_mcd, compute_discriminator_accuracy, MetricsTracker
    # Validate inputs
    if not preprocessed_dir and not data_dir:
        raise ValueError("Either --preprocessed_dir or --data_dir must be provided")

    # Setup device
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    logger.info(f"Using device: {dev}")

    # Create dataset based on input type
    index_path = None

    if preprocessed_dir:
        # Use preprocessed data (recommended)
        logger.info(f"Loading preprocessed data from {preprocessed_dir}")
        dataset = PreprocessedDataset(preprocessed_dir)
        index_path = dataset.get_index_path()
        if index_path:
            logger.info(f"Found FAISS index: {index_path}")
    else:
        # Legacy: process raw audio on the fly
        from src.modules.inference_rvc import load_hubert_model
        from src.models.rmvpe import RMVPE
        from src.constants import DEFAULT_RMVPE_PATH

        # Default features_dir to data_dir/.features if not specified
        if features_dir is None:
            features_dir = str(Path(data_dir) / ".features")

        # Load feature extraction models (only if not using cache)
        hubert = None
        rmvpe = None
        features_cache = Path(features_dir) / "features_cache.pt"
        if not features_cache.exists():
            logger.info("Loading HuBERT...")
            hubert = load_hubert_model(dev, is_half=False)
            logger.info("Loading RMVPE...")
            rmvpe = RMVPE(DEFAULT_RMVPE_PATH, is_half=False, device=dev)
        else:
            logger.info("Found cached features, skipping model loading")

        # Create dataset (will use cache if available)
        logger.info("Creating dataset...")
        dataset = VoiceDataset(
            data_dir=data_dir,
            hubert_model=hubert,
            rmvpe_model=rmvpe,
            device=dev,
            features_dir=features_dir,
        )

        # Create FAISS index from collected features
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        index_path = str(output_dir / Path(output_path).stem) + ".index"
        dataset.create_faiss_index(index_path)

    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy index to output dir if using preprocessed data
    if preprocessed_dir and index_path:
        import shutil
        output_index = str(output_dir / Path(output_path).stem) + ".index"
        if index_path != output_index:
            shutil.copy(index_path, output_index)
            index_path = output_index
            logger.info(f"Copied FAISS index to {output_index}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Load pretrained model (with enc_q for training)
    logger.info(f"Loading pretrained model from {pretrained_path}...")
    model = load_synthesizer_for_training(pretrained_path, dev)
    model.train()

    # Create discriminators for GAN training
    discriminator = None
    optim_d = None
    if use_gan:
        from src.models.discriminator import MultiPeriodDiscriminatorV2
        logger.info("Creating discriminator for GAN training...")
        discriminator = MultiPeriodDiscriminatorV2().to(dev)

        # Try to load pretrained discriminator weights if available
        disc_path = pretrained_path.replace("G", "D")  # f0G48k.pth -> f0D48k.pth
        if Path(disc_path).exists():
            logger.info(f"Loading pretrained discriminator from {disc_path}")
            disc_ckpt = torch.load(disc_path, map_location="cpu", weights_only=False)
            if "model" in disc_ckpt:
                disc_state = disc_ckpt["model"]
                missing, unexpected = discriminator.load_state_dict(disc_state, strict=False)
                if missing:
                    logger.warning(f"Discriminator missing {len(missing)} keys")
                if unexpected:
                    logger.warning(f"Discriminator unexpected {len(unexpected)} keys")
                logger.info("Loaded pretrained discriminator weights successfully")
        else:
            logger.info("No pretrained discriminator found, training from scratch")

        optim_d = torch.optim.AdamW(
            discriminator.parameters(),
            lr=learning_rate, betas=(0.8, 0.99), weight_decay=0.01
        )

    # Setup generator optimizer
    optim_g = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.8, 0.99), weight_decay=0.01)
    scaler = GradScaler("cuda", enabled=use_fp16)

    # Use ReduceLROnPlateau for adaptive LR (reduces LR when loss plateaus)
    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim_g, mode='min', factor=0.5, patience=10, min_lr=learning_rate * 0.01
    )
    scheduler_d = None
    if optim_d:
        scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim_d, mode='min', factor=0.5, patience=10, min_lr=learning_rate * 0.01
        )

    # Suppress harmless scheduler warning
    import warnings
    warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*optimizer.step.*")

    # Setup stats logger
    stats_logger = StatsLogger(stats_csv) if stats_csv else None
    if stats_csv:
        logger.info(f"Logging training stats to: {stats_csv}")

    # Training loop
    logger.info("Starting training...")
    logger.info(f"FAISS index saved to: {index_path}")
    logger.info(f"GAN training: {'enabled' if use_gan else 'disabled'}")
    logger.info(f"Mixed precision (FP16): {'enabled' if use_fp16 else 'disabled (recommended)'}")
    logger.info(f"Discriminator warmup: {disc_warmup_epochs} epochs")
    if compute_metrics:
        logger.info("Quality metrics computation: enabled (MCD, D accuracy)")

    # KL warmup: start with low weight, increase to full weight (1.0) over first 10 epochs
    # Reference RVC uses c_kl=1.0 - this is CRITICAL for proper VAE training
    kl_warmup_epochs = 10
    kl_target = 1.0  # Must match reference! Was incorrectly 0.1

    # Metrics tracker for accumulating batch metrics
    metrics_tracker = MetricsTracker() if compute_metrics else None

    for epoch in range(1, epochs + 1):
        epoch_losses = {}
        num_batches = 0
        if metrics_tracker:
            metrics_tracker.reset()

        # KL coefficient warmup - ramps from 0 to kl_target over warmup period
        kl_coef = min(1.0, epoch / kl_warmup_epochs) * kl_target

        # Discriminator warmup: only train generator for first N epochs
        use_disc_this_epoch = use_gan and (epoch > disc_warmup_epochs)

        if use_gan and epoch == disc_warmup_epochs + 1:
            logger.info("Discriminator warmup complete - now training with GAN losses")

        desc = f"Epoch {epoch}/{epochs}"
        if use_gan and not use_disc_this_epoch:
            desc += " (warmup)"
        pbar = tqdm(dataloader, desc=desc)

        # Debug: Log shapes on first batch of first epoch
        first_batch_logged = False

        for batch in pbar:
            if not first_batch_logged and epoch == 1:
                logger.info("=" * 50)
                logger.info("First batch shapes (for debugging):")
                logger.info(f"  phone:        {batch['phone'].shape}")
                logger.info(f"  phone_lengths:{batch['phone_lengths']}")
                logger.info(f"  pitch:        {batch['pitch'].shape}")
                logger.info(f"  pitchf:       {batch['pitchf'].shape}")
                logger.info(f"  spec:         {batch['spec'].shape}  (should be [B, 1025, T])")
                logger.info(f"  spec_lengths: {batch['spec_lengths']}")
                logger.info(f"  wave:         {batch['wave'].shape}")
                logger.info(f"  Model segment_size: {model.segment_size}")
                logger.info("=" * 50)
                first_batch_logged = True
            losses = train_step_gan(
                model, discriminator if use_disc_this_epoch else None,
                batch, optim_g, optim_d if use_disc_this_epoch else None,
                scaler, dev, c_kl=kl_coef, use_gan=use_disc_this_epoch, use_fp16=use_fp16,
                compute_metrics=compute_metrics,
            )

            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v
            num_batches += 1

            # Update progress bar
            postfix = {
                'loss': f"{losses['loss']:.3f}",
                'mel': f"{losses['loss_mel']:.3f}",
            }
            if use_gan and 'loss_adv' in losses:
                postfix['adv'] = f"{losses['loss_adv']:.3f}"
            pbar.set_postfix(postfix)

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= num_batches

        # Log epoch summary
        log_msg = f"Epoch {epoch}: loss={epoch_losses['loss']:.4f}, mel={epoch_losses['loss_mel']:.4f}"
        if 'loss_kl' in epoch_losses:
            log_msg += f", kl={epoch_losses['loss_kl']:.4f}"
        if use_gan and 'loss_adv' in epoch_losses:
            log_msg += f", adv={epoch_losses['loss_adv']:.4f}, fm={epoch_losses.get('loss_fm', 0):.4f}"
        # Add quality metrics to log
        if compute_metrics:
            if 'mcd' in epoch_losses:
                log_msg += f", MCD={epoch_losses['mcd']:.2f}dB"
            if 'd_acc_real' in epoch_losses:
                log_msg += f", D_acc(R/F)={epoch_losses['d_acc_real']:.1%}/{epoch_losses['d_acc_fake']:.1%}"
        logger.info(log_msg)

        # Log to CSV if stats_logger is provided
        if stats_logger is not None:
            current_lr = optim_g.param_groups[0]['lr']
            # Separate metrics from losses for proper logging
            metrics_dict = {}
            if compute_metrics:
                for key in ['mcd', 'd_acc_real', 'd_acc_fake']:
                    if key in epoch_losses:
                        metrics_dict[key] = epoch_losses[key]
            stats_logger.log_dict(
                epoch=epoch, losses=epoch_losses, learning_rate=current_lr, metrics=metrics_dict
            )

        # Step scheduler with the loss (ReduceLROnPlateau needs the metric)
        scheduler_g.step(epoch_losses['loss'])
        if scheduler_d and use_disc_this_epoch:
            scheduler_d.step(epoch_losses['loss'])

        # Save checkpoint
        if epoch % save_every == 0 or epoch == epochs:
            # Save enc_q state before removing (to restore for continued training)
            enc_q_state = {k: v.clone() for k, v in model.enc_q.state_dict().items()}

            # Delete enc_q before saving (not needed for inference)
            del model.enc_q

            # Prepare checkpoint in reference format
            config = [
                model.spec_channels,
                model.segment_size,
                model.inter_channels,
                model.hidden_channels,
                model.filter_channels,
                model.n_heads,
                model.n_layers,
                model.kernel_size,
                model.p_dropout,
                model.resblock,
                model.resblock_kernel_sizes,
                model.resblock_dilation_sizes,
                model.upsample_rates,
                model.upsample_initial_channel,
                model.upsample_kernel_sizes,
                model.spk_embed_dim,
                model.gin_channels,
                48000,  # sr
            ]

            checkpoint = {
                'weight': model.state_dict(),
                'config': config,
                'version': 'v2',
                'f0': 1,
                'info': f'Fine-tuned model, epoch {epoch}',
            }

            save_path = output_path if epoch == epochs else str(output_dir / f"checkpoint_epoch{epoch}.pth")
            torch.save(checkpoint, save_path)
            logger.info(f"Saved checkpoint to {save_path}")

            # Re-add enc_q with trained weights for continued training
            from src.models.synthesizer_rvc import PosteriorEncoder
            model.enc_q = PosteriorEncoder(
                1025, model.inter_channels, model.hidden_channels,
                5, 1, 16, gin_channels=model.gin_channels,
            ).to(dev)
            model.enc_q.load_state_dict(enc_q_state)

    # Close stats logger
    if stats_logger is not None:
        stats_logger.close()
        logger.info(f"Training stats saved to: {stats_csv}")

    logger.info(f"Training complete!")
    logger.info(f"  Model: {output_path}")
    logger.info(f"  Index: {index_path}")
    logger.info(f"Run inference with: python -m src.scripts.infer_rvc -i input.wav -m {output_path} -o output.wav --index {index_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune RVC model on your voice",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using preprocessed data (recommended):
  python -m src.scripts.preprocess --data_dir ./my_voice --out_dir ./features
  python -m src.scripts.finetune --preprocessed_dir ./features --pretrained f0G48k.pth --output my_model.pth

  # Using raw audio directly (slower):
  python -m src.scripts.finetune --data_dir ./my_voice --pretrained f0G48k.pth --output my_model.pth
        """
    )

    # Input data (one of these is required)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--preprocessed_dir",
                             help="Directory with preprocessed features from preprocess.py (recommended)")
    input_group.add_argument("--data_dir",
                             help="Directory containing raw voice recordings (slower, preprocesses on the fly)")

    # Required arguments
    parser.add_argument("--pretrained", required=True, help="Path to pretrained model (.pth)")
    parser.add_argument("--output", required=True, help="Output path for fine-tuned model")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    parser.add_argument("--save_every", type=int, default=100, help="Save checkpoint every N epochs")

    # GAN training options
    parser.add_argument("--gan", action="store_true", default=True,
                        help="Use GAN training (better quality, slower)")
    parser.add_argument("--no-gan", dest="gan", action="store_false",
                        help="Disable GAN training (faster, lower quality)")
    parser.add_argument("--disc-warmup", type=int, default=5,
                        help="Epochs to train generator only before adding discriminator (default: 5)")

    # Legacy options (for --data_dir mode)
    parser.add_argument("--features_dir", default=None,
                        help="Directory to cache preprocessed features (only for --data_dir mode)")

    # Advanced options
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Use mixed precision FP16 (faster but can be unstable)")
    parser.add_argument("--stats_csv", default=None,
                        help="Path to CSV file for logging training statistics")
    parser.add_argument("--compute_metrics", action="store_true", default=False,
                        help="Compute quality metrics (MCD, F0 RMSE, D accuracy) during training")

    args = parser.parse_args()

    finetune(
        data_dir=args.data_dir,
        preprocessed_dir=args.preprocessed_dir,
        pretrained_path=args.pretrained,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        save_every=args.save_every,
        use_gan=args.gan,
        features_dir=args.features_dir,
        use_fp16=args.fp16,
        disc_warmup_epochs=args.disc_warmup,
        stats_csv=args.stats_csv,
        compute_metrics=args.compute_metrics,
    )


if __name__ == "__main__":
    main()