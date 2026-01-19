# src/models/generator.py
"""RVC Generator with VAE-style training using SynthesizerTrn."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrizations
from torch.utils.data import DataLoader
import torchaudio.functional as F_audio
from scipy.fftpack import dct
from transformers import AutoFeatureExtractor, WavLMForXVector

from src.constants import DEFAULT_RMVPE_PATH
from src.models.rmvpe import RMVPE
from src.data_models.data_models import InputData, OutputData, PreprocessedData, PreprocessedSample
from src.models.synthesizer import SynthesizerTrn
from src.models.model_base import ModelBase
from src.utils.jar import Jar
from src.utils.training_history import TrainingHistory, TrainingHistoryEntry
from src.utils.wandb_details import WandbDetails


def _weight_norm(module: nn.Module) -> nn.Module:
    """Apply weight normalization using new parametrizations API."""
    return parametrizations.weight_norm(module, name="weight", dim=0)


class DiscriminatorP(nn.Module):
    """Period-based discriminator for multi-period discrimination."""

    def __init__(self, period: int) -> None:
        super().__init__()
        self.period = int(period)
        self.convs = nn.ModuleList(
            [
                _weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
                _weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
                _weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
                _weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
                _weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))),
            ]
        )
        self.post = _weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap: list[torch.Tensor] = []
        b, t = x.shape
        if t % self.period != 0:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad), mode="reflect")
            t = t + pad
        x = x.view(b, 1, t // self.period, self.period)
        for c in self.convs:
            x = c(x)
            x = F.leaky_relu(x, 0.1, inplace=True)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x.flatten(1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    """Multi-period discriminator with extended periods for better quality."""

    def __init__(self, periods: tuple[int, ...] = (2, 3, 5, 7, 11, 17)) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(p) for p in periods])

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        outs: list[torch.Tensor] = []
        fmaps: list[list[torch.Tensor]] = []
        for d in self.discriminators:
            o, f = d(x)
            outs.append(o)
            fmaps.append(f)
        return outs, fmaps


class DiscriminatorS(nn.Module):
    """Scale-based discriminator."""

    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                _weight_norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
                _weight_norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                _weight_norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                _weight_norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                _weight_norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                _weight_norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                _weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.post = _weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap: list[torch.Tensor] = []
        x = x.unsqueeze(1)
        for c in self.convs:
            x = c(x)
            x = F.leaky_relu(x, 0.1, inplace=True)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x.flatten(1, -1), fmap


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator with pooling between scales."""

    def __init__(self) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorS(), DiscriminatorS(), DiscriminatorS()])
        self.pooling = nn.ModuleList(
            [
                nn.AvgPool1d(4, 2, padding=2),
                nn.AvgPool1d(4, 2, padding=2),
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        outs: list[torch.Tensor] = []
        fmaps: list[list[torch.Tensor]] = []
        cur = x
        for i, d in enumerate(self.discriminators):
            o, f = d(cur)
            outs.append(o)
            fmaps.append(f)
            if i < len(self.pooling):
                cur = self.pooling[i](cur.unsqueeze(1)).squeeze(1)
        return outs, fmaps


def _d_loss(real_outs: list[torch.Tensor], fake_outs: list[torch.Tensor]) -> torch.Tensor:
    """Discriminator loss: classify real as 1, fake as 0."""
    loss = torch.zeros((), device=real_outs[0].device)
    for r, f in zip(real_outs, fake_outs, strict=True):
        loss = loss + torch.mean((r - 1.0) ** 2) + torch.mean((f - 0.0) ** 2)
    return loss


def _g_adv_loss(fake_outs: list[torch.Tensor]) -> torch.Tensor:
    """Generator adversarial loss: make fake outputs appear real."""
    loss = torch.zeros((), device=fake_outs[0].device)
    for f in fake_outs:
        loss = loss + torch.mean((f - 1.0) ** 2)
    return loss


def _fm_loss(real_fmaps: list[list[torch.Tensor]], fake_fmaps: list[list[torch.Tensor]]) -> torch.Tensor:
    """Feature matching loss: match intermediate discriminator features."""
    loss = torch.zeros((), device=real_fmaps[0][0].device)
    for dr, df in zip(real_fmaps, fake_fmaps, strict=True):
        for r, f in zip(dr, df, strict=True):
            loss = loss + torch.mean(torch.abs(r - f))
    return loss


def _kl_loss(
    z_p: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
) -> torch.Tensor:
    """
    KL divergence loss for VAE training.

    Computes KL(q || p) where:
    - q is the posterior (from enc_q)
    - p is the prior (from enc_p)
    - z_p is the flow-transformed posterior sample

    This formulation ensures gradients flow through the flow module.
    """
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    return kl.mean()


def _compute_mcd(mel_real: torch.Tensor, mel_fake: torch.Tensor, n_mfcc: int = 13) -> float:
    """
    Compute Mel Cepstral Distortion between real and generated mel spectrograms.

    Args:
        mel_real: [B, n_mels, T] log-mel spectrogram of real audio
        mel_fake: [B, n_mels, T] log-mel spectrogram of generated audio
        n_mfcc: Number of MFCCs to use (excluding c0)

    Returns:
        MCD in dB (lower is better, typically 4-8 dB is good)
    """
    # Convert to numpy [B, T, n_mels]
    mel_real_np = mel_real.detach().cpu().numpy().transpose(0, 2, 1)
    mel_fake_np = mel_fake.detach().cpu().numpy().transpose(0, 2, 1)

    # Apply DCT to get MFCCs
    mfcc_real = dct(mel_real_np, type=2, axis=-1, norm='ortho')[..., :n_mfcc]
    mfcc_fake = dct(mel_fake_np, type=2, axis=-1, norm='ortho')[..., :n_mfcc]

    # MCD formula (excluding c0, hence starting from index 1)
    diff = mfcc_real[..., 1:] - mfcc_fake[..., 1:]
    mcd = (10.0 / np.log(10)) * np.sqrt(2.0) * np.mean(np.sqrt(np.sum(diff ** 2, axis=-1)))
    return float(mcd)


def _compute_f0_correlation(f0_real: torch.Tensor, f0_gen: torch.Tensor) -> float:
    """
    Compute F0 Pearson correlation between real and generated F0.

    Args:
        f0_real: [B, T] ground truth F0 in Hz
        f0_gen: [B, T] generated/extracted F0 in Hz

    Returns:
        Pearson correlation coefficient (higher is better, -1 to 1)
    """
    f0_real_np = f0_real.detach().cpu().numpy().flatten()
    f0_gen_np = f0_gen.detach().cpu().numpy().flatten()

    # Only compare voiced regions (F0 > 0)
    voiced_mask = (f0_real_np > 0) & (f0_gen_np > 0)
    if voiced_mask.sum() < 10:
        return 0.0

    f0_real_v = f0_real_np[voiced_mask]
    f0_gen_v = f0_gen_np[voiced_mask]

    # Pearson correlation
    corr = np.corrcoef(f0_real_v, f0_gen_v)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def _compute_speaker_similarity(
    wav_real: torch.Tensor,
    wav_fake: torch.Tensor,
    encoder: nn.Module,
    feature_extractor: Any,
    sr: int = 48000,
) -> float:
    """
    Compute speaker embedding cosine similarity between real and generated audio.

    Args:
        wav_real: [B, N] real audio waveform
        wav_fake: [B, N] generated audio waveform
        encoder: Pre-loaded SpeakerVerification model from transformers
        feature_extractor: Pre-loaded feature extractor for the speaker model
        sr: Sample rate of the audio

    Returns:
        Cosine similarity (higher is better, 0 to 1)
    """
    # Speaker verification model expects 16kHz audio
    target_sr = 16000

    similarities = []
    for i in range(wav_real.shape[0]):
        # Resample to 16kHz
        # Ensure float32 for high-quality resampling and metrics
        wav_r = F_audio.resample(wav_real[i:i+1].cpu().float(), sr, target_sr)
        wav_f = F_audio.resample(wav_fake[i:i+1].cpu().float(), sr, target_sr)

        # Skip if audio too short (less than 0.5s)
        if wav_r.shape[1] < target_sr * 0.5 or wav_f.shape[1] < target_sr * 0.5:
            continue

        # Get embeddings using Transformers
        # Feature extractor handles normalization and format
        inputs_r = feature_extractor(
            wav_r.squeeze(0).numpy(), sampling_rate=target_sr, return_tensors="pt"
        ).to(device=encoder.device, dtype=torch.float32)
        inputs_f = feature_extractor(
            wav_f.squeeze(0).numpy(), sampling_rate=target_sr, return_tensors="pt"
        ).to(device=encoder.device, dtype=torch.float32)

        with torch.no_grad():
            # WavLMForXVector returns an object with 'embeddings'
            emb_real = encoder(**inputs_r).embeddings.squeeze(0).cpu().numpy()
            emb_fake = encoder(**inputs_f).embeddings.squeeze(0).cpu().numpy()

        # Cosine similarity
        sim = np.dot(emb_real, emb_fake) / (
            np.linalg.norm(emb_real) * np.linalg.norm(emb_fake) + 1e-8
        )
        similarities.append(float(sim))

    return float(np.mean(similarities)) if similarities else 0.0


@dataclass(frozen=True)
class MelCfg:
    """Mel-spectrogram configuration."""
    sample_rate: int
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float | None = None
    log_eps: float = 1e-5


def _mel_filterbank(
    device: torch.device,
    dtype: torch.dtype,
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> torch.Tensor:
    """Create mel filterbank."""
    def hz_to_mel(x: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(torch.clamp(x, min=1e-8) / 700.0 + 1.0)

    def mel_to_hz(x: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10.0 ** (x / 2595.0) - 1.0)

    n_freq = n_fft // 2 + 1
    fmin_t = torch.tensor(fmin, device=device, dtype=dtype)
    fmax_t = torch.tensor(fmax, device=device, dtype=dtype)

    m_min = hz_to_mel(fmin_t)
    m_max = hz_to_mel(fmax_t)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2, device=device, dtype=dtype)
    hz_pts = mel_to_hz(m_pts)

    bins = torch.floor((n_fft + 1) * hz_pts / sr).to(torch.int64)
    bins = torch.clamp(bins, 0, n_freq - 1)

    fb = torch.zeros((n_mels, n_freq), device=device, dtype=dtype)
    for m in range(1, n_mels + 1):
        left = int(bins[m - 1].item())
        center = int(bins[m].item())
        right = int(bins[m + 1].item())
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        left = max(left, 0)
        center = min(center, n_freq - 1)
        right = min(right, n_freq - 1)
        if center > left:
            fb[m - 1, left:center] = torch.linspace(0.0, 1.0, center - left, device=device, dtype=dtype)
        if right > center:
            fb[m - 1, center:right] = torch.linspace(1.0, 0.0, right - center, device=device, dtype=dtype)
    return fb


def _mel_spectrogram(wav: torch.Tensor, cfg: MelCfg) -> torch.Tensor:
    """Compute mel-spectrogram from waveform."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    window = torch.hann_window(cfg.win_length, device=wav.device, dtype=wav.dtype)
    stft = torch.stft(
        wav,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    mag = stft.abs().pow(2.0)
    fmax = float(cfg.fmax) if cfg.fmax is not None else float(cfg.sample_rate / 2)
    fb = _mel_filterbank(wav.device, wav.dtype, cfg.sample_rate, cfg.n_fft, cfg.n_mels, float(cfg.fmin), fmax)
    mel = torch.einsum("mf,bft->bmt", fb, mag)
    return torch.log(torch.clamp(mel, min=cfg.log_eps))


def _strip_module_prefix(sd: dict[str, Any]) -> dict[str, Any]:
    if not sd:
        return sd
    out: dict[str, Any] = {}
    for k, v in sd.items():
        nk = k[7:] if k.startswith("module.") else k
        out[nk] = v
    return out


def _extract_sd(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        for k in ("model", "state_dict"):
            v = obj.get(k)
            if isinstance(v, dict):
                return _strip_module_prefix(v)
        return _strip_module_prefix(obj)
    raise TypeError("Unsupported checkpoint object type.")


def _load_checkpoint_any(name_or_path: str) -> dict[str, Any]:
    p = Path(name_or_path)
    if p.exists():
        return _extract_sd(torch.load(str(p), map_location="cpu", weights_only=False))
    return Jar().get(name_or_path)


def _load_matching(module: nn.Module, sd: dict[str, Any]) -> None:
    target = module.state_dict()
    filtered: dict[str, Any] = {}
    for k, v in sd.items():
        if k in target and hasattr(v, "shape") and v.shape == target[k].shape:
            filtered[k] = v
    module.load_state_dict(filtered, strict=False)


class Generator(ModelBase):
    """
    RVC Generator with VAE-style training.

    Uses SynthesizerTrn which combines:
    - TextEncoder (enc_p): HuBERT content -> latent prior
    - PosteriorEncoder (enc_q): mel-spec -> latent posterior
    - ResidualCouplingBlock (flow): latent transformation
    - GeneratorNSF (dec): latent + F0 -> audio

    Args:
        content_dim: HuBERT feature dimension (768)
        hidden_dim: Latent dimension (192)
        use_pitch: Whether to use F0 conditioning (always True for NSF)
        content_sr: Content feature sample rate
        hop_length: Hop length for content features
        target_sr: Target audio sample rate
        learning_rate: Learning rate for optimizers
        wandb_details: Optional W&B configuration
    """

    def __init__(
        self,
        content_dim: int = 768,
        hidden_dim: int = 192,
        use_pitch: bool = True,
        content_sr: int = 16000,
        hop_length: int = 320,
        target_sr: int = 48000,
        learning_rate: float = 1e-4,
        wandb_details: WandbDetails | None = None,
    ) -> None:
        super().__init__(wandb_details)
        self.content_dim = int(content_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_pitch = True  # Always True for NSF-based generation
        self.content_sr = int(content_sr)
        self.hop_length = int(hop_length)
        self.target_sr = int(target_sr)
        self.learning_rate = float(learning_rate)

        self.samples_per_frame = int(round(self.target_sr * self.hop_length / float(self.content_sr)))

        # Main synthesizer (replaces old generator_module)
        self.synthesizer = SynthesizerTrn(
            spec_channels=80,
            inter_channels=hidden_dim,
            hidden_channels=hidden_dim,
            filter_channels=768,
            n_layers=6,
            upsample_rates=(5, 4, 4, 4, 3),  # product = 960 for 48kHz
            upsample_initial_channel=512,
            upsample_kernel_sizes=(10, 8, 8, 8, 6),
            n_speakers=0,  # Single speaker
            sr=target_sr,
        )

        # Discriminators with extended periods
        self.mpd = MultiPeriodDiscriminator(periods=(2, 3, 5, 7, 11, 17))
        self.msd = MultiScaleDiscriminator()

        # Optimizers
        self.opt_g = torch.optim.AdamW(
            self.synthesizer.parameters(),
            lr=self.learning_rate,
            betas=(0.8, 0.99),
            weight_decay=0.01,
        )
        self.opt_d = torch.optim.AdamW(
            list(self.mpd.parameters()) + list(self.msd.parameters()),
            lr=self.learning_rate,
            betas=(0.8, 0.99),
            weight_decay=0.01,
        )

        # Mel config for posterior encoder input
        self.mel_cfg = MelCfg(
            sample_rate=self.target_sr,
            hop_length=self.samples_per_frame,
            n_mels=80,
        )
        self.history_entries: list[TrainingHistoryEntry] = []

    def get_config_for_wandb(self) -> dict[str, Any]:
        return {
            "content_dim": self.content_dim,
            "hidden_dim": self.hidden_dim,
            "use_pitch": self.use_pitch,
            "target_sr": self.target_sr,
            "learning_rate": self.learning_rate,
            "architecture": "SynthesizerTrn",
        }

    def load_pretrained(
        self,
        pretrained: str | None = None,
        pretrained_g: str | None = None,
        pretrained_d: str | None = None,
    ) -> None:
        """Load pretrained weights."""
        if pretrained:
            state = _load_checkpoint_any(pretrained)
            if isinstance(state, dict) and "synthesizer_state" in state:
                _load_matching(self.synthesizer, state["synthesizer_state"])
                if "mpd_state" in state:
                    _load_matching(self.mpd, state["mpd_state"])
                if "msd_state" in state:
                    _load_matching(self.msd, state["msd_state"])
                return
            _load_matching(self.synthesizer, _extract_sd(state))

        if pretrained_g:
            _load_matching(self.synthesizer, _load_checkpoint_any(pretrained_g))
        if pretrained_d:
            d_sd = _load_checkpoint_any(pretrained_d)
            if isinstance(d_sd, dict) and "mpd_state" in d_sd and "msd_state" in d_sd:
                _load_matching(self.mpd, d_sd["mpd_state"])
                _load_matching(self.msd, d_sd["msd_state"])
            else:
                _load_matching(self.mpd, _extract_sd(d_sd))
                _load_matching(self.msd, _extract_sd(d_sd))

    def load_pretrained_rvc(
        self,
        version: str = "v2",
        load_discriminator: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Load official RVC pretrained weights from HuggingFace.

        This downloads and loads the pretrained base models that were trained
        on large multi-speaker datasets. Fine-tuning from these weights is
        much faster than training from scratch.

        Args:
            version: Model version ("v2")
            load_discriminator: Whether to also load discriminator weights
            verbose: Print loading progress

        Example:
            >>> g = Generator(target_sr=48000)
            >>> g.load_pretrained_rvc()  # Downloads and loads pretrained weights
            >>> g.train(data, epochs=200)  # Fine-tune on your data
        """
        from src.models.pretrained import PretrainedManager

        manager = PretrainedManager(
            sample_rate=self.target_sr,
            version=version,
        )

        # Load generator/synthesizer weights
        manager.load_into_synthesizer(self.synthesizer, verbose=verbose)

        # Optionally load discriminator weights
        if load_discriminator:
            manager.load_into_discriminators(self.mpd, self.msd, verbose=verbose)

        if verbose:
            print("Pretrained weights loaded successfully!")
            print("You can now fine-tune on your target speaker data.")

    def train(
        self,
        data: PreprocessedData,
        epochs: int = 50,
        batch_size: int = 4,
        fp16: bool = True,
        device: str | None = None,
        c_mel: float = 45.0,
        c_fm: float = 2.0,
        c_kl: float = 1.0,
        stats_logger: Any | None = None,
        expensive_metrics_every: int | None = 5,
    ) -> None:
        """
        Train the generator with VAE-style training.

        Args:
            data: Preprocessed training data
            epochs: Number of training epochs
            batch_size: Batch size
            fp16: Whether to use mixed precision training
            device: Device to train on
            c_mel: Mel-spectrogram loss weight
            c_fm: Feature matching loss weight
            c_kl: KL divergence loss weight
            stats_logger: Optional StatsLogger instance for CSV logging
            expensive_metrics_every: Compute F0 corr and speaker sim every N epochs
        """
        self.init_wandb_if_needed()

        samples = [
            PreprocessedSample(
                content_vector=data.content_vectors[i],
                pitch_feature=data.pitch_features[i],
                audio=data.audios[i],
            )
            for i in range(len(data.content_vectors))
        ]
        loader = DataLoader(
            samples,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=0,
            pin_memory=True,
        )

        dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.synthesizer.to(dev)
        self.mpd.to(dev)
        self.msd.to(dev)

        self.synthesizer.train()
        self.mpd.train()
        self.msd.train()

        scaler = torch.amp.GradScaler(enabled=bool(fp16 and dev.type == "cuda"))

        speaker_encoder: nn.Module | None = None
        speaker_feature_extractor: Any = None
        rmvpe_model: Any = None

        for epoch in tqdm(range(epochs), desc="Training"):
            g_sum = 0.0
            d_sum = 0.0
            kl_sum = 0.0
            mcd_sum = 0.0
            nb = 0

            compute_expensive = expensive_metrics_every is not None and ((epoch % expensive_metrics_every == 0) or (epoch == epochs - 1))

            if compute_expensive and speaker_encoder is None:
                # Load WavLM instead of SpeechBrain
                model_id = "microsoft/wavlm-base-plus-sv"
                speaker_encoder = WavLMForXVector.from_pretrained(model_id).to(dev)
                speaker_feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
                speaker_encoder.eval()

            if compute_expensive and rmvpe_model is None:
                rmvpe_model = RMVPE(DEFAULT_RMVPE_PATH, is_half=False, device=dev)

            f0_corr_samples: list[float] = []
            spk_sim_samples: list[float] = []

            for content, f0, mel, wav_real in loader:
                content = content.to(dev, non_blocking=True)
                f0 = f0.to(dev, non_blocking=True)
                mel = mel.to(dev, non_blocking=True)
                wav_real = wav_real.to(dev, non_blocking=True)

                # --- Forward pass through synthesizer
                with torch.amp.autocast(enabled=scaler.is_enabled(), device_type=dev.type):
                    outputs = self.synthesizer(content, f0, mel)
                    wav_fake = outputs["audio"]

                # Align lengths
                min_len = min(wav_fake.shape[1], wav_real.shape[1])
                wav_fake = wav_fake[:, :min_len]
                wav_real = wav_real[:, :min_len]

                # --- Discriminator update
                with torch.amp.autocast(enabled=scaler.is_enabled(), device_type=dev.type):
                    wf_det = wav_fake.detach()
                    real_mpd, _ = self.mpd(wav_real)
                    fake_mpd, _ = self.mpd(wf_det)
                    real_msd, _ = self.msd(wav_real)
                    fake_msd, _ = self.msd(wf_det)
                    loss_d = _d_loss(real_mpd, fake_mpd) + _d_loss(real_msd, fake_msd)

                self.opt_d.zero_grad(set_to_none=True)
                scaler.scale(loss_d).backward()
                scaler.step(self.opt_d)

                # --- Generator update
                with torch.amp.autocast(enabled=scaler.is_enabled(), device_type=dev.type):
                    # Re-forward for generator gradients
                    outputs = self.synthesizer(content, f0, mel)
                    wav_fake = outputs["audio"]

                    # Align again
                    wav_fake = wav_fake[:, :min_len]

                    # Feature maps for matching
                    _, real_mpd_fm = self.mpd(wav_real)
                    _, real_msd_fm = self.msd(wav_real)
                    fake_mpd_g, fake_mpd_fm = self.mpd(wav_fake)
                    fake_msd_g, fake_msd_fm = self.msd(wav_fake)

                    # Losses
                    loss_adv = _g_adv_loss(fake_mpd_g) + _g_adv_loss(fake_msd_g)
                    loss_fm = _fm_loss(real_mpd_fm, fake_mpd_fm) + _fm_loss(real_msd_fm, fake_msd_fm)

                    mel_real = _mel_spectrogram(wav_real, self.mel_cfg)
                    mel_fake = _mel_spectrogram(wav_fake, self.mel_cfg)
                    loss_mel = torch.mean(torch.abs(mel_real - mel_fake))

                    # KL divergence loss (uses z_p for gradients to flow)
                    loss_kl = _kl_loss(
                        outputs["z_p"],
                        outputs["logs_q"],
                        outputs["m_p"],
                        outputs["logs_p"],
                    )

                    loss_g = loss_adv + (c_fm * loss_fm) + (c_mel * loss_mel) + (c_kl * loss_kl)

                self.opt_g.zero_grad(set_to_none=True)
                scaler.scale(loss_g).backward()
                scaler.step(self.opt_g)
                scaler.update()

                mcd = _compute_mcd(mel_real, mel_fake)
                mcd_sum += mcd

                if compute_expensive:
                    with torch.no_grad():
                        # Extract F0 from generated audio (resample to 16kHz first)
                        wav_fake_16k = F_audio.resample(
                            wav_fake.detach().cpu().float(), self.target_sr, 16000
                        )
                        
                        for i in range(wav_fake_16k.shape[0]):
                            f0_gen = rmvpe_model.infer_from_audio(
                                wav_fake_16k[i].numpy(), thred=0.03
                            )
                            # Downsample to match training F0 (hop 160 -> 320)
                            f0_gen = f0_gen[::2]
                            f0_real_i = f0[i].cpu()
                            # Align lengths
                            min_f0_len = min(len(f0_gen), f0_real_i.shape[0])
                            f0_corr = _compute_f0_correlation(
                                f0_real_i[:min_f0_len].unsqueeze(0),
                                torch.from_numpy(f0_gen[:min_f0_len]).unsqueeze(0),
                            )
                            f0_corr_samples.append(f0_corr)

                        # Speaker similarity
                        spk_sim = _compute_speaker_similarity(
                            wav_real.detach(),
                            wav_fake.detach(),
                            speaker_encoder,
                            speaker_feature_extractor,
                            sr=self.target_sr,
                        )
                        spk_sim_samples.append(spk_sim)

                g_sum += float(loss_g.item())
                d_sum += float(loss_d.item())
                kl_sum += float(loss_kl.item())
                nb += 1

            avg_g = g_sum / max(1, nb)
            avg_d = d_sum / max(1, nb)
            avg_kl = kl_sum / max(1, nb)
            avg_mcd = mcd_sum / max(1, nb)

            avg_f0_corr: float | None = None
            avg_spk_sim: float | None = None
            if f0_corr_samples:
                avg_f0_corr = float(np.mean(f0_corr_samples))
            if spk_sim_samples:
                avg_spk_sim = float(np.mean(spk_sim_samples))

            entry = TrainingHistoryEntry(
                epoch=epoch,
                total_loss=avg_g,
                d_loss=avg_d,
                kl_loss=avg_kl,
                mcd=avg_mcd,
                f0_corr=avg_f0_corr,
                speaker_sim=avg_spk_sim,
            )
            self.history_entries.append(entry)

            if self.wandb_details is not None:
                self.log_to_wandb(entry)

            # Log to CSV if stats_logger is provided
            if stats_logger is not None:
                stats_logger.log_dict(
                    epoch=epoch,
                    losses={
                        "loss": avg_g,
                        "loss_d": avg_d,
                        "loss_kl": avg_kl,
                        "mcd": avg_mcd,
                        "f0_corr": avg_f0_corr,
                        "speaker_sim": avg_spk_sim,
                    },
                    learning_rate=self.learning_rate,
                )

            print(f"Epoch {epoch}: G={avg_g:.4f} D={avg_d:.4f} KL={avg_kl:.4f}")

            # Build log message
            log_msg = f"Epoch {epoch}: G={avg_g:.4f} D={avg_d:.4f} KL={avg_kl:.4f} MCD={avg_mcd:.2f}dB"
            if avg_f0_corr is not None:
                log_msg += f" F0corr={avg_f0_corr:.3f}"
            if avg_spk_sim is not None:
                log_msg += f" SpkSim={avg_spk_sim:.3f}"
            print(log_msg)

        self.finish_wandb_if_needed()

    def predict(self, X: InputData, batch_size: int = 8) -> OutputData:
        """Generate audio from content features."""
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.synthesizer.eval().to(dev)

        wavs: list[Any] = []
        with torch.inference_mode():
            for i in range(0, len(X.content_vectors), batch_size):
                content = torch.from_numpy(X.content_vectors[i:i + batch_size]).float()
                f0 = torch.from_numpy(X.pitch_features[i:i + batch_size]).float()

                # Transpose content to [B, 768, T]
                content = content.transpose(1, 2).to(dev)
                f0 = f0.to(dev)

                y = self.synthesizer.infer(content, f0).cpu().numpy()
                for j in range(y.shape[0]):
                    wavs.append(y[j])

        return OutputData(wav_data=wavs)

    def infer_single(
        self,
        content: torch.Tensor,
        f0: torch.Tensor,
        device: str | None = None,
    ) -> torch.Tensor:
        """
        Inference for a single sample.

        Args:
            content: [T, 768] content features
            f0: [T] F0 in Hz

        Returns:
            [T*upp] generated audio
        """
        dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.synthesizer.eval().to(dev)

        with torch.inference_mode():
            # Add batch dimension and transpose
            content = content.unsqueeze(0).transpose(1, 2).to(dev)  # [1, 768, T]
            f0 = f0.unsqueeze(0).to(dev)  # [1, T]

            audio = self.synthesizer.infer(content, f0)
            return audio.squeeze(0).cpu()

    def get_history(self) -> TrainingHistory:
        return TrainingHistory.from_entries(self.history_entries)

    def get_state_dict(self) -> dict[str, Any]:
        return {
            "synthesizer_state": self.synthesizer.state_dict(),
            "mpd_state": self.mpd.state_dict(),
            "msd_state": self.msd.state_dict(),
            "opt_g_state": self.opt_g.state_dict(),
            "opt_d_state": self.opt_d.state_dict(),
            "config": {
                "content_dim": self.content_dim,
                "hidden_dim": self.hidden_dim,
                "use_pitch": self.use_pitch,
                "content_sr": self.content_sr,
                "hop_length": self.hop_length,
                "target_sr": self.target_sr,
                "learning_rate": self.learning_rate,
            },
            "history_entries": self.history_entries,
        }

    @classmethod
    def load_state_dict(cls, state_dict: dict[str, Any]) -> "Generator":
        cfg = state_dict["config"]
        model = cls(
            content_dim=int(cfg["content_dim"]),
            hidden_dim=int(cfg.get("hidden_dim", 192)),
            use_pitch=bool(cfg["use_pitch"]),
            content_sr=int(cfg["content_sr"]),
            hop_length=int(cfg["hop_length"]),
            target_sr=int(cfg["target_sr"]),
            learning_rate=float(cfg["learning_rate"]),
        )
        _load_matching(model.synthesizer, state_dict.get("synthesizer_state", {}))
        if "mpd_state" in state_dict:
            _load_matching(model.mpd, state_dict["mpd_state"])
        if "msd_state" in state_dict:
            _load_matching(model.msd, state_dict["msd_state"])
        model.history_entries = state_dict.get("history_entries", [])
        return model

    def _collate_fn(
        self,
        batch: list[PreprocessedSample],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate function that produces content, f0, mel, and audio.

        Returns:
            content: [B, 768, T] HuBERT content features
            f0: [B, T] F0 in Hz
            mel: [B, 80, T] mel-spectrogram
            wav: [B, N] audio waveform
        """
        t = int(batch[0].content_vector.shape[0])
        n = int(t * self.samples_per_frame)

        contents: list[torch.Tensor] = []
        f0s: list[torch.Tensor] = []
        mels: list[torch.Tensor] = []
        wavs: list[torch.Tensor] = []

        for s in batch:
            c = torch.from_numpy(s.content_vector.astype(np.float32))  # [Ti, 768]
            p = torch.from_numpy(s.pitch_feature.astype(np.float32))   # [Ti]
            w = torch.from_numpy(s.audio.astype(np.float32))           # [Ni]

            c = self._pad_or_trim_2d(c, t)  # [T, 768]
            p = self._pad_or_trim_1d(p, t)  # [T]
            w = self._pad_or_trim_1d(w, n)  # [N]

            # Transpose content to [768, T]
            c = c.transpose(0, 1)

            # Compute mel-spectrogram from audio
            mel = _mel_spectrogram(w.unsqueeze(0), self.mel_cfg).squeeze(0)  # [80, T_mel]

            # Align mel length with content
            mel = self._pad_or_trim_2d(mel.transpose(0, 1), t).transpose(0, 1)  # [80, T]

            contents.append(c)
            f0s.append(p)
            mels.append(mel)
            wavs.append(w)

        return (
            torch.stack(contents, dim=0),  # [B, 768, T]
            torch.stack(f0s, dim=0),        # [B, T]
            torch.stack(mels, dim=0),       # [B, 80, T]
            torch.stack(wavs, dim=0),       # [B, N]
        )

    @staticmethod
    def _pad_or_trim_1d(x: torch.Tensor, target: int) -> torch.Tensor:
        if x.numel() >= target:
            return x[:target]
        return F.pad(x, (0, target - x.numel()))

    @staticmethod
    def _pad_or_trim_2d(x: torch.Tensor, target_t: int) -> torch.Tensor:
        t, c = x.shape
        if t >= target_t:
            return x[:target_t, :]
        pad = torch.zeros((target_t - t, c), dtype=x.dtype)
        return torch.cat([x, pad], dim=0)
