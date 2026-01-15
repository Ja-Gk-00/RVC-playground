# src/models/audio_encoder.py
from __future__ import annotations

import warnings
from pathlib import Path

import torch
import torchaudio

from src.constants import DEFAULT_HUBERT_PATH, DEFAULT_RMVPE_PATH


class AudioEncoder:
    """
    Content: HuBERT_BASE (torchaudio pipeline) or a TorchScript model if provided.
    Pitch:   RMVPE model (TorchScript or weights with architecture), else torchaudio pitch fallback.
    """

    def __init__(
        self,
        hubert_path: str | None = None,
        rmvpe_path: str | None = None,
        device: torch.device | None = None,
        allow_unsafe_rmvpe_load: bool = False,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.allow_unsafe_rmvpe_load = allow_unsafe_rmvpe_load

        hubert_path_p = Path(hubert_path) if hubert_path else Path(DEFAULT_HUBERT_PATH)
        rmvpe_path_p = Path(rmvpe_path) if rmvpe_path else Path(DEFAULT_RMVPE_PATH)

        self.content_model = None
        if hubert_path_p.exists():
            try:
                self.content_model = torch.jit.load(str(hubert_path_p), map_location=self.device)
            except Exception:
                self.content_model = None

        if self.content_model is None:
            self.content_model = torchaudio.pipelines.HUBERT_BASE.get_model().to(self.device)

        self.content_model.eval()

        self.pitch_model = None
        self.use_pitch_model = False
        if rmvpe_path_p.exists():
            self._try_load_rmvpe(rmvpe_path_p)

        self.content_sr = 16000
        self.hop_length = 320

    # ---- public API (keep compatibility with older calls) ----
    def encode(self, wav: torch.Tensor, sr: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_from_tensor(wav, sr)

    def encode_from_file(self, audio_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        waveform, sr = torchaudio.load(audio_path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        else:
            waveform = waveform[:1]
        return self.encode_from_tensor(waveform, sr)

    def encode_from_tensor(self, audio: torch.Tensor, sr: int) -> tuple[torch.Tensor, torch.Tensor]:
        waveform = audio.detach()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform[:1]
        else:
            raise ValueError("Audio tensor must have shape [n_samples] or [channels, n_samples].")

        if sr != self.content_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.content_sr)

        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

        waveform = waveform.to(self.device)

        content = self._extract_content_features(waveform).to("cpu")
        f0 = self._extract_pitch_features(waveform).to("cpu")

        # force time alignment
        t = int(content.shape[0])
        if f0.numel() < t:
            f0 = torch.nn.functional.pad(f0, (0, t - f0.numel()))
        elif f0.numel() > t:
            f0 = f0[:t]

        return content, f0

    # ---- internals ----
    def _try_load_rmvpe(self, rmvpe_path: Path) -> None:
        # 1) Try TorchScript first (fastest)
        try:
            self.pitch_model = torch.jit.load(str(rmvpe_path), map_location=self.device)
            self.pitch_model.eval()
            self.use_pitch_model = True
            self._rmvpe_type = "torchscript"
            return
        except Exception as e:
            warnings.warn(f"Failed to load RMVPE through jit with architecture: {e}")

        # 2) Try loading with our RMVPE architecture implementation
        try:
            from src.models.rmvpe import RMVPE
            self.pitch_model = RMVPE(
                model_path=str(rmvpe_path),
                is_half=False,
                device=self.device,
            )
            self.pitch_model.eval()
            self.use_pitch_model = True
            self._rmvpe_type = "pytorch"
            print("Loaded the RMVPE model")
            return
        except Exception as e:
            warnings.warn(f"Failed to load RMVPE with architecture: {e}")
            pass

        # 3) Fallback to torchaudio pitch
        self.pitch_model = None
        self.use_pitch_model = False
        self._rmvpe_type = None
        warnings.warn(
            f"Could not load RMVPE from {rmvpe_path}. Falling back to torchaudio pitch detection. "
            "For better quality, ensure rmvpe.pt contains valid weights."
        )

    def _extract_content_features(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            out = self.content_model(waveform)
            # torchaudio hubert often returns (features, lengths)
            if isinstance(out, tuple):
                out = out[0]
            # expected [B, T, D]
            if out.dim() == 3:
                return out[0]
            # scripted variants sometimes return [T, D] directly
            if out.dim() == 2:
                return out
            raise RuntimeError(f"Unexpected HuBERT output shape: {tuple(out.shape)}")

    def _extract_pitch_features(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.use_pitch_model and self.pitch_model is not None:
            with torch.inference_mode():
                w = waveform.squeeze(0)
                # Handle different RMVPE types
                if hasattr(self, '_rmvpe_type') and self._rmvpe_type == "pytorch":
                    # Our RMVPE implementation - works on device
                    f0 = self.pitch_model(w.to(self.device))
                    # RMVPE uses 160 hop, we need 320 hop - downsample by 2
                    if f0.numel() > 1:
                        f0 = f0[::2]  # Take every other frame
                else:
                    # TorchScript version
                    f0 = self.pitch_model(w.to("cpu"))
                return f0.to("cpu").to(torch.float32)

        # Fallback to torchaudio pitch detection
        frame_time = self.hop_length / float(self.content_sr)
        f0_vals = torchaudio.functional.detect_pitch_frequency(
            waveform.to("cpu"), self.content_sr, frame_time=frame_time
        )
        return f0_vals[0].to(torch.float32)
