import warnings
from pathlib import Path

import torch
import torchaudio

from src.constants import DEFAULT_HUBERT_PATH, DEFAULT_RMVPE_PATH


class AudioEncoder:
    def __init__(
        self,
        hubert_path: str | None = None,
        rmvpe_path: str | None = None,
        device: torch.device | None = None,
        allow_unsafe_rmvpe_load: bool = False,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.allow_unsafe_rmvpe_load = allow_unsafe_rmvpe_load

        hubert_path_p = Path(hubert_path) if hubert_path else Path(DEFAULT_HUBERT_PATH)
        rmvpe_path_p = Path(rmvpe_path) if rmvpe_path else Path(DEFAULT_RMVPE_PATH)

        # HuBERT / ContentVec
        self.content_model = None
        if hubert_path_p.exists():
            try:
                self.content_model = torch.jit.load(str(hubert_path_p), map_location=device)
            except Exception:
                self.content_model = None
        if self.content_model is None:
            self.content_model = torchaudio.pipelines.HUBERT_BASE.get_model().to(device)
        self.content_model.eval()

        # RMVPE (optional)
        self.pitch_model = None
        self.use_pitch_model = False
        if rmvpe_path_p.exists():
            self._try_load_rmvpe(rmvpe_path_p)

        self.content_sr = 16000
        self.hop_length = 320

    def _try_load_rmvpe(self, rmvpe_path: Path) -> None:
        # 1) TorchScript path (preferred)
        try:
            self.pitch_model = torch.jit.load(str(rmvpe_path), map_location=self.device)
            self.pitch_model.eval()
            self.use_pitch_model = True
            return
        except Exception:
            self.pitch_model = None
            self.use_pitch_model = False

        # 2) State_dict checkpoints require RMVPE architecture code (not in repo)
        # Additionally, torch.load defaults can error on some PyTorch versions.
        if not self.allow_unsafe_rmvpe_load:
            warnings.warn(
                f"RMVPE is not TorchScript ({rmvpe_path}). "
                "Falling back to torchaudio pitch. "
                "If you trust this checkpoint and have RMVPE architecture code integrated, "
                "you may enable allow_unsafe_rmvpe_load=True."
            )
            return

        try:
            _ = torch.load(str(rmvpe_path), map_location="cpu", weights_only=False)
        except Exception as e:
            warnings.warn(
                f"RMVPE unsafe load failed ({rmvpe_path}): {e}. Falling back to torchaudio pitch."
            )
            return

        warnings.warn(
            "RMVPE loaded via torch.load, but RMVPE architecture is not present in this repo, "
            "so it cannot be executed. Falling back to torchaudio pitch."
        )

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

        content_units = self._extract_content_features(waveform)
        f0 = self._extract_pitch_features(waveform)

        t = content_units.shape[0]
        if f0.numel() < t:
            f0 = torch.nn.functional.pad(f0, (0, t - f0.numel()))
        elif f0.numel() > t:
            f0 = f0[:t]

        return content_units, f0

    def _extract_content_features(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            out = self.content_model(waveform)
            if isinstance(out, tuple):
                out = out[0]
        return out[0].to("cpu")

    def _extract_pitch_features(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.use_pitch_model and self.pitch_model is not None:
            with torch.inference_mode():
                wave_cpu = waveform.squeeze(0).to("cpu")
                try:
                    f0 = self.pitch_model(wave_cpu)
                except Exception as e:
                    raise RuntimeError("RMVPE pitch inference failed.") from e
            return f0.to(torch.float32)

        frame_time = self.hop_length / float(self.content_sr)
        f0_vals = torchaudio.functional.detect_pitch_frequency(
            waveform.to("cpu"), self.content_sr, frame_time=frame_time
        )
        return f0_vals[0].to(torch.float32)
