import torch
import torchaudio
from pathlib import Path

from src.constants import DEFAULT_HUBERT_PATH, DEFAULT_RMVPE_PATH


class AudioEncoder:
    def __init__(
        self,
        device: torch.device | None = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        hubert_path = Path(DEFAULT_HUBERT_PATH)
        if hubert_path.exists():
            try:
                self.content_model = torch.jit.load(str(hubert_path), map_location=device)
            except Exception:
                self.content_model = torchaudio.pipelines.HUBERT_BASE.get_model().to(device)
        else:
            self.content_model = torchaudio.pipelines.HUBERT_BASE.get_model().to(device)
        
        self.content_model.eval()
        self.pitch_model = None
        
        rmvpe_path = Path(DEFAULT_RMVPE_PATH)
        if rmvpe_path.exists():
            try:
                self.pitch_model = torch.jit.load(str(rmvpe_path), map_location=device)
                self.pitch_model.eval()
            except Exception:
                self.pitch_model = None
        
        self.use_pitch_model = self.pitch_model is not None
        self.content_sr = 16000
        self.hop_length = 320

    def encode_from_file(self, audio_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        waveform, sr = torchaudio.load(audio_path)  # [channels, n_samples]
        waveform = waveform.mean(dim=0, keepdim=True)  # [1, n_samples]
        return self.encode_from_tensor(waveform, sr)

    def encode_from_tensor(
        self,
        audio: torch.Tensor,  # [channels, n_samples]
        sr: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:  # ([n_frames, content_dim], [n_frames])
        waveform = audio.detach().clone()
        if waveform.dim() == 2:
            waveform = waveform[:1]
        else:
            waveform = waveform.unsqueeze(0)

        if sr != self.content_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.content_sr)
        
        waveform = waveform.to(self.device)

        content_units = self._extract_content_features(waveform)
        f0 = self._extract_pitch_features(waveform)

        return content_units, f0

    def _extract_content_features(
        self,
        waveform: torch.Tensor,  # [1, n_samples]
    ) -> torch.Tensor:  # [n_frames, content_dim]
        with torch.inference_mode():
            content_feats = self.content_model(waveform)
            if isinstance(content_feats, tuple):
                content_feats = content_feats[0]
        content_units = content_feats[0].cpu()
        return content_units

    def _extract_pitch_features(
        self,
        waveform: torch.Tensor,  # [1, n_samples]
    ) -> torch.Tensor:  # [n_frames]
        if self.use_pitch_model:
            with torch.inference_mode():
                wave_cpu = waveform.squeeze(0).to("cpu")
                try:
                    f0 = self.pitch_model(wave_cpu)  # [n_frames]
                except Exception as e:
                    raise RuntimeError("RMVPE pitch model inference failed.") from e
                f0 = f0.to(torch.float32)
        else:
            frame_time = self.hop_length / float(self.content_sr)
            f0_vals = torchaudio.functional.detect_pitch_frequency(
                waveform, self.content_sr, frame_time=frame_time
            )
            f0 = f0_vals[0].cpu()
        
        return f0
