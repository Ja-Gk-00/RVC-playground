from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data_models.data_models import InputData, OutputData, PreprocessedData, PreprocessedSample
from src.models.generator_module import GeneratorModule
from src.models.model_base import ModelBase
from src.utils.training_history import TrainingHistory, TrainingHistoryEntry
from src.utils.wandb_details import WandbDetails


def _mrstft_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Multi-resolution STFT loss (lightweight).
    x, y: [B, N]
    """
    # Small set of resolutions to keep it fast.
    cfgs = [
        (1024, 256, 1024),
        (2048, 512, 2048),
        (512, 128, 512),
    ]

    loss = x.new_zeros(())
    eps = 1e-7

    for n_fft, hop, win in cfgs:
        w = torch.hann_window(win, device=x.device, dtype=x.dtype)

        X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win, window=w, return_complex=True)
        Y = torch.stft(y, n_fft=n_fft, hop_length=hop, win_length=win, window=w, return_complex=True)

        X_mag = (X.abs() + eps)
        Y_mag = (Y.abs() + eps)

        sc = torch.norm(Y_mag - X_mag, p="fro") / (torch.norm(Y_mag, p="fro") + eps)
        mag = F.l1_loss(torch.log(X_mag), torch.log(Y_mag))
        loss = loss + sc + mag

    return loss / float(len(cfgs))


class Generator(ModelBase):
    """Training wrapper + serialization over GeneratorModule."""

    def __init__(
        self,
        content_dim: int = 768,  # RVC v2: 768 features. :contentReference[oaicite:4]{index=4}
        hidden_dim: int = 192,
        use_pitch: bool = True,
        content_sr: int = 16000,
        hop_length: int = 320,
        target_sr: int = 48000,
        learning_rate: float = 1e-4,
        wandb_details: WandbDetails | None = None,
        device: torch.device | None = None,
        pitch_norm_hz: float = 700.0,
        stft_loss_weight: float = 1.0,
        l1_loss_weight: float = 1.0,
    ) -> None:
        super().__init__(wandb_details)

        self.content_dim = int(content_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_pitch = bool(use_pitch)
        self.content_sr = int(content_sr)
        self.hop_length = int(hop_length)
        self.target_sr = int(target_sr)
        self.learning_rate = float(learning_rate)

        self.pitch_norm_hz = float(pitch_norm_hz)
        self.stft_loss_weight = float(stft_loss_weight)
        self.l1_loss_weight = float(l1_loss_weight)

        self.samples_per_frame = int(round(self.target_sr * self.hop_length / float(self.content_sr)))

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.generator_module = GeneratorModule(
            content_dim=self.content_dim,
            hidden_dim=self.hidden_dim,
            use_pitch=self.use_pitch,
            content_sr=self.content_sr,
            hop_length=self.hop_length,
            target_sr=self.target_sr,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.generator_module.parameters(), lr=self.learning_rate, betas=(0.8, 0.99))
        self.history_entries: list[TrainingHistoryEntry] = []

    def to(self, device: torch.device) -> "Generator":
        self.device = device
        self.generator_module.to(device)
        return self

    def get_config_for_wandb(self) -> dict[str, Any]:
        return {
            "content_dim": self.content_dim,
            "hidden_dim": self.hidden_dim,
            "use_pitch": self.use_pitch,
            "content_sr": self.content_sr,
            "hop_length": self.hop_length,
            "target_sr": self.target_sr,
            "learning_rate": self.learning_rate,
            "pitch_norm_hz": self.pitch_norm_hz,
            "stft_loss_weight": self.stft_loss_weight,
            "l1_loss_weight": self.l1_loss_weight,
        }

    def train(self, data: PreprocessedData, epochs: int = 10, batch_size: int = 16) -> None:
        self.init_wandb_if_needed()

        samples = [
            PreprocessedSample(
                content_vector=data.content_vectors[i],
                pitch_feature=data.pitch_features[i],
                audio=data.audios[i],
            )
            for i in range(len(data.content_vectors))
        ]

        dataloader = DataLoader(
            samples,
            batch_size=int(batch_size),
            shuffle=True,
            collate_fn=self._collate_fn,
            drop_last=False,
        )

        self.generator_module.train()

        for epoch in range(int(epochs)):
            epoch_loss = 0.0
            num_batches = 0

            for contents, pitches, audios in dataloader:
                contents = contents.to(self.device, non_blocking=True)  # [B, T, C]
                pitches = pitches.to(self.device, non_blocking=True)    # [B, T]
                audios = audios.to(self.device, non_blocking=True)      # [B, T*samples_per_frame]

                if contents.shape[-1] != self.content_dim:
                    raise ValueError(
                        f"content_dim mismatch: batch has C={contents.shape[-1]} but model expects {self.content_dim}. "
                        "Fix preprocessing to use the same HuBERT checkpoint and content_dim."
                    )

                # normalize pitch to ~[0,1] range; keep 0 for unvoiced
                if self.use_pitch:
                    pitches = torch.clamp(pitches, min=0.0)
                    pitches = torch.clamp(pitches / self.pitch_norm_hz, max=2.0)
                    outputs = self.generator_module(contents, pitches)
                else:
                    outputs = self.generator_module(contents, None)

                if outputs.ndim == 3:
                    outputs = outputs.squeeze(1)

                # Align lengths (should already match, but keep safe)
                min_len = min(int(outputs.shape[1]), int(audios.shape[1]))
                outputs = outputs[:, :min_len]
                audios = audios[:, :min_len]

                loss_l1 = F.l1_loss(outputs, audios)
                loss_stft = _mrstft_loss(outputs, audios)
                loss = (self.l1_loss_weight * loss_l1) + (self.stft_loss_weight * loss_stft)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator_module.parameters(), max_norm=5.0)
                self.optimizer.step()

                epoch_loss += float(loss.item())
                num_batches += 1

            avg_loss = epoch_loss / max(1, num_batches)
            entry = TrainingHistoryEntry(epoch=epoch, total_loss=avg_loss)
            self.history_entries.append(entry)

            if self.wandb_details is not None:
                self.log_to_wandb(entry)

            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

        self.finish_wandb_if_needed()

    def predict(self, X: InputData, batch_size: int = 16) -> OutputData:
        self.generator_module.eval()
        wav_data: list[Any] = []

        with torch.inference_mode():
            for i in range(0, len(X.content_vectors), int(batch_size)):
                batch_content = torch.from_numpy(X.content_vectors[i : i + batch_size]).float().to(self.device)
                batch_pitch = torch.from_numpy(X.pitch_features[i : i + batch_size]).float().to(self.device)

                if batch_content.shape[-1] != self.content_dim:
                    raise ValueError(
                        f"content_dim mismatch in predict: got {batch_content.shape[-1]} expected {self.content_dim}."
                    )

                if self.use_pitch:
                    batch_pitch = torch.clamp(batch_pitch, min=0.0)
                    batch_pitch = torch.clamp(batch_pitch / self.pitch_norm_hz, max=2.0)
                    outputs = self.generator_module(batch_content, batch_pitch)
                else:
                    outputs = self.generator_module(batch_content, None)

                if outputs.ndim == 3:
                    outputs = outputs.squeeze(1)

                for j in range(int(outputs.shape[0])):
                    wav_data.append(outputs[j].detach().cpu().numpy())

        return OutputData(wav_data=wav_data)

    def get_history(self) -> TrainingHistory:
        return TrainingHistory.from_entries(self.history_entries)

    def get_state_dict(self) -> dict[str, Any]:
        return {
            "generator_state": self.generator_module.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.get_config_for_wandb(),
            "history_entries": self.history_entries,
        }

    @classmethod
    def load_state_dict(cls, state_dict: dict[str, Any]) -> "Generator":
        config = state_dict["config"]
        generator = cls(
            content_dim=config["content_dim"],
            hidden_dim=config["hidden_dim"],
            use_pitch=config["use_pitch"],
            content_sr=config["content_sr"],
            hop_length=config["hop_length"],
            target_sr=config["target_sr"],
            learning_rate=config["learning_rate"],
            pitch_norm_hz=config.get("pitch_norm_hz", 700.0),
            stft_loss_weight=config.get("stft_loss_weight", 1.0),
            l1_loss_weight=config.get("l1_loss_weight", 1.0),
        )
        generator.generator_module.load_state_dict(state_dict["generator_state"])
        if state_dict.get("optimizer_state") is not None:
            generator.optimizer.load_state_dict(state_dict["optimizer_state"])
        generator.history_entries = state_dict.get("history_entries", [])
        return generator

    def _collate_fn(self, batch: list[PreprocessedSample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Use first sample's frame count as the batch contract (stable & fast).
        target_frames = int(batch[0].content_vector.shape[0])
        target_audio_len = int(target_frames * self.samples_per_frame)

        contents = self._pad_trim_2d([torch.tensor(s.content_vector).float() for s in batch], target_frames)
        pitches = self._pad_trim_1d([torch.tensor(s.pitch_feature).float() for s in batch], target_frames)
        audios = self._pad_trim_1d([torch.tensor(s.audio).float() for s in batch], target_audio_len)

        return torch.stack(contents), torch.stack(pitches), torch.stack(audios)

    @staticmethod
    def _pad_trim_1d(tensors: list[torch.Tensor], target_len: int) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        for t in tensors:
            t = t.flatten()
            if t.numel() < target_len:
                t = F.pad(t, (0, target_len - t.numel()))
            else:
                t = t[:target_len]
            out.append(t)
        return out

    @staticmethod
    def _pad_trim_2d(tensors: list[torch.Tensor], target_frames: int) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        for t in tensors:
            if t.dim() != 2:
                raise ValueError("content_vector must be 2D [T, C].")
            if t.shape[0] < target_frames:
                pad = torch.zeros((target_frames - t.shape[0], t.shape[1]), dtype=t.dtype)
                t = torch.cat([t, pad], dim=0)
            else:
                t = t[:target_frames]
            out.append(t)
        return out
