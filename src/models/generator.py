from typing import Any
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.model_base import ModelBase
from src.models.generator_module import GeneratorModule
from src.data_models.data_models import InputData, OutputData, PreprocessedData, PreprocessedSample
from src.utils.training_history import TrainingHistory, TrainingHistoryEntry
from src.utils.wandb_details import WandbDetails


class Generator(ModelBase):
    def __init__(
        self,
        content_dim: int = 256,
        hidden_dim: int = 128,
        use_pitch: bool = True,
        content_sr: int = 16000,
        hop_length: int = 320,
        target_sr: int = 48000,
        learning_rate: float = 1e-4,
        wandb_details: WandbDetails | None = None,
    ):
        super().__init__(wandb_details)
        self.content_dim = content_dim
        self.hidden_dim = hidden_dim
        self.use_pitch = use_pitch
        self.content_sr = content_sr
        self.hop_length = hop_length
        self.target_sr = target_sr
        self.learning_rate = learning_rate
        
        self.generator_module = GeneratorModule(
            content_dim=content_dim,
            hidden_dim=hidden_dim,
            use_pitch=use_pitch,
            content_sr=content_sr,
            hop_length=hop_length,
            target_sr=target_sr,
        )
        self.optimizer = torch.optim.Adam(self.generator_module.parameters(), lr=learning_rate)
        self.history_entries: list[TrainingHistoryEntry] = []

    def get_config_for_wandb(self) -> dict[str, Any]:
        return {
            "content_dim": self.content_dim,
            "hidden_dim": self.hidden_dim,
            "use_pitch": self.use_pitch,
            "content_sr": self.content_sr,
            "hop_length": self.hop_length,
            "target_sr": self.target_sr,
            "learning_rate": self.learning_rate,
        }

    def train(
        self,
        data: PreprocessedData,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> None:
        self.init_wandb_if_needed()

        samples = [
            PreprocessedSample(
                content_vector=data.content_vectors[i],  # [n_frames, content_dim]
                pitch_feature=data.pitch_features[i],  # [n_frames]
                audio=data.audios[i],  # [n_samples]
            )
            for i in range(len(data.content_vectors))
        ]

        dataloader = DataLoader(
            samples,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

        self.generator_module.train()
        criterion = torch.nn.L1Loss()

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for contents, pitches, audios in dataloader:
                # contents: [batch_size, n_frames, content_dim]
                # pitches: [batch_size, n_frames]
                # audios: [batch_size, n_samples]
                
                if self.use_pitch:
                    pitches = pitches.unsqueeze(-1) if pitches.ndim == 2 else pitches  # [batch_size, n_frames, 1]
                    inputs = torch.cat([contents, pitches], dim=-1)  # [batch_size, n_frames, content_dim+1]
                else:
                    inputs = contents

                outputs = self.generator_module(inputs)  # [batch_size, n_samples]

                if outputs.ndim == 3:
                    outputs = outputs.squeeze(1)

                min_len = min(outputs.shape[1], audios.shape[1])
                outputs = outputs[:, :min_len]
                audios = audios[:, :min_len]

                loss = criterion(outputs, audios)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            entry = TrainingHistoryEntry(epoch=epoch, total_loss=avg_loss)
            self.history_entries.append(entry)

            if self.wandb_details is not None:
                self.log_to_wandb(entry)

            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

        self.finish_wandb_if_needed()

    def predict(
        self,
        X: InputData,
        batch_size: int = 32,
    ) -> OutputData:
        self.generator_module.eval()
        wav_data = []

        with torch.inference_mode():
            for i in range(0, len(X.content_vectors), batch_size):
                batch_content = torch.from_numpy(X.content_vectors[i:i+batch_size]).float()  # [batch, n_frames, content_dim]
                batch_pitch = torch.from_numpy(X.pitch_features[i:i+batch_size]).float()  # [batch, n_frames]

                if self.use_pitch:
                    batch_pitch = batch_pitch.unsqueeze(-1) if batch_pitch.ndim == 2 else batch_pitch
                    inputs = torch.cat([batch_content, batch_pitch], dim=-1)
                else:
                    inputs = batch_content

                outputs = self.generator_module(inputs)  # [batch, n_samples]

                if outputs.ndim == 3:
                    outputs = outputs.squeeze(1)

                for j in range(outputs.shape[0]):
                    wav = outputs[j].cpu().numpy()
                    wav_data.append(wav)

        return OutputData(wav_data=wav_data)

    def get_history(self) -> TrainingHistory:
        return TrainingHistory.from_entries(self.history_entries)

    def get_state_dict(self) -> dict[str, Any]:
        return {
            "generator_state": self.generator_module.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
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
    def load_state_dict(
        cls,
        state_dict: dict[str, Any],
    ) -> "Generator":
        config = state_dict["config"]
        generator = cls(
            content_dim=config["content_dim"],
            hidden_dim=config["hidden_dim"],
            use_pitch=config["use_pitch"],
            content_sr=config["content_sr"],
            hop_length=config["hop_length"],
            target_sr=config["target_sr"],
            learning_rate=config["learning_rate"],
        )

        generator.generator_module.load_state_dict(state_dict["generator_state"])

        if state_dict["optimizer_state"] is not None:
            generator.optimizer.load_state_dict(state_dict["optimizer_state"])

        generator.history_entries = state_dict.get("history_entries", [])

        return generator

    def _collate_fn(
        self,
        batch: list[PreprocessedSample],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_len = batch[0].content_vector.shape[0]

        contents = self._normalize_length(
            [torch.tensor(s.content_vector).float() for s in batch], target_len
        )
        pitches = self._normalize_length(
            [torch.tensor(s.pitch_feature).float() for s in batch], target_len
        )
        audios = self._normalize_length(
            [torch.tensor(s.audio).float() for s in batch], target_len
        )

        # [batch_size, target_len, content_dim], [batch_size, target_len], [batch_size, target_len]
        return torch.stack(contents), torch.stack(pitches), torch.stack(audios)

    def _normalize_length(
        self,
        tensors: list[torch.Tensor],
        target_len: int,
    ) -> list[torch.Tensor]:
        return [F.pad(t, (0, max(0, target_len - t.shape[0])))[:target_len] for t in tensors]
