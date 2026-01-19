from dataclasses import dataclass, field


@dataclass
class TrainingHistoryEntry:
    epoch: int
    total_loss: float
    d_loss: float = 0.0
    kl_loss: float = 0.0
    mcd: float | None = None  # Mel Cepstral Distortion
    f0_corr: float | None = None  # F0 correlation
    speaker_sim: float | None = None  # Speaker similarity

    def to_wandb_dict(self) -> dict[str, float | None]:
        return {
            "train_loss": self.total_loss,
            "d_loss": self.d_loss,
            "kl_loss": self.kl_loss,
            "mcd": self.mcd,
            "f0_corr": self.f0_corr,
            "speaker_sim": self.speaker_sim,
        }


@dataclass
class TrainingHistory:
    total_loss: list[float] = field(default_factory=list)
    d_loss: list[float] = field(default_factory=list)
    kl_loss: list[float] = field(default_factory=list)
    mcd: list[float | None] = field(default_factory=list)
    f0_corr: list[float | None] = field(default_factory=list)
    speaker_sim: list[float | None] = field(default_factory=list)

    @staticmethod
    def from_entries(entries: list[TrainingHistoryEntry]) -> "TrainingHistory":
        ordered_entries = sorted(entries, key=lambda x: x.epoch)
        return TrainingHistory(
            total_loss=[entry.total_loss for entry in ordered_entries],
            d_loss=[entry.d_loss for entry in ordered_entries],
            kl_loss=[entry.kl_loss for entry in ordered_entries],
            mcd=[entry.mcd for entry in ordered_entries],
            f0_corr=[entry.f0_corr for entry in ordered_entries],
            speaker_sim=[entry.speaker_sim for entry in ordered_entries],
        )
