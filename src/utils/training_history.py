from dataclasses import dataclass


@dataclass
class TrainingHistoryEntry:
    epoch: int
    total_loss: float
    # TODO: Expand

    def to_wandb_dict(self) -> dict[str, float | None]:
        return {"train_loss": self.total_loss}


@dataclass
class TrainingHistory:
    total_loss: list[float] = None
    
    def __post_init__(self):
        self.train_loss = []

    @staticmethod
    def from_entries(entries: list[TrainingHistoryEntry]) -> "TrainingHistory":
        ordered_entries = sorted(entries, key=lambda x: x.epoch)
        return TrainingHistory(
            total_loss=[entry.total_loss for entry in ordered_entries],
        )
