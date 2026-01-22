# src/utils/stats_logger.py
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EpochStats:

    epoch: int
    loss: float
    loss_mel: float = 0.0
    loss_kl: float = 0.0
    loss_adv: float = 0.0
    loss_fm: float = 0.0
    loss_d: float = 0.0
    learning_rate: float = 0.0
    # Quality metrics
    mcd: float = 0.0
    f0_rmse: float = 0.0
    f0_corr: float = 0.0
    d_acc_real: float = 0.0
    d_acc_fake: float = 0.0
    snr: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "epoch": self.epoch,
            "loss": self.loss,
            "loss_mel": self.loss_mel,
            "loss_kl": self.loss_kl,
            "loss_adv": self.loss_adv,
            "loss_fm": self.loss_fm,
            "loss_d": self.loss_d,
            "learning_rate": self.learning_rate,
            "mcd": self.mcd,
            "f0_rmse": self.f0_rmse,
            "f0_corr": self.f0_corr,
            "d_acc_real": self.d_acc_real,
            "d_acc_fake": self.d_acc_fake,
            "snr": self.snr,
        }
        result.update(self.extra)
        return result


class StatsLogger:

    def __init__(self, csv_path: str | Path | None) -> None:
        self.csv_path = Path(csv_path) if csv_path else None
        self._file = None
        self._writer = None
        self._header_written = False
        self._stats: list[EpochStats] = []

        if self.csv_path:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.csv_path, "w", newline="", encoding="utf-8")

    def log(self, stats: EpochStats) -> None:
        self._stats.append(stats)

        if self._file is None:
            return

        row = stats.to_dict()

        if not self._header_written:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
            self._header_written = True

        self._writer.writerow(row)
        self._file.flush() 

    def log_dict(
        self,
        epoch: int,
        losses: dict[str, float],
        learning_rate: float = 0.0,
        metrics: dict[str, float] | None = None,
    ) -> None:
        metrics = metrics or {}
        stats = EpochStats(
            epoch=epoch,
            loss=losses.get("loss", 0.0),
            loss_mel=losses.get("loss_mel", 0.0),
            loss_kl=losses.get("loss_kl", 0.0),
            loss_adv=losses.get("loss_adv", 0.0),
            loss_fm=losses.get("loss_fm", 0.0),
            loss_d=losses.get("loss_d", 0.0),
            learning_rate=learning_rate,
            mcd=metrics.get("mcd", 0.0),
            f0_rmse=metrics.get("f0_rmse", 0.0),
            f0_corr=metrics.get("f0_corr", 0.0),
            d_acc_real=metrics.get("d_acc_real", 0.0),
            d_acc_fake=metrics.get("d_acc_fake", 0.0),
            snr=metrics.get("snr", 0.0),
        )
        standard_keys = {
            "loss", "loss_mel", "loss_kl", "loss_adv", "loss_fm", "loss_d",
            "mcd", "f0_rmse", "f0_corr", "d_acc_real", "d_acc_fake", "snr",
        }
        for key, value in losses.items():
            if key not in standard_keys:
                stats.extra[key] = value
        for key, value in metrics.items():
            if key not in standard_keys:
                stats.extra[key] = value

        self.log(stats)

    def get_stats(self) -> list[EpochStats]:
        return self._stats.copy()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "StatsLogger":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def load_stats_csv(csv_path: str | Path) -> list[dict[str, Any]]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Stats file not found: {csv_path}")

    stats = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted = {}
            for key, value in row.items():
                try:
                    if "." in str(value) or "e" in str(value).lower():
                        converted[key] = float(value)
                    else:
                        converted[key] = int(value)
                except (ValueError, TypeError):
                    converted[key] = value
            stats.append(converted)

    return stats
