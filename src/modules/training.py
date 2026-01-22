# src/modules/training.py
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data_models.data_models import PreprocessedData
from src.models.generator import Generator


def _load_feature_triplets(feature_dir: Path) -> PreprocessedData:
    content_files = sorted(feature_dir.glob("*_units.npy"))
    f0_files = sorted(feature_dir.glob("*_f0.npy"))
    audio_files = sorted(feature_dir.glob("*_audio.npy"))

    if not content_files:
        raise ValueError(f"No *_units.npy found in {feature_dir}")
    if not f0_files:
        raise ValueError(f"No *_f0.npy found in {feature_dir}")
    if not audio_files:
        raise ValueError(
            f"No *_audio.npy found in {feature_dir}. Your preprocessing must save aligned waveform targets per segment."
        )

    print(f"Loading {len(content_files)} samples from {feature_dir}")

    contents = [np.load(f) for f in content_files]
    f0s = [np.load(f) for f in f0_files]
    audios = [np.load(f) for f in audio_files]

    return PreprocessedData(
        content_vectors=np.array(contents, dtype=object),
        pitch_features=np.array(f0s, dtype=object),
        audios=audios,
    )


def train_generator_from_features(
    feature_dir: str,
    epochs: int = 200,
    batch_size: int = 4,
    content_dim: int = 768,
    hidden_dim: int = 192,
    target_sr: int = 48000,
    learning_rate: float = 1e-4,
    model_name: str = "my_voice_rvc",
    pretrained_rvc: bool = False,
    pretrained: str | None = None,
    pretrained_g: str | None = None,
    pretrained_d: str | None = None,
    fp16: bool = True,
    device: str | None = None,
    stats_csv: str | None = None,
) -> Generator:

    from src.utils.stats_logger import StatsLogger
    feature_path = Path(feature_dir)
    data = _load_feature_triplets(feature_path)

    print(f"\n{'='*60}")
    print("RVC Training Configuration")
    print(f"{'='*60}")
    print(f"  Samples: {len(data.content_vectors)}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Target SR: {target_sr}")
    print(f"  FP16: {fp16}")
    print(f"  Pretrained RVC: {pretrained_rvc}")
    print(f"{'='*60}\n")

    g = Generator(
        content_dim=content_dim,
        hidden_dim=hidden_dim,
        use_pitch=True,  # Always True for NSF
        target_sr=target_sr,
        learning_rate=learning_rate,
    )

    # Load pretrained weights
    if pretrained_rvc:
        print("Loading official RVC pretrained weights...")
        g.load_pretrained_rvc(version="v2", load_discriminator=True, verbose=True)
    elif pretrained or pretrained_g or pretrained_d:
        print("Loading custom pretrained weights...")
        g.load_pretrained(
            pretrained=pretrained,
            pretrained_g=pretrained_g,
            pretrained_d=pretrained_d,
        )
    else:
        print("WARNING: Training from scratch. This may take a long time.")
        print("         Consider using --pretrained_rvc for faster training.")

    # Setup stats logger
    stats_logger = StatsLogger(stats_csv) if stats_csv else None
    if stats_csv:
        print(f"Logging training stats to: {stats_csv}")

    try:
        g.train(
            data,
            epochs=epochs,
            batch_size=batch_size,
            fp16=fp16,
            device=device,
            stats_logger=stats_logger,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        g.save(model_name)
        print(f"Model saved as: {model_name}")
        if stats_logger:
            stats_logger.close()
        return g

    if stats_logger:
        stats_logger.close()
        print(f"Training stats saved to: {stats_csv}")

    g.save(model_name)
    print(f"\nTraining complete! Model saved as: {model_name}")
    return g
