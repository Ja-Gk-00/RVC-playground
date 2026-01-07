# src/modules/training.py
import numpy as np
from pathlib import Path

from src.models.generator import Generator
from src.data_models.data_models import PreprocessedData


def train_generator_from_features(
    feature_dir: str,
    epochs: int = 50,
    batch_size: int = 1,
    content_dim: int | None = None,
    use_pitch: bool = True,
    target_sr: int = 48000,
    learning_rate: float = 1e-4,
    model_name: str = "my_voice_rvc",
) -> Generator:
    feature_dir_p = Path(feature_dir)

    content_files = sorted(feature_dir_p.glob("*_units.npy"))
    f0_files = sorted(feature_dir_p.glob("*_f0.npy"))
    audio_files = sorted(feature_dir_p.glob("*_audio.npy"))

    if not content_files or not f0_files:
        raise ValueError(f"No *_units.npy or *_f0.npy files found in {feature_dir_p}")

    if not audio_files:
        raise ValueError(
            f"No *_audio.npy found in {feature_dir_p}. "
            "Preprocessing must save aligned waveform targets per segment (seg_000001_audio.npy)."
        )

    # Auto-detect content_dim from data to avoid silent/broken inference due to mismatch
    first = np.load(content_files[0])
    detected_dim = int(first.shape[1])
    if content_dim is None:
        content_dim = detected_dim
    elif int(content_dim) != detected_dim:
        raise ValueError(f"content_dim={content_dim} but features have dim={detected_dim}. Use the detected value.")

    contents = [np.load(f).astype(np.float32) for f in content_files]
    f0s = [np.load(f).astype(np.float32) for f in f0_files]
    audios = [np.load(f).astype(np.float32) for f in audio_files]

    preprocessed = PreprocessedData(
        content_vectors=np.array(contents, dtype=object),
        pitch_features=np.array(f0s, dtype=object),
        audios=audios,
    )

    generator = Generator(
        content_dim=int(content_dim),
        use_pitch=bool(use_pitch),
        target_sr=int(target_sr),
        learning_rate=float(learning_rate),
    )

    generator.train(preprocessed, epochs=int(epochs), batch_size=int(batch_size))
    generator.save(model_name)
    print(f"Training complete! Model saved as: {model_name}")

    return generator
