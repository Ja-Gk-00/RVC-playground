import numpy as np
from pathlib import Path

from src.models.generator import Generator
from src.data_models.data_models import PreprocessedData


def train_generator_from_features(
    feature_dir: str,
    epochs: int = 10,
    batch_size: int = 2,
    content_dim: int = 256,
    use_pitch: bool = True,
    target_sr: int = 48000,
    learning_rate: float = 1e-4,
    model_name: str = "trained_generator",
) -> Generator:
    feature_dir = Path(feature_dir)
    
    content_files = sorted(feature_dir.glob("*_units.npy"))
    f0_files = sorted(feature_dir.glob("*_f0.npy"))
    audio_files = sorted(feature_dir.glob("*_audio.npy"))
    
    if not content_files or not f0_files or not audio_files:
        raise ValueError(f"No feature files found in {feature_dir}")
    
    contents = [np.load(f) for f in content_files]  # list of [n_frames_i, content_dim]
    f0s = [np.load(f) for f in f0_files]  # list of [n_frames_i]
    audios = [np.load(f) for f in audio_files]  # list of [n_samples_i]
    
    preprocessed = PreprocessedData(
        content_vectors=np.array(contents, dtype=object),  # [n_samples, n_frames, content_dim]
        pitch_features=np.array(f0s, dtype=object),  # [n_samples, n_frames]
        audios=audios,  # list of [n_samples_i]
    )
    
    generator = Generator(
        content_dim=content_dim,
        use_pitch=use_pitch,
        target_sr=target_sr,
        learning_rate=learning_rate,
    )
    
    generator.train(preprocessed, epochs=epochs, batch_size=batch_size)
    generator.save(model_name)
    
    print(f"Training complete! Model saved as: {model_name}")
    return generator
