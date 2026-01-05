import os

os.environ["TORCHAUDIO_USE_TORCHCODEC"] = "0"

import torchaudio
import numpy as np
import torch
from pathlib import Path

from src.models.audio_encoder import AudioEncoder
from src.models.index_creator import IndexCreator
from src.models.preprocessor import Preprocessor
from src.data_models.data_models import UnprocessedTrainingData

if hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend("sox_io")


def preprocess_for_training(
    data_dir: str,
    out_dir: str,
    quiet: bool = False,
) -> None:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = Preprocessor()
    encoder = AudioEncoder()

    segmented = preprocessor.preprocess(
        UnprocessedTrainingData(audio_dir_path=str(data_dir)), 
        quiet=quiet
    )

    all_content_vectors = []
    
    for i, segment in enumerate(segmented.segments):
        try:
            waveform = torch.from_numpy(segment.audio).unsqueeze(0)  # [1, n_samples]
            
            content, f0 = encoder.encode_from_tensor(waveform.to(torch.float32), segment.sample_rate)
            content_np = content.numpy().astype(np.float32)  # [n_frames, content_dim]
            f0_np = f0.numpy().astype(np.float32)  # [n_frames]

            segment_id = f"segment_{i:06d}"
            np.save(out_dir / f"{segment_id}_units.npy", content_np)
            np.save(out_dir / f"{segment_id}_f0.npy", f0_np)
            np.save(out_dir / f"{segment_id}_audio.npy", segment.audio)  # Save audio for training

            all_content_vectors.append(content_np)
            
            if not quiet:
                print(f"Processed {segment_id}: {content_np.shape[0]} frames.")
        except Exception as e:
            print(f"Warning: failed to process segment {i}: {e}")

    if not all_content_vectors:
        print("No features extracted, index creation skipped.")
        return

    combined_vectors = np.concatenate(all_content_vectors, axis=0).astype("float32")  # [total_frames, content_dim]
    dim = combined_vectors.shape[1]
    index_creator = IndexCreator(dimension=dim)
    index_creator.add(combined_vectors)
    index_creator.save(str(out_dir / "faiss_index.index"))
    np.save(out_dir / "content_vectors.npy", combined_vectors)
    
    if not quiet:
        print(f"FAISS index saved. Indexed {combined_vectors.shape[0]} vectors.")
