import torch
import torchaudio
from pathlib import Path

from src.models.audio_encoder import AudioEncoder
from src.models.preprocessor import Preprocessor
from src.models.generator import Generator
from src.data_models.data_models import InputData


def run_inference(
    input_path: str,
    generator: Generator,
    output_path: str,
) -> None:
    waveform, sample_rate = torchaudio.load(input_path)  # [channels, n_samples]
    
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    audio_np = waveform.squeeze().numpy()  # [n_samples]
    
    preprocessor = Preprocessor()
    audio_segment = preprocessor.prepare_inference_audio(audio_np, sample_rate)
    
    encoder = AudioEncoder()
    content, f0 = encoder.encode_from_tensor(
        torch.from_numpy(audio_segment.audio).unsqueeze(0),
        audio_segment.sample_rate,
    )

    input_data = InputData(
        content_vectors=content.unsqueeze(0).numpy(),  # [1, n_frames, content_dim]
        pitch_features=f0.unsqueeze(0).numpy(),  # [1, n_frames]
    )

    output = generator.predict(input_data, batch_size=1)
    wav = torch.from_numpy(output.wav_data[0])
    wav = wav.clamp(-1.0, 1.0)

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(output_path, wav.unsqueeze(0), generator.target_sr)
    print(f"Inference complete. Saved to {output_path}")
