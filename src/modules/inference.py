# src/modules/inference.py
import numpy as np
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
    hubert_path: str | None = None,
    rmvpe_path: str | None = None,
    normalize_output: bool = True,
) -> None:
    waveform, sample_rate = torchaudio.load(input_path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    audio_np = waveform.squeeze(0).detach().cpu().numpy().astype(np.float32)

    # Preprocess do 16k (content_sr generatora), żeby nie było rozjazdu czasowego
    preprocessor = Preprocessor(target_sample_rate=int(generator.content_sr))
    audio_segment = preprocessor.prepare_inference_audio(audio_np, sample_rate)

    device = generator.device
    encoder = AudioEncoder(hubert_path=hubert_path, rmvpe_path=rmvpe_path, device=device)

    content, f0 = encoder.encode_from_tensor(
        torch.from_numpy(audio_segment.audio).unsqueeze(0),  # [1, n_samples]
        audio_segment.sample_rate,
    )
    # content: [T, C], f0: [T]
    content_np = content.detach().cpu().numpy().astype(np.float32)
    f0_np = f0.detach().cpu().numpy().astype(np.float32)

    # Krytyczne: dopasuj content_dim do modelu (pad/trunc), gdy checkpoint HuBERT różni się minimalnie
    if content_np.shape[1] != generator.content_dim:
        if content_np.shape[1] < generator.content_dim:
            pad = np.zeros((content_np.shape[0], generator.content_dim - content_np.shape[1]), dtype=np.float32)
            content_np = np.concatenate([content_np, pad], axis=1)
        else:
            content_np = content_np[:, : generator.content_dim]

    input_data = InputData(
        content_vectors=content_np[None, ...],  # [1, T, C]
        pitch_features=f0_np[None, ...],        # [1, T]
    )

    generator.to(device)
    output = generator.predict(input_data, batch_size=1)
    wav = torch.from_numpy(output.wav_data[0]).to(torch.float32)

    # Ochrona przed NaN/Inf
    wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    wav = wav.clamp(-1.0, 1.0)

    if normalize_output:
        peak = float(wav.abs().max().item())
        if peak > 1e-3:
            wav = wav / peak * 0.95

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    torchaudio.save(output_path, wav.unsqueeze(0).cpu(), int(generator.target_sr))
    print(f"Inference complete. Saved to {output_path}")
