# src/models/rmvpe.py
"""RMVPE (Robust Model for Vocal Pitch Estimation) - matches reference RVC implementation."""
from __future__ import annotations

from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiGRU(nn.Module):
    def __init__(self, input_features: int, hidden_features: int, num_layers: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, momentum: float = 0.01) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "shortcut"):
            return self.conv(x) + x
        else:
            return self.conv(x) + self.shortcut(x)


class ResEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        n_blocks: int = 1,
        momentum: float = 0.01,
    ) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor):
        for conv in self.conv:
            x = conv(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        else:
            return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_size: int,
        n_encoders: int,
        kernel_size,
        n_blocks: int,
        out_channels: int = 16,
        momentum: float = 0.01,
    ) -> None:
        super().__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []
        for i in range(self.n_encoders):
            self.layers.append(
                ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum)
            )
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x: torch.Tensor):
        concat_tensors: List[torch.Tensor] = []
        x = self.bn(x)
        for layer in self.layers:
            t, x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_inters: int,
        n_blocks: int,
        momentum: float = 0.01,
    ) -> None:
        super().__init__()
        self.n_inters = n_inters
        self.layers = nn.ModuleList()
        self.layers.append(ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum))
        for _ in range(self.n_inters - 1):
            self.layers.append(ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride,
        n_blocks: int = 1,
        momentum: float = 0.01,
    ) -> None:
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, (3, 3), stride, (1, 1), out_padding, bias=False
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x: torch.Tensor, concat_tensor: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for conv2 in self.conv2:
            x = conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_decoders: int,
        stride,
        n_blocks: int,
        momentum: float = 0.01,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_decoders = n_decoders
        for _ in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum))
            in_channels = out_channels

    def forward(self, x: torch.Tensor, concat_tensors: List[torch.Tensor]) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(
        self,
        kernel_size,
        n_blocks: int,
        en_de_layers: int = 5,
        inter_layers: int = 4,
        in_channels: int = 1,
        en_out_channels: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2,
            self.encoder.out_channel,
            inter_layers,
            n_blocks,
        )
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class E2E(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        n_gru: int,
        kernel_size,
        en_de_layers: int = 5,
        inter_layers: int = 4,
        in_channels: int = 1,
        en_out_channels: int = 16,
    ) -> None:
        super().__init__()
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * 128, 256, n_gru),
                nn.Linear(512, 360),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * 128, 360),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class MelSpectrogram(nn.Module):
    """Mel spectrogram extractor matching reference RVC implementation."""

    def __init__(
        self,
        is_half: bool,
        n_mel_channels: int,
        sampling_rate: int,
        win_length: int,
        hop_length: int,
        n_fft: int | None = None,
        mel_fmin: int = 0,
        mel_fmax: int | None = None,
        clamp: float = 1e-5,
    ) -> None:
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp
        self.is_half = is_half

        # Build mel filterbank (HTK-style to match librosa with htk=True)
        mel_basis = self._mel_filterbank(sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax or (sampling_rate // 2))
        self.register_buffer("mel_basis", mel_basis)

    def _hz_to_mel(self, f: np.ndarray) -> np.ndarray:
        """HTK-style Hz to mel conversion."""
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def _mel_to_hz(self, m: np.ndarray) -> np.ndarray:
        """HTK-style mel to Hz conversion."""
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    def _mel_filterbank(self, sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> torch.Tensor:
        """Create mel filterbank (HTK-style)."""
        # FFT frequencies
        fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)

        # Mel points
        mel_min = self._hz_to_mel(np.array(fmin))
        mel_max = self._hz_to_mel(np.array(fmax))
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)

        # Create filterbank
        filterbank = np.zeros((n_mels, n_fft // 2 + 1))
        for i in range(n_mels):
            lower = hz_points[i]
            center = hz_points[i + 1]
            upper = hz_points[i + 2]

            # Rising slope
            for j, freq in enumerate(fft_freqs):
                if lower <= freq < center:
                    filterbank[i, j] = (freq - lower) / (center - lower + 1e-10)
                elif center <= freq <= upper:
                    filterbank[i, j] = (upper - freq) / (upper - center + 1e-10)

        return torch.from_numpy(filterbank).float()

    def forward(self, audio: torch.Tensor, center: bool = True) -> torch.Tensor:
        """Extract mel spectrogram from audio."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Get or create hann window
        window = torch.hann_window(self.win_length, device=audio.device)

        # STFT
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=center,
            return_complex=True,
        )
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))

        # Mel transform
        mel_output = torch.matmul(self.mel_basis.to(audio.device), magnitude)

        if self.is_half:
            mel_output = mel_output.half()

        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec


class RMVPE:
    """RMVPE pitch extractor - matches reference RVC implementation exactly."""

    def __init__(
        self,
        model_path: str,
        is_half: bool = False,
        device: str | torch.device | None = None,
    ) -> None:
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device) if isinstance(device, str) else device
        self.is_half = is_half

        # Mel extractor: MelSpectrogram(is_half, 128, 16000, 1024, 160, None, 30, 8000)
        self.mel_extractor = MelSpectrogram(
            is_half=is_half,
            n_mel_channels=128,
            sampling_rate=16000,
            win_length=1024,
            hop_length=160,
            n_fft=None,
            mel_fmin=30,
            mel_fmax=8000,
        ).to(self.device)

        # Model: E2E(4, 1, (2, 2))
        self.model = E2E(4, 1, (2, 2))
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt)
        self.model.eval()

        if is_half:
            self.model = self.model.half()
        else:
            self.model = self.model.float()

        self.model = self.model.to(self.device)

        # Cents mapping: 20 * np.arange(360) + 1997.3794084376191
        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))  # 368 total

    def mel2hidden(self, mel: torch.Tensor) -> torch.Tensor:
        """Process mel spectrogram through model."""
        with torch.no_grad():
            n_frames = mel.shape[-1]
            # Pad to multiple of 32
            n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
            if n_pad > 0:
                mel = F.pad(mel, (0, n_pad), mode="constant")

            mel = mel.half() if self.is_half else mel.float()
            hidden = self.model(mel)
            return hidden[:, :n_frames]

    def decode(self, hidden: np.ndarray, thred: float = 0.03) -> np.ndarray:
        """Decode hidden representation to F0."""
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        return f0

    def to_local_average_cents(self, salience: np.ndarray, thred: float = 0.05) -> np.ndarray:
        """Convert salience to cents using local averaging."""
        center = np.argmax(salience, axis=1)  # Frame indices
        salience = np.pad(salience, ((0, 0), (4, 4)))  # Pad to 368

        center += 4
        todo_salience = []
        todo_cents_mapping = []
        starts = center - 4
        ends = center + 5

        for idx in range(salience.shape[0]):
            todo_salience.append(salience[:, starts[idx]:ends[idx]][idx])
            todo_cents_mapping.append(self.cents_mapping[starts[idx]:ends[idx]])

        todo_salience = np.array(todo_salience)  # [frames, 9]
        todo_cents_mapping = np.array(todo_cents_mapping)  # [frames, 9]

        product_sum = np.sum(todo_salience * todo_cents_mapping, 1)
        weight_sum = np.sum(todo_salience, 1)
        devided = product_sum / weight_sum

        maxx = np.max(salience, axis=1)
        devided[maxx <= thred] = 0

        return devided

    def infer_from_audio(self, audio: np.ndarray | torch.Tensor, thred: float = 0.03) -> np.ndarray:
        """Extract F0 from audio waveform at 16kHz."""
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)

        mel = self.mel_extractor(audio.float().to(self.device).unsqueeze(0), center=True)
        hidden = self.mel2hidden(mel)
        hidden = hidden.squeeze(0).cpu().numpy()

        if self.is_half:
            hidden = hidden.astype("float32")

        f0 = self.decode(hidden, thred=thred)
        return f0

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract F0 from audio tensor at 16kHz. Returns tensor."""
        f0 = self.infer_from_audio(audio.cpu().numpy(), thred=0.03)
        return torch.from_numpy(f0).float()

    def eval(self) -> "RMVPE":
        """For compatibility with nn.Module interface."""
        return self
