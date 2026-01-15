# src/models/source_module.py
"""Neural Source Filter module for harmonic source generation from F0."""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class SourceModuleHnNSF(nn.Module):
    """
    Neural Source Filter - generates harmonic source signals from F0.

    This is the critical component that differentiates RVC from simple vocoders.
    Instead of just using F0 as a conditioning signal, we generate actual
    sine wave harmonics that serve as excitation signals for the neural vocoder.

    Args:
        sampling_rate: Output audio sample rate
        harmonic_num: Number of harmonics (0 = fundamental only)
        sine_amp: Amplitude of sine waves
        noise_std: Standard deviation of noise component
        voiced_threshold: F0 below this is considered unvoiced
    """

    def __init__(
        self,
        sampling_rate: int = 48000,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.sampling_rate = sampling_rate
        self.harmonic_num = harmonic_num
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold

        # Linear layer to merge harmonics
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        # Initialize to pass through fundamental (average for multi-harmonic)
        nn.init.constant_(self.l_linear.weight, 1.0 / (harmonic_num + 1))
        nn.init.zeros_(self.l_linear.bias)
        self.l_tanh = nn.Tanh()

    def _f0_to_rad_per_sample(self, f0: torch.Tensor) -> torch.Tensor:
        """Convert F0 in Hz to radians per sample."""
        return f0 * 2 * math.pi / self.sampling_rate

    def _generate_sine_wave(
        self,
        f0: torch.Tensor,
        upp: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate sine waves from F0.

        Args:
            f0: [B, 1, T] fundamental frequency in Hz
            upp: Upsampling factor (samples per F0 frame)

        Returns:
            sine_waves: [B, T*upp, harmonic_num+1] sine waves for each harmonic
            uv: [B, T*upp, 1] voiced/unvoiced mask
            noise: [B, T*upp, 1] noise component
        """
        batch_size = f0.shape[0]
        f0_len = f0.shape[2]
        device = f0.device

        # Upsample F0 to audio rate using nearest neighbor
        # f0: [B, 1, T] -> [B, 1, T*upp]
        f0_upsampled = torch.nn.functional.interpolate(
            f0,
            size=f0_len * upp,
            mode='nearest'
        )
        f0_upsampled = f0_upsampled.squeeze(1)  # [B, T*upp]

        # Create voiced/unvoiced mask
        uv = (f0_upsampled > self.voiced_threshold).float()  # [B, T*upp]

        # Generate harmonics
        # For each harmonic n, frequency is (n+1) * f0
        rad_per_sample = self._f0_to_rad_per_sample(f0_upsampled)  # [B, T*upp]

        # Random initial phase for each batch
        rand_ini = torch.rand(batch_size, self.harmonic_num + 1, device=device)
        rand_ini[:, 0] = 0  # Fundamental starts at 0 phase

        sine_waves = []
        for i in range(self.harmonic_num + 1):
            harmonic_mult = i + 1
            # Cumulative sum for phase
            phase = torch.cumsum(rad_per_sample * harmonic_mult, dim=1)  # [B, T*upp]
            phase = phase + rand_ini[:, i:i+1] * 2 * math.pi

            # Generate sine wave
            sine = torch.sin(phase)  # [B, T*upp]

            # Zero out unvoiced regions
            sine = sine * uv

            sine_waves.append(sine)

        # Stack harmonics: [B, T*upp, harmonic_num+1]
        sine_waves = torch.stack(sine_waves, dim=2)

        # Generate noise
        noise = torch.randn(batch_size, f0_len * upp, 1, device=device) * self.noise_std

        # Reshape uv for output: [B, T*upp, 1]
        uv = uv.unsqueeze(2)

        return sine_waves, uv, noise

    def forward(
        self,
        f0: torch.Tensor,
        upp: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate harmonic source signal from F0.

        Args:
            f0: [B, 1, T] fundamental frequency in Hz
            upp: Upsampling factor (samples per F0 frame)

        Returns:
            har_source: [B, 1, T*upp] harmonic source signal
            noi_source: [B, 1, T*upp] noise source signal
            uv: [B, 1, T*upp] voiced/unvoiced mask
        """
        # Generate sine waves for all harmonics
        sine_waves, uv, noise = self._generate_sine_wave(f0, upp)

        # Merge harmonics using learned linear combination
        # sine_waves: [B, T*upp, harmonic_num+1] -> [B, T*upp, 1]
        har_source = self.l_linear(sine_waves)
        har_source = self.l_tanh(har_source)

        # Scale by amplitude
        har_source = har_source * self.sine_amp

        # Transpose for output: [B, T*upp, 1] -> [B, 1, T*upp]
        har_source = har_source.transpose(1, 2)
        uv = uv.transpose(1, 2)
        noise = noise.transpose(1, 2)

        return har_source, noise, uv
