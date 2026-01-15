# src/models/encoders.py
"""Encoder modules for RVC: TextEncoder and PosteriorEncoder."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    """
    Processes HuBERT content features into latent space.

    Projects 768-dim HuBERT features to hidden_channels, processes through
    convolutional layers, and outputs mean + log-variance for VAE sampling.

    Args:
        in_channels: Input dimension (768 for HuBERT)
        hidden_channels: Latent dimension
        filter_channels: Intermediate filter dimension
        n_heads: Number of attention heads (unused in simplified version)
        n_layers: Number of convolutional layers
        kernel_size: Convolution kernel size
        p_dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int = 768,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers

        # Project HuBERT features to hidden dimension
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)

        # F0 embedding - quantize F0 to 256 bins
        self.emb_pitch = nn.Embedding(256, hidden_channels)

        # Convolutional encoder layers
        self.encoder = nn.ModuleList()
        for _ in range(n_layers):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_channels,
                        filter_channels,
                        kernel_size,
                        padding=kernel_size // 2
                    ),
                    nn.ReLU(),
                    nn.Conv1d(
                        filter_channels,
                        hidden_channels,
                        kernel_size,
                        padding=kernel_size // 2
                    ),
                    nn.Dropout(p_dropout),
                )
            )

        # Layer normalization
        self.norm = nn.LayerNorm(hidden_channels)

        # Output projection to mean and log-variance
        self.proj = nn.Conv1d(hidden_channels, hidden_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        f0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode content features.

        Args:
            x: [B, 768, T] HuBERT content features
            f0: [B, T] F0 in Hz

        Returns:
            x: [B, hidden, T] encoded features
            m: [B, hidden, T] mean
            logs: [B, hidden, T] log standard deviation
        """
        # Project to hidden dimension
        x = self.pre(x)  # [B, hidden, T]

        # Quantize F0 to bins using mel-scale (matches reference RVC)
        # Convert Hz to mel scale, then normalize to 0-255 range
        # Mel formula: m = 1127 * ln(1 + f/700)
        # For f0 range ~50-1100 Hz, mel range is ~76-1071
        f0_mel = 1127.0 * torch.log1p(f0 / 700.0)  # [B, T]
        f0_mel_min = 1127.0 * math.log(1 + 50.0 / 700.0)   # ~76
        f0_mel_max = 1127.0 * math.log(1 + 1100.0 / 700.0)  # ~1071
        f0_norm = (f0_mel - f0_mel_min) / (f0_mel_max - f0_mel_min)  # [0, 1]
        f0_bins = torch.clamp(f0_norm * 255, 1, 255).long()  # [B, T], 1-255 (0 reserved for unvoiced)
        # Mark unvoiced frames (f0 <= 0) as bin 0
        f0_bins = torch.where(f0 > 0, f0_bins, torch.zeros_like(f0_bins))

        f0_emb = self.emb_pitch(f0_bins)  # [B, T, hidden]
        f0_emb = f0_emb.transpose(1, 2)  # [B, hidden, T]

        # Add F0 embedding and apply CRITICAL scaling (matches reference RVC)
        x = x + f0_emb
        x = x * math.sqrt(self.hidden_channels)  # Critical for proper gradient flow
        x = F.leaky_relu(x, 0.1)  # Activation after embedding (matches reference)

        # Process through encoder layers with residual connections
        for layer in self.encoder:
            x = x + layer(x)

        # Layer norm (need to transpose for LayerNorm)
        x = x.transpose(1, 2)  # [B, T, hidden]
        x = self.norm(x)
        x = x.transpose(1, 2)  # [B, hidden, T]

        # Project to mean and log-variance
        stats = self.proj(x)  # [B, hidden*2, T]
        m, logs = torch.split(stats, self.hidden_channels, dim=1)

        return x, m, logs


class PosteriorEncoder(nn.Module):
    """
    Encodes mel-spectrograms to latent space during training.

    Uses WaveNet-style dilated convolutions to process mel features
    and output latent z with mean and log-variance.

    Args:
        in_channels: Number of mel bins (80)
        hidden_channels: Hidden dimension
        out_channels: Output latent dimension
        kernel_size: Convolution kernel size
        dilation_rate: Base dilation rate
        n_layers: Number of WaveNet layers
        gin_channels: Speaker embedding dimension (0 = no speaker conditioning)
    """

    def __init__(
        self,
        in_channels: int = 80,
        hidden_channels: int = 192,
        out_channels: int = 192,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 16,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers

        # Input projection
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)

        # WaveNet-style layers with gated activation
        self.enc = nn.ModuleList()
        for i in range(n_layers):
            dilation = dilation_rate ** (i % 4)  # Cycle dilations: 1, 1, 1, 1, 2, 2, 2, 2, ...
            padding = (kernel_size - 1) * dilation // 2
            self.enc.append(
                nn.Conv1d(
                    hidden_channels,
                    hidden_channels * 2,  # For GLU
                    kernel_size,
                    padding=padding,
                    dilation=dilation
                )
            )

        # Output projection to mean and log-variance
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

        # Optional speaker conditioning
        self.gin_channels = gin_channels
        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, hidden_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode mel-spectrogram to latent.

        Args:
            x: [B, 80, T] mel-spectrogram
            g: [B, gin_channels, 1] speaker embedding (optional)

        Returns:
            z: [B, out_channels, T] sampled latent
            m: [B, out_channels, T] mean
            logs: [B, out_channels, T] log standard deviation
        """
        # Project to hidden dimension
        x = self.pre(x)  # [B, hidden, T]

        # Add speaker conditioning if provided
        if g is not None and self.gin_channels > 0:
            x = x + self.cond(g)

        # Process through WaveNet layers
        for layer in self.enc:
            residual = x
            x = layer(x)  # [B, hidden*2, T]
            # Gated Linear Unit activation
            x_tanh, x_sigmoid = torch.split(x, self.hidden_channels, dim=1)
            x = torch.tanh(x_tanh) * torch.sigmoid(x_sigmoid)
            x = x + residual  # Residual connection

        # Project to statistics
        stats = self.proj(x)  # [B, out*2, T]
        m, logs = torch.split(stats, self.out_channels, dim=1)

        # Sample using reparameterization trick
        z = m + torch.randn_like(m) * torch.exp(logs)

        return z, m, logs
