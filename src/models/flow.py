# src/models/flow.py
"""Normalizing flow modules for RVC."""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualCouplingLayer(nn.Module):
    """
    Single coupling layer for normalizing flow.

    Splits input channels in half, uses first half to predict transformation
    parameters for second half. This creates an invertible transformation.

    Args:
        channels: Total number of channels (will be split in half)
        hidden_channels: Hidden dimension for internal processing
        kernel_size: Convolution kernel size
        dilation_rate: Base dilation rate
        n_layers: Number of WaveNet layers
        gin_channels: Speaker embedding dimension (0 = no conditioning)
        mean_only: If True, only predict mean (simpler, more stable)
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 4,
        gin_channels: int = 0,
        mean_only: bool = True,
    ) -> None:
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.hidden_channels = hidden_channels

        # Input projection
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)

        # WaveNet-style encoder
        self.enc = nn.ModuleList()
        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = (kernel_size - 1) * dilation // 2
            self.enc.append(
                nn.Conv1d(
                    hidden_channels,
                    hidden_channels * 2,  # For GLU
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
            )

        # Output projection
        out_channels = self.half_channels if mean_only else self.half_channels * 2
        self.post = nn.Conv1d(hidden_channels, out_channels, 1)

        # Initialize output to zero for stable training start
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

        # Optional speaker conditioning
        self.gin_channels = gin_channels
        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, hidden_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor | None = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        """
        Forward or reverse pass through coupling layer.

        Args:
            x: [B, channels, T] input tensor
            g: [B, gin_channels, 1] speaker embedding (optional)
            reverse: If True, compute inverse transformation

        Returns:
            [B, channels, T] transformed tensor
        """
        # Split channels in half
        x0, x1 = torch.split(x, self.half_channels, dim=1)

        # Process first half to get transformation parameters
        h = self.pre(x0)  # [B, hidden, T]

        # Add speaker conditioning
        if g is not None and self.gin_channels > 0:
            h = h + self.cond(g)

        # WaveNet layers
        for layer in self.enc:
            residual = h
            h = layer(h)
            # GLU activation
            h_tanh, h_sigmoid = torch.split(h, self.hidden_channels, dim=1)
            h = torch.tanh(h_tanh) * torch.sigmoid(h_sigmoid)
            h = h + residual

        # Get transformation parameters
        stats = self.post(h)

        if self.mean_only:
            m = stats
            logs = torch.zeros_like(m)
        else:
            m, logs = torch.split(stats, self.half_channels, dim=1)

        # Apply transformation
        if not reverse:
            # Forward: x1_out = x1 * exp(logs) + m
            x1 = m + x1 * torch.exp(logs)
        else:
            # Reverse: x1_out = (x1 - m) * exp(-logs)
            x1 = (x1 - m) * torch.exp(-logs)

        return torch.cat([x0, x1], dim=1)


class ResidualCouplingBlock(nn.Module):
    """
    Stack of coupling layers with channel flipping.

    Multiple coupling layers are stacked, with channels flipped between
    each layer to ensure all channels can be transformed.

    Args:
        channels: Total number of channels
        hidden_channels: Hidden dimension for coupling layers
        kernel_size: Convolution kernel size
        dilation_rate: Base dilation rate
        n_layers: Number of WaveNet layers per coupling layer
        n_flows: Number of coupling layers in the block
        gin_channels: Speaker embedding dimension
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 4,
        n_flows: int = 4,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        self.n_flows = n_flows

        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels,
                    mean_only=True,  # More stable training
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor | None = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        """
        Forward or reverse pass through flow block.

        Args:
            x: [B, channels, T] input tensor
            g: [B, gin_channels, 1] speaker embedding (optional)
            reverse: If True, compute inverse transformation

        Returns:
            [B, channels, T] transformed tensor
        """
        if not reverse:
            # Forward pass
            for flow in self.flows:
                x = flow(x, g, reverse=False)
                x = torch.flip(x, dims=[1])  # Flip channels
        else:
            # Reverse pass (reverse order)
            for flow in reversed(self.flows):
                x = torch.flip(x, dims=[1])  # Flip channels first
                x = flow(x, g, reverse=True)

        return x
