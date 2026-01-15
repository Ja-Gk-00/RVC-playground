# src/models/generator_module.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrizations

from src.models.source_module import SourceModuleHnNSF


def weight_norm(module: nn.Module, name: str = "weight", dim: int = 0) -> nn.Module:
    """Apply weight normalization using new parametrizations API."""
    return parametrizations.weight_norm(module, name=name, dim=dim)


def remove_weight_norm(module: nn.Module, name: str = "weight") -> nn.Module:
    """Remove weight normalization using new parametrizations API."""
    return parametrizations.remove_parametrizations(module, name)


@dataclass(frozen=True)
class GeneratorConfig:
    cond_dim: int
    upsample_rates: tuple[int, ...]
    upsample_kernel_sizes: tuple[int, ...]
    upsample_initial_channel: int = 512
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, int, int], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    pre_kernel_size: int = 7
    post_kernel_size: int = 7
    leaky_slope: float = 0.1


def _default_upsample_for_samples_per_frame(samples_per_frame: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if samples_per_frame == 960:
        rates = (5, 4, 4, 4, 3)
        ks = tuple(r * 2 for r in rates)
        return rates, ks

    remaining = int(samples_per_frame)
    rates_list: list[int] = []
    for f in (5, 4, 3, 2):
        while remaining % f == 0 and len(rates_list) < 8:
            rates_list.append(f)
            remaining //= f

    if remaining != 1:
        rates_list = [int(samples_per_frame)]

    rates = tuple(rates_list)
    ks = tuple(max(4, r * 2) for r in rates)
    return rates, ks


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: tuple[int, int, int], leaky_slope: float) -> None:
        super().__init__()
        self.leaky_slope = float(leaky_slope)
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        padding=((kernel_size - 1) // 2) * d,
                        dilation=d,
                    )
                )
                for d in dilations
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        padding=(kernel_size - 1) // 2,
                        dilation=1,
                    )
                )
                for _ in dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2, strict=True):
            xt = F.leaky_relu(x, self.leaky_slope)
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.leaky_slope)
            xt = c2(xt)
            x = xt + x
        return x


class GeneratorModule(nn.Module):
    def __init__(
        self,
        config: GeneratorConfig | None = None,
        *,
        content_dim: int | None = None,
        cond_dim: int | None = None,
        content_sr: int = 16000,
        hop_length: int = 320,
        target_sr: int = 48000,
        upsample_initial_channel: int = 512,
        **_: Any,
    ) -> None:
        super().__init__()

        if config is None:
            cd = int(cond_dim) if cond_dim is not None else (int(content_dim) if content_dim is not None else None)
            if cd is None:
                raise TypeError("GeneratorModule requires 'config' or 'content_dim'/'cond_dim'.")

            samples_per_frame = int(round(int(target_sr) * int(hop_length) / float(int(content_sr))))
            rates, ks = _default_upsample_for_samples_per_frame(samples_per_frame)

            config = GeneratorConfig(
                cond_dim=cd,
                upsample_rates=rates,
                upsample_kernel_sizes=ks,
                upsample_initial_channel=int(upsample_initial_channel),
            )

        self.cfg = config
        self.leaky_slope = float(config.leaky_slope)

        ch = int(config.upsample_initial_channel)
        self.conv_pre = weight_norm(
            nn.Conv1d(config.cond_dim, ch, kernel_size=config.pre_kernel_size, padding=(config.pre_kernel_size - 1) // 2)
        )

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        cur_ch = ch
        for i, (u, k) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes, strict=True)):
            next_ch = max(8, cur_ch // 2)
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        cur_ch,
                        next_ch,
                        kernel_size=int(k),
                        stride=int(u),
                        padding=int((k - u) // 2),
                    )
                )
            )
            for rk, rd in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes, strict=True):
                self.resblocks.append(ResBlock(next_ch, int(rk), tuple(int(x) for x in rd), self.leaky_slope))
            cur_ch = next_ch

        self.num_kernels = len(config.resblock_kernel_sizes)
        self.conv_post = weight_norm(
            nn.Conv1d(cur_ch, 1, kernel_size=config.post_kernel_size, padding=(config.post_kernel_size - 1) // 2)
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        # cond: [B, T, cond_dim]
        if cond.dim() != 3:
            raise ValueError("GeneratorModule expects cond with shape [B, T, cond_dim].")

        x = cond.transpose(1, 2)  # [B, cond_dim, T]
        x = self.conv_pre(x)

        rb_i = 0
        for up in self.ups:
            x = F.leaky_relu(x, self.leaky_slope)
            x = up(x)

            xs = 0.0
            for _ in range(self.num_kernels):
                xs = xs + self.resblocks[rb_i](x)
                rb_i += 1
            x = xs / float(self.num_kernels)

        x = F.leaky_relu(x, self.leaky_slope)
        x = self.conv_post(x)
        x = torch.tanh(x)  # [-1, 1]

        return x.squeeze(1)  # [B, N]


class GeneratorNSF(nn.Module):
    """
    NSF-based HiFi-GAN generator that injects harmonic sources.

    Key difference from vanilla HiFi-GAN: At each upsampling layer,
    we add the appropriately-downsampled harmonic source signal.
    This provides explicit pitch information to guide synthesis.

    Args:
        initial_channel: Input channel dimension (from encoder latent)
        resblock_kernel_sizes: Kernel sizes for residual blocks
        resblock_dilation_sizes: Dilation patterns for residual blocks
        upsample_rates: Upsampling factors for each layer
        upsample_initial_channel: Initial channel count after pre-conv
        upsample_kernel_sizes: Kernel sizes for upsampling convolutions
        gin_channels: Speaker embedding dimension (0 = no speaker conditioning)
        sr: Sample rate
        is_half: Whether to use half precision
    """

    def __init__(
        self,
        initial_channel: int = 192,
        resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        upsample_rates: tuple[int, ...] = (5, 4, 4, 4, 3),
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes: tuple[int, ...] = (10, 8, 8, 8, 6),
        gin_channels: int = 0,
        sr: int = 48000,
        is_half: bool = False,
    ) -> None:
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.sr = sr
        self.is_half = is_half

        # Calculate total upsampling factor
        self.upp = int(np.prod(upsample_rates))

        # Source module for F0 -> harmonic source
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sr,
            harmonic_num=0,
        )

        # Pre-conv (input from encoder latents)
        self.conv_pre = nn.Conv1d(
            initial_channel,
            upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_ch = upsample_initial_channel // (2 ** i)
            out_ch = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        in_ch,
                        out_ch,
                        kernel_size=k,
                        stride=u,
                        padding=(k - u) // 2,
                    )
                )
            )

        # Noise convolutions - one per upsampling layer
        # These process the harmonic source at appropriate resolutions
        self.noise_convs = nn.ModuleList()
        for i in range(len(upsample_rates)):
            out_ch = upsample_initial_channel // (2 ** (i + 1))
            if i + 1 < len(upsample_rates):
                # Downsample source to match this layer's resolution
                stride_f0 = int(np.prod(upsample_rates[i + 1:]))
                kernel_f0 = stride_f0 * 2
                padding_f0 = stride_f0 // 2
            else:
                # Last layer - no downsampling needed
                stride_f0 = 1
                kernel_f0 = 1
                padding_f0 = 0
            self.noise_convs.append(
                nn.Conv1d(
                    1,
                    out_ch,
                    kernel_size=kernel_f0,
                    stride=stride_f0,
                    padding=padding_f0,
                )
            )

        # ResBlocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d, leaky_slope=0.1))

        # Post-conv
        final_ch = upsample_initial_channel // (2 ** len(self.ups))
        self.conv_post = nn.Conv1d(final_ch, 1, kernel_size=7, stride=1, padding=3, bias=False)

        # Optional speaker conditioning
        self.gin_channels = gin_channels
        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(
        self,
        x: torch.Tensor,
        f0: torch.Tensor,
        g: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate audio from latent and F0.

        Args:
            x: [B, hidden, T] latent from encoder
            f0: [B, T] F0 in Hz
            g: [B, gin_channels, 1] speaker embedding (optional)

        Returns:
            [B, T*upp] generated audio
        """
        # Generate harmonic source at full audio rate
        f0_input = f0.unsqueeze(1)  # [B, 1, T]
        har_source, noi_source, uv = self.m_source(f0_input, self.upp)
        # har_source: [B, 1, T*upp]

        # Pre-conv
        x = self.conv_pre(x)  # [B, upsample_initial, T]

        # Add speaker embedding if provided
        if g is not None and self.gin_channels > 0:
            x = x + self.cond(g)

        # Upsample with source injection
        for i, (up, noise_conv) in enumerate(zip(self.ups, self.noise_convs)):
            x = F.leaky_relu(x, 0.1)
            x = up(x)

            # Add harmonic source at this resolution
            x_source = noise_conv(har_source)

            # Align sizes (transposed conv can produce off-by-one differences)
            min_len = min(x.shape[2], x_source.shape[2])
            x = x[:, :, :min_len]
            x_source = x_source[:, :, :min_len]

            x = x + x_source

            # ResBlocks
            xs = torch.zeros_like(x)
            for j in range(self.num_kernels):
                xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # Post-conv
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x.squeeze(1)  # [B, T*upp]

    def remove_all_weight_norm(self) -> None:
        """Remove weight normalization for inference."""
        for up in self.ups:
            remove_weight_norm(up)
        for block in self.resblocks:
            for conv in block.convs1:
                remove_weight_norm(conv)
            for conv in block.convs2:
                remove_weight_norm(conv)
