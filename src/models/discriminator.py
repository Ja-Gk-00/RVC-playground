# src/models/discriminator.py
"""
RVC Discriminators for GAN training.

RVC uses a combined discriminator structure:
- DiscriminatorS (scale discriminator) as first element
- Multiple DiscriminatorP (period discriminators) with periods [2,3,5,7,11,17,23,37] for v2

This matches the pretrained f0D48k.pth structure.
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class DiscriminatorS(nn.Module):
    """Scale discriminator - analyzes audio at different scales."""

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorP(nn.Module):
    """Period discriminator - analyzes audio reshaped by period."""

    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3, use_spectral_norm: bool = False):
        super().__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []

        # Reshape by period: [B, 1, T] -> [B, 1, T//period, period]
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminatorV2(nn.Module):
    """
    Combined Multi-Period and Multi-Scale Discriminator (v2).

    This is the structure used in RVC v2 pretrained models (f0D48k.pth).
    It combines DiscriminatorS with multiple DiscriminatorP instances.
    """

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        # V2 uses these periods
        periods = [2, 3, 5, 7, 11, 17, 23, 37]

        # First element is scale discriminator, rest are period discriminators
        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs += [DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[
        List[torch.Tensor], List[torch.Tensor],
        List[List[torch.Tensor]], List[List[torch.Tensor]]
    ]:
        """
        Args:
            y: real audio [B, 1, T]
            y_hat: generated audio [B, 1, T]

        Returns:
            y_d_rs: real discriminator outputs
            y_d_gs: generated discriminator outputs
            fmap_rs: real feature maps
            fmap_gs: generated feature maps
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# Alias for backwards compatibility
MultiPeriodDiscriminator = MultiPeriodDiscriminatorV2


def discriminator_loss(disc_real_outputs: List[torch.Tensor], disc_generated_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """
    Discriminator loss - real should be 1, generated should be 0.

    Uses least-squares GAN loss for stability.
    """
    loss = 0
    r_losses = []
    g_losses = []

    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Generator loss - wants discriminator to output 1 for generated samples.
    """
    loss = 0
    gen_losses = []

    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l.item())
        loss += l

    return loss, gen_losses


def feature_matching_loss(fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]) -> torch.Tensor:
    """
    Feature matching loss - generator should produce features similar to real audio.

    This helps stabilize training and improves quality.
    """
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2  # Scale factor from reference
