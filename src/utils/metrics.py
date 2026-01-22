# src/utils/metrics.py
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def compute_mcd(
    generated_mel: torch.Tensor,
    target_mel: torch.Tensor,
    n_mfcc: int = 13,
) -> float:

    # Ensure 3D tensors
    if generated_mel.dim() == 2:
        generated_mel = generated_mel.unsqueeze(0)
    if target_mel.dim() == 2:
        target_mel = target_mel.unsqueeze(0)

    # Align lengths
    min_len = min(generated_mel.shape[-1], target_mel.shape[-1])
    generated_mel = generated_mel[..., :min_len]
    target_mel = target_mel[..., :min_len]

    gen_mfcc = _mel_to_mfcc(generated_mel, n_mfcc)
    tgt_mfcc = _mel_to_mfcc(target_mel, n_mfcc)

    diff = gen_mfcc[:, 1:n_mfcc, :] - tgt_mfcc[:, 1:n_mfcc, :]  # Skip c0
    mcd_per_frame = torch.sqrt(2 * torch.sum(diff ** 2, dim=1))  # [B, T]

    # Average over frames and batch, convert to dB
    mcd = (10.0 / math.log(10)) * mcd_per_frame.mean()

    return float(mcd.item())


def _mel_to_mfcc(mel: torch.Tensor, n_mfcc: int) -> torch.Tensor:
    n_mels = mel.shape[1]
    n = torch.arange(n_mels, device=mel.device, dtype=mel.dtype)
    k = torch.arange(n_mfcc, device=mel.device, dtype=mel.dtype)

    # DCT-II: C[k,n] = cos(pi * k * (2n + 1) / (2N))
    dct_matrix = torch.cos(
        math.pi * k.unsqueeze(1) * (2 * n.unsqueeze(0) + 1) / (2 * n_mels)
    )  # [n_mfcc, n_mels]

    mel_t = mel.transpose(1, 2)  # [B, T, n_mels]
    mfcc_t = torch.matmul(mel_t, dct_matrix.T)  # [B, T, n_mfcc]
    mfcc = mfcc_t.transpose(1, 2)  # [B, n_mfcc, T]

    return mfcc


def compute_f0_rmse(
    generated_f0: torch.Tensor,
    target_f0: torch.Tensor,
    voiced_threshold: float = 50.0,
) -> float:
    # Flatten to 1D
    gen_f0 = generated_f0.flatten().float()
    tgt_f0 = target_f0.flatten().float()

    # Align lengths
    min_len = min(len(gen_f0), len(tgt_f0))
    gen_f0 = gen_f0[:min_len]
    tgt_f0 = tgt_f0[:min_len]

    voiced_mask = (gen_f0 > voiced_threshold) & (tgt_f0 > voiced_threshold)

    if voiced_mask.sum() == 0:
        return 0.0

    diff = gen_f0[voiced_mask] - tgt_f0[voiced_mask]
    rmse = torch.sqrt(torch.mean(diff ** 2))

    return float(rmse.item())


def compute_f0_correlation(
    generated_f0: torch.Tensor,
    target_f0: torch.Tensor,
    voiced_threshold: float = 50.0,
) -> float:

    # Flatten to 1D
    gen_f0 = generated_f0.flatten().float()
    tgt_f0 = target_f0.flatten().float()

    # Align lengths
    min_len = min(len(gen_f0), len(tgt_f0))
    gen_f0 = gen_f0[:min_len]
    tgt_f0 = tgt_f0[:min_len]

    voiced_mask = (gen_f0 > voiced_threshold) & (tgt_f0 > voiced_threshold)

    if voiced_mask.sum() < 2:
        return 0.0

    gen_voiced = gen_f0[voiced_mask]
    tgt_voiced = tgt_f0[voiced_mask]

    gen_mean = gen_voiced.mean()
    tgt_mean = tgt_voiced.mean()

    gen_centered = gen_voiced - gen_mean
    tgt_centered = tgt_voiced - tgt_mean

    numerator = (gen_centered * tgt_centered).sum()
    denominator = torch.sqrt((gen_centered ** 2).sum() * (tgt_centered ** 2).sum())

    if denominator < 1e-8:
        return 0.0

    correlation = numerator / denominator

    return float(correlation.item())


def compute_discriminator_accuracy(
    real_outputs: list[torch.Tensor],
    fake_outputs: list[torch.Tensor],
    threshold: float = 0.5,
) -> dict[str, float]:
    real_correct = 0
    real_total = 0
    fake_correct = 0
    fake_total = 0

    for real_out in real_outputs:
        real_pred = (torch.sigmoid(real_out) > threshold).float()
        real_correct += real_pred.sum().item()
        real_total += real_pred.numel()

    for fake_out in fake_outputs:
        fake_pred = (torch.sigmoid(fake_out) < threshold).float()
        fake_correct += fake_pred.sum().item()
        fake_total += fake_pred.numel()

    real_acc = real_correct / max(1, real_total)
    fake_acc = fake_correct / max(1, fake_total)
    overall_acc = (real_correct + fake_correct) / max(1, real_total + fake_total)

    return {
        "d_acc_real": real_acc,
        "d_acc_fake": fake_acc,
        "d_acc_overall": overall_acc,
    }


def compute_snr(
    generated: torch.Tensor,
    target: torch.Tensor,
) -> float:

    gen = generated.flatten().float()
    tgt = target.flatten().float()

    min_len = min(len(gen), len(tgt))
    gen = gen[:min_len]
    tgt = tgt[:min_len]

    # Normalize
    gen = gen - gen.mean()
    tgt = tgt - tgt.mean()

    signal_power = (tgt ** 2).mean()
    noise_power = ((gen - tgt) ** 2).mean()

    if noise_power < 1e-10:
        return 100.0 

    snr = 10 * torch.log10(signal_power / noise_power)

    return float(snr.item())


class MetricsTracker:
    def __init__(self):
        self._values: dict[str, list[float]] = {}

    def update(self, **kwargs: float) -> None:
        for name, value in kwargs.items():
            if name not in self._values:
                self._values[name] = []
            if not math.isnan(value) and not math.isinf(value):
                self._values[name].append(value)

    def compute(self) -> dict[str, float]:
        return {
            name: sum(values) / len(values) if values else 0.0
            for name, values in self._values.items()
        }

    def reset(self) -> None:
        self._values.clear()
