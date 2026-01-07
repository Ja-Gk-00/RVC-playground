from __future__ import annotations

import torch
import torch.nn as nn


class GeneratorModule(nn.Module):
    """
    Minimal waveform generator conditioned on (content units) + optional pitch.

    Contract:
      content_units: [B, T, C]
      pitch:        [B, T] or [B, T, 1] (Hz)
      output:       [B, T * upsample_factor] in [-1, 1]
    """

    def __init__(
        self,
        content_dim: int = 768,  # RVC v2 uses 768-dim features. :contentReference[oaicite:3]{index=3}
        hidden_dim: int = 192,
        use_pitch: bool = True,
        content_sr: int = 16000,
        hop_length: int = 320,
        target_sr: int = 48000,
    ) -> None:
        super().__init__()
        self.content_dim = int(content_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_pitch = bool(use_pitch)
        self.content_sr = int(content_sr)
        self.hop_length = int(hop_length)
        self.target_sr = int(target_sr)

        self.upsample_factor = int(round(self.target_sr * self.hop_length / float(self.content_sr)))

        self.content_conv = nn.Conv1d(self.content_dim, self.hidden_dim, kernel_size=1)
        self.pitch_conv = nn.Conv1d(1, self.hidden_dim, kernel_size=1) if self.use_pitch else None

        self.pre = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
        )

        self.upsample_layers = self._create_upsample_layers(self.upsample_factor, self.hidden_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.1)

    @staticmethod
    def _split_factor(upsample_factor: int) -> list[int]:
        # Prefer smaller strides for stability.
        if upsample_factor >= 24:
            a = 4
            b = 4
            c = max(1, upsample_factor // 16)
            return [a, b, c]
        return [upsample_factor]

    def _create_upsample_layers(self, upsample_factor: int, hidden_dim: int) -> nn.ModuleList:
        factors = self._split_factor(int(upsample_factor))

        layers = nn.ModuleList()
        in_ch = int(hidden_dim)

        out_ch_seq = [128, 64, 1] if len(factors) == 3 else [64, 1]
        for i, s in enumerate(factors):
            out_ch = out_ch_seq[i] if i < len(out_ch_seq) else max(1, in_ch // 2)

            padding = (s + 1) // 2
            output_padding = (2 * padding) - s  # ensures exact length = in_len * stride

            layers.append(
                nn.ConvTranspose1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=2 * s,
                    stride=s,
                    padding=padding,
                    output_padding=output_padding,
                )
            )
            in_ch = out_ch

        return layers

    def forward(self, content_units: torch.Tensor, pitch: torch.Tensor | None = None) -> torch.Tensor:
        if content_units.dim() == 2:
            content_units = content_units.unsqueeze(0)
        if content_units.dim() != 3:
            raise ValueError("content_units must have shape [B, T, C] or [T, C].")

        x = content_units.transpose(1, 2)  # [B, C, T]
        x = self.content_conv(x)  # [B, H, T]

        if self.use_pitch:
            if pitch is None:
                pitch = torch.zeros((content_units.size(0), content_units.size(1)), device=content_units.device)

            if pitch.dim() == 1:
                pitch = pitch.unsqueeze(0)
            if pitch.dim() == 2:
                p = pitch.unsqueeze(1)  # [B, 1, T]
            elif pitch.dim() == 3:
                p = pitch.transpose(1, 2)  # [B, 1, T] if [B, T, 1]
            else:
                raise ValueError("pitch must have shape [B, T], [T] or [B, T, 1].")

            if self.pitch_conv is None:
                raise RuntimeError("Internal error: pitch_conv is None while use_pitch=True.")

            p_embed = self.pitch_conv(p.to(dtype=x.dtype))
            if p_embed.shape[-1] != x.shape[-1]:
                m = min(int(p_embed.shape[-1]), int(x.shape[-1]))
                p_embed = p_embed[..., :m]
                x = x[..., :m]
            x = x + p_embed

        x = self.pre(x)

        for layer in self.upsample_layers:
            x = layer(x)
            if layer is not self.upsample_layers[-1]:
                x = self.activation(x)

        # Keep output bounded; helps avoid explosion.
        x = torch.tanh(x)
        return x.squeeze(1)  # [B, n_samples]
