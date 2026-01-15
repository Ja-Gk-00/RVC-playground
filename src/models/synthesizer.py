# src/models/synthesizer.py
"""Full RVC Synthesizer combining all components."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.encoders import TextEncoder, PosteriorEncoder
from src.models.flow import ResidualCouplingBlock
from src.models.generator_module import GeneratorNSF


class SynthesizerTrn(nn.Module):
    """
    Full RVC synthesizer with VAE-style training.

    Training path:
        content (HuBERT) -> enc_p -> z_p (prior)
        mel-spec -> enc_q -> z_q (posterior)
        z_q -> flow (inverse) -> z_p_hat
        z_q -> dec -> audio
        Loss = reconstruction + KL(z_q || z_p)

    Inference path:
        content -> enc_p -> z_p
        z_p -> flow (forward) -> z
        z -> dec -> audio

    Args:
        spec_channels: Number of mel-spectrogram bins
        segment_size: Segment size in frames for training
        inter_channels: Latent dimension
        hidden_channels: Hidden dimension for encoders
        filter_channels: Filter dimension for text encoder
        n_heads: Number of attention heads
        n_layers: Number of encoder layers
        kernel_size: Convolution kernel size
        p_dropout: Dropout probability
        resblock_kernel_sizes: Kernel sizes for generator residual blocks
        resblock_dilation_sizes: Dilation patterns for generator
        upsample_rates: Upsampling factors for generator
        upsample_initial_channel: Initial channel count for generator
        upsample_kernel_sizes: Kernel sizes for upsampling convolutions
        n_speakers: Number of speakers (0 = single speaker)
        gin_channels: Speaker embedding dimension
        sr: Sample rate
    """

    def __init__(
        self,
        spec_channels: int = 80,
        segment_size: int = 32,
        inter_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.0,
        resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        upsample_rates: tuple[int, ...] = (5, 4, 4, 4, 3),
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes: tuple[int, ...] = (10, 8, 8, 8, 6),
        n_speakers: int = 0,
        gin_channels: int = 256,
        sr: int = 48000,
    ) -> None:
        super().__init__()
        self.spec_channels = spec_channels
        self.segment_size = segment_size
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels if n_speakers > 0 else 0
        self.sr = sr

        # Content encoder (HuBERT -> latent prior)
        self.enc_p = TextEncoder(
            in_channels=768,  # HuBERT dimension
            hidden_channels=inter_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
        )

        # Posterior encoder (mel -> latent)
        self.enc_q = PosteriorEncoder(
            in_channels=spec_channels,
            hidden_channels=inter_channels,
            out_channels=inter_channels,
            n_layers=16,
            gin_channels=self.gin_channels,
        )

        # Flow (transforms between prior and posterior)
        self.flow = ResidualCouplingBlock(
            channels=inter_channels,
            hidden_channels=inter_channels,
            n_flows=4,
            gin_channels=self.gin_channels,
        )

        # Decoder (NSF generator)
        self.dec = GeneratorNSF(
            initial_channel=inter_channels,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_rates=upsample_rates,
            upsample_initial_channel=upsample_initial_channel,
            upsample_kernel_sizes=upsample_kernel_sizes,
            gin_channels=self.gin_channels,
            sr=sr,
        )

        # Speaker embedding
        if n_speakers > 0:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def forward(
        self,
        content: torch.Tensor,
        f0: torch.Tensor,
        mel: torch.Tensor,
        sid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Training forward pass.

        Args:
            content: [B, 768, T] HuBERT content features
            f0: [B, T] F0 in Hz
            mel: [B, 80, T] mel-spectrogram
            sid: [B] speaker IDs (optional)

        Returns:
            Dictionary containing:
                - audio: [B, T*upp] generated audio
                - m_p: [B, inter, T] prior mean
                - logs_p: [B, inter, T] prior log-std
                - m_q: [B, inter, T] posterior mean
                - logs_q: [B, inter, T] posterior log-std
                - z_p: [B, inter, T] prior latent
                - z_q: [B, inter, T] posterior latent
        """
        # Get speaker embedding
        g = None
        if self.n_speakers > 0 and sid is not None:
            g = self.emb_g(sid).unsqueeze(-1)  # [B, gin, 1]

        # Encode content -> prior statistics
        _, m_p, logs_p = self.enc_p(content, f0)

        # Encode mel -> posterior sample
        z_q, m_q, logs_q = self.enc_q(mel, g)

        # Flow: transform posterior to prior space for KL
        z_p = self.flow(z_q, g, reverse=False)

        # Decode from posterior sample
        audio = self.dec(z_q, f0, g)

        return {
            "audio": audio,
            "m_p": m_p,
            "logs_p": logs_p,
            "m_q": m_q,
            "logs_q": logs_q,
            "z_p": z_p,
            "z_q": z_q,
        }

    @torch.inference_mode()
    def infer(
        self,
        content: torch.Tensor,
        f0: torch.Tensor,
        sid: torch.Tensor | None = None,
        noise_scale: float = 0.66666,  # Match reference RVC coefficient
    ) -> torch.Tensor:
        """
        Inference forward pass.

        Args:
            content: [B, 768, T] HuBERT content features
            f0: [B, T] F0 in Hz
            sid: [B] speaker IDs (optional)
            noise_scale: Scale for sampling noise (0 = deterministic)

        Returns:
            [B, T*upp] generated audio
        """
        # Get speaker embedding
        g = None
        if self.n_speakers > 0 and sid is not None:
            g = self.emb_g(sid).unsqueeze(-1)

        # Encode content -> prior
        _, m_p, logs_p = self.enc_p(content, f0)

        # Sample from prior
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        # Flow: transform to decoder space
        z = self.flow(z_p, g, reverse=True)

        # Decode
        audio = self.dec(z, f0, g)

        return audio
