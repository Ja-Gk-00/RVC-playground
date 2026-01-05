import torch
import torch.nn as nn


class GeneratorModule(nn.Module):
    def __init__(
        self,
        content_dim: int = 256,
        hidden_dim: int = 128,
        use_pitch: bool = True,
        content_sr: int = 16000,
        hop_length: int = 320,
        target_sr: int = 48000,
    ):
        super(GeneratorModule, self).__init__()
        self.content_dim = content_dim
        self.hidden_dim = hidden_dim
        self.use_pitch = use_pitch
        self.content_sr = content_sr
        self.hop_length = hop_length
        self.target_sr = target_sr

        self.upsample_factor = int(round(target_sr * hop_length / float(content_sr)))

        self.content_conv = nn.Conv1d(content_dim, hidden_dim, kernel_size=1)
        self.pitch_conv = nn.Conv1d(1, hidden_dim, kernel_size=1) if use_pitch else None

        self.upsample_layers = self._create_upsample_layers(self.upsample_factor, hidden_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        
    def _create_upsample_layers(
        self,
        upsample_factor: int,
        hidden_dim: int,
    ) -> nn.ModuleList:
        factors = (
            [5, 5, upsample_factor // 25]
            if upsample_factor >= 25
            else [upsample_factor]
        )
        
        upsample_layers = nn.ModuleList()
        in_channels = hidden_dim
        out_channels_sequence = [64, 32, 1] if len(factors) == 3 else [32, 1]

        for i, factor in enumerate(factors):
            out_channels = (
                out_channels_sequence[i]
                if i < len(out_channels_sequence)
                else max(1, in_channels // 2)
            )
            kernel_size = factor * 2
            layer = nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size=kernel_size, stride=factor, padding=(factor // 2)
            )
            upsample_layers.append(layer)
            in_channels = out_channels
        
        return upsample_layers

    def forward(self, content_units: torch.Tensor, pitch: torch.Tensor | None = None) -> torch.Tensor:
        if content_units.dim() == 2:
            content_units = content_units.unsqueeze(0)
        
        x = content_units.transpose(1, 2)  # [batch, content_dim, n_frames]
        x = self.content_conv(x)  # [batch, hidden_dim, n_frames]
        
        if self.use_pitch and pitch is not None:
            pitch_reshaped = pitch.unsqueeze(1) if pitch.dim() == 2 else pitch.transpose(1, 2)  # [batch, 1, n_frames]
            pitch_embed = self.pitch_conv(pitch_reshaped.to(x.dtype))  # [batch, hidden_dim, n_frames]
            
            if pitch_embed.shape[2] != x.shape[2]:
                min_len = min(pitch_embed.shape[2], x.shape[2])
                pitch_embed = pitch_embed[:, :, :min_len]
                x = x[:, :, :min_len]
            
            x = x + pitch_embed  # [batch, hidden_dim, n_frames]

        for layer in self.upsample_layers:
            x = layer(x)  # Progressive upsampling: [batch, channels, n_samples]
            if layer is not self.upsample_layers[-1]:
                x = self.activation(x)

        waveform = x.squeeze(1)  # [batch, n_samples]
        return waveform

