"""
1D convolutional autoencoder backbone.

Owner: Ekene

Phase 1 scope: a plain static autoencoder, no quantization, no replay logic.
Trained on nominal (non-anomalous) data to reconstruct in-distribution
sequences with low reconstruction error -- that reconstruction error is the
anomaly signal used everywhere downstream.

Keep this backbone MCU-sized: this is meant to eventually run on a Cortex-M4/M7
under a 256 KB SRAM budget (encoder + decoder + replay buffer + NCM combined),
so avoid making the backbone alone unreasonably large. A few conv layers with
modest channel counts is the right starting point, not a deep stack.
"""

import torch
import torch.nn as nn


class Conv1DAutoencoder(nn.Module):
    """
    Minimal 1D conv autoencoder.

    Input/output shape: (batch, channels, length)

    TODO(Ekene):
    - Pick channel counts / depth appropriate for MCU deployment (log
      parameter count as you go -- this matters for the 256 KB budget later).
    - Confirm input length and channel count once Taiwo's loaders are ready.
    - Expose `.encode()` separately from `.forward()` -- the replay buffer
      (Phase 2, Samuel) needs access to the raw encoder output before the
      decoder, so keep that boundary clean.
    """

    def __init__(self, in_channels: int = 1, latent_channels: int = 8):
        super().__init__()
        # Placeholder architecture -- replace with a real sized design.
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, latent_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_channels, 16, kernel_size=5, stride=2,
                                padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, in_channels, kernel_size=5, stride=2,
                                padding=2, output_padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def reconstruction_error(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Per-sample MSE, shape (batch,). Used as the anomaly score."""
    return ((x - x_hat) ** 2).flatten(1).mean(dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
