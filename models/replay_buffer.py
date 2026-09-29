"""
Quantized latent replay buffer.

Owner: Samuel (Phase 2 -- stubbed now so the module boundary exists early)

A fixed-size ring buffer of quantized encoder outputs, used to fine-tune the
decoder and shallow adaptation layers when drift is detected, without full
retraining. Ablation dimensions per the proposal: buffer size, quantization
bit-width (2/4/8-bit, plus 3/6-bit per the CPAL positioning notes), and
replay sampling strategy.

Not implemented yet -- Phase 1 work (data pipeline, static baseline) does
not depend on this file.
"""

import torch


class QuantizedReplayBuffer:
    def __init__(self, capacity: int, bit_width: int = 8):
        self.capacity = capacity
        self.bit_width = bit_width
        raise NotImplementedError("Samuel: Phase 2")

    def add(self, latents: torch.Tensor) -> None:
        raise NotImplementedError

    def sample(self, n: int) -> torch.Tensor:
        raise NotImplementedError
