"""
Train the static 1D conv autoencoder baseline.

Owner: Ekene

Phase 1 deliverable: train on the stationary/pretrain segment only (see
data/splits.py), report reconstruction MSE on a held-out slice of that same
segment, and log parameter count. No replay, no quantization, no drift
handling here -- this is the "static autoencoder, no continual learning"
baseline referenced in the proposal's Section 5 (Baselines).
"""

import torch
from torch.utils.data import DataLoader

from data.loaders import load_mimii, load_skab
from data.splits import make_temporal_split
from models.autoencoder import Conv1DAutoencoder, reconstruction_error, count_parameters


def train(
    dataset_name: str = "mimii",
    data_root: str = "./raw_data",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
):
    """
    TODO(Ekene):
    - Wire up a real Dataset/DataLoader around SensorSequence objects.
    - Train only on the pretrain segment's normal samples.
    - Report: final reconstruction MSE (train + held-out), parameter count,
      rough training time. Save the trained weights somewhere the team can
      load them from (Phase 2 needs the frozen encoder).
    """
    loader_fn = load_mimii if dataset_name == "mimii" else load_skab
    sequences = list(loader_fn(data_root))
    split = make_temporal_split(sequences)

    model = Conv1DAutoencoder()
    print(f"Model parameter count: {count_parameters(model)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    raise NotImplementedError("Ekene: implement the training loop")


if __name__ == "__main__":
    train()
