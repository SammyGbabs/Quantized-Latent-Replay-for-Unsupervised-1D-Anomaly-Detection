"""
NCM-style drift discriminator.

Owner: Samuel (Phase 2 -- stubbed now so the module boundary exists early)

Maintains cluster centroids over reconstruction embeddings. New observations
are compared to the nearest centroid; far-from-all-centroids observations
are candidate anomalies, and clusters that keep receiving novel-but-consistent
observations spawn a new cluster (drift). Needs a concrete spawning rule
(distance threshold, minimum count, consistency window) before implementation.

Not implemented yet -- Phase 1 work does not depend on this file.
"""

import torch


class NCMDiscriminator:
    def __init__(self, distance_threshold: float, min_count: int = 20):
        self.distance_threshold = distance_threshold
        self.min_count = min_count
        raise NotImplementedError("Samuel: Phase 2")

    def score(self, embedding: torch.Tensor) -> float:
        """Distance to nearest centroid."""
        raise NotImplementedError

    def update(self, embedding: torch.Tensor) -> bool:
        """Update cluster state; return True if a new cluster was spawned."""
        raise NotImplementedError
