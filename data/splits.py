"""
Sequential temporal splits.

Owner: Taiwo

Per the proposal (Section 5): splits must be sequential, not random, so the
streaming nature of the problem is preserved. Each dataset is divided into:
  1. a stationary pretraining segment (used to train the static AE baseline)
  2. a streaming segment with introduced or naturally occurring drift

Do not shuffle across the boundary between these two segments.
"""

from dataclasses import dataclass
from typing import Sequence
from data.loaders import SensorSequence


@dataclass
class TemporalSplit:
    pretrain: list[SensorSequence]
    stream: list[SensorSequence]


def make_temporal_split(
    sequences: Sequence[SensorSequence],
    pretrain_fraction: float = 0.4,
) -> TemporalSplit:
    """
    Split sequences (already in time order) into a pretraining segment and a
    streaming segment.

    Args:
        sequences: ordered sequences from a loader in data/loaders.py.
        pretrain_fraction: fraction of the (time-ordered) data to use as the
            stationary pretraining segment. Remainder is the streaming
            segment.

    Returns:
        TemporalSplit with both segments in original order.

    TODO(Taiwo):
    - Confirm pretrain_fraction default with the team once we've looked at
      segment lengths for MIMII/SKAB.
    - Add a report_split_stats() helper: segment lengths, anomaly ratios per
      segment, so we can sanity-check the drift setup before training.
    """
    raise NotImplementedError("Taiwo: implement temporal split logic")


def report_split_stats(split: TemporalSplit) -> dict:
    """
    Return basic stats for a sanity check: lengths and anomaly ratios of
    each segment.
    """
    raise NotImplementedError("Taiwo: implement stats reporting")
