"""
Evaluation: AUC, forgetting rate, adaptation speed, memory footprint, latency.

Owner: shared (Phase 3)

Per the proposal's Section 5 (Metrics):
- Anomaly detection AUC on held-out anomalous segments.
- Forgetting rate: drop in detection performance on early normal patterns
  after adapting to later ones.
- Adaptation speed: number of observations needed to reliably detect a
  newly emerged anomaly type after its introduction.
- Memory footprint: encoder + decoder + replay buffer + NCM discriminator,
  combined, checked against the 256 KB SRAM ceiling.
- Inference and update latency per observation on target hardware.

Not implemented yet -- comes together once train_baseline.py, the replay
buffer, and the NCM discriminator all exist.
"""


def compute_auc(scores, labels):
    raise NotImplementedError("Phase 3")


def compute_forgetting_rate(model, early_data, late_data):
    raise NotImplementedError("Phase 3")


def compute_adaptation_speed(model, stream, new_anomaly_onset_idx):
    raise NotImplementedError("Phase 3")


def measure_memory_footprint(encoder, decoder, replay_buffer, ncm):
    raise NotImplementedError("Phase 3/4")
