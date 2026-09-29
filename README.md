# QLR-1D: Quantized Latent Replay for Unsupervised 1D Anomaly Detection

Extending quantized latent replay (QLR) continual learning to unsupervised
anomaly detection on 1D sensor streams (MIMII, SKAB), targeting Cortex-M4/M7
class MCUs under a 256 KB SRAM budget.

See the team proposal doc (Working Implementation version) for full context,
scope, phases, and roles. This README only tracks code structure.

## Repo layout

```
qlr-1d-anomaly/
├── data/
│   ├── loaders.py        # MIMII / SKAB loading                  (Taiwo)
│   └── splits.py         # sequential temporal splits             (Taiwo)
├── models/
│   ├── autoencoder.py    # 1D conv autoencoder backbone           (Ekene)
│   ├── replay_buffer.py  # quantized latent replay buffer         (Samuel)
│   └── ncm.py            # NCM-style drift discriminator          (Samuel)
├── train_baseline.py     # trains the static AE baseline          (Ekene)
├── evaluate.py           # AUC / forgetting / adaptation metrics  (shared, Phase 3)
└── notebooks/            # exploration, not for production code
```

## Phase 1 ownership (current)

- **Taiwo** — `data/loaders.py`, `data/splits.py`
- **Ekene** — `models/autoencoder.py`, `train_baseline.py`
- **Samuel** — `models/replay_buffer.py`, `models/ncm.py` (Phase 2, stubbed here early)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Conventions

- All models take/return `torch.Tensor` of shape `(batch, channels, length)`.
- No shuffling across the stationary/streaming split boundary — order matters.
- Keep parameter counts and memory footprints logged; this project lives or
  dies on the accuracy-vs-memory tradeoff.
