# ASV

This repository contains a refactored version of the multi-boat sectors environment and
Double-DQN training loop. The codebase is now organised as a Python package named
`asv`, which groups environment logic, configuration dataclasses, hyperparameters and
training utilities into focused modules.

## Project layout

```
asv/
  __init__.py            # Public package exports
  agent.py               # Double-DQN agent implementation
  boat.py                # Vessel dynamics and control sessions
  config.py              # Environment and spawn configuration dataclasses
  env.py                 # Multi-boat sectors environment
  hyperparams.py         # Global hyperparameter definitions and helpers
  models.py              # Neural network architectures
  replay.py              # Experience replay buffer
  training.py            # Training loop orchestration
scripts/
  train.py               # Command-line entry point
env3.py                  # Backwards compatible exports for legacy imports
train3.py                # Delegates to ``scripts/train.py``
```

All default values used across the experiment are centralised in
`asv/hyperparams.py`. The `GlobalHyperParameters` dataclass gathers the training,
environment, boat, turn-session and spawn settings in one place, and the CLI exposes
all training hyperparameters for command-line overrides.

## Training

Run the trainer via the helper script:

```bash
python scripts/train.py --episodes 100 --steps_per_episode 500 --print-hparams
```

Pass `--print-hparams` to print the resolved hyperparameters before training starts.
Legacy entry points continue to work:

```bash
python train3.py --episodes 100
```

The training script writes checkpoints and logs under `results/<timestamp>/` and, when
requested, generates a CSV action log capturing every action issued by each boat.
