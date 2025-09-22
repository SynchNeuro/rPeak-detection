# ECG R-Peak Detection Workspace

This repository collects experiments for detecting R-peaks from OpenBCI EEG/ECG recordings. The codebase focuses on preprocessing raw text exports, training PyTorch-based models, and generating diagnostics for model quality and signal alignment.

## Quick Start
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run the debugging harness to verify data access and basic models:
   ```bash
   python scripts/debug.py
   ```
3. Launch training with the default configuration:
   ```bash
   python scripts/train.py --config configs/base_config.yaml
   ```

Training, evaluation, and visualization outputs are written to `outputs/`. Raw OpenBCI captures (`OpenBCI-RAW-*.txt`) remain in the repository root for reproducibility. Keep new recordings anonymised before committing.

## Project Layout
- `src/`: data loading, preprocessing utilities, model definitions, trainers, and plotting helpers
- `scripts/`: CLI entry points for debugging, training, and evaluation
- `configs/`: YAML configuration files (base defaults plus model-specific overrides)
- `outputs/`: generated models, plots, and logs (typically gitignored)

## Contributor Resources
For coding standards, testing expectations, and PR workflow, see the [Repository Guidelines](AGENTS.md).
