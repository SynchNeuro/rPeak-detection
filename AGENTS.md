# Repository Guidelines

## Project Structure & Module Organization
- `src/` hosts the R-peak pipeline; key packages: `data/` (loading OpenBCI exports), `preprocessing/` (filters, splits), `models/` (TimesNet variants, U-Net), `training/` (trainer, metrics), `visualization/` (plots), and `utils/` (configuration, helpers).
- `scripts/` provides CLI entry points (`train.py`, `evaluate.py`, `debug.py`) that add `src/` to the path.
- `configs/` stores YAML configurations; start from `configs/base_config.yaml`. Model-specific overrides live in `configs/model_configs/`.
- `Time-Series-Library/` bundles third-party building blocks used by advanced models—treat it as vendored code and avoid editing unless syncing upstream.
- Raw recordings (`OpenBCI-RAW-*.txt`) live in the repo root and feed into training; keep new captures in the same format under version control only when anonymised.
- `outputs/` is the default artifact location (checkpoints, plots, logs); keep it out of commits unless sharing sample metrics.

## Build, Test, and Development Commands
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/debug.py            # Sanity-check data loaders, models, preprocessing
python scripts/train.py --config configs/base_config.yaml --epochs 30 --device auto
python scripts/evaluate.py --model-path outputs/models/best.pt --create-plots
```
Use `--create-default-configs` with `train.py` to scaffold extra configs, and pass `--dry-run` to validate data without training.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indents; keep modules under 500 lines by splitting helpers into `src/utils/`.
- Prefer descriptive snake_case names for functions and variables; class names stay in CapWords.
- Add type hints and docstrings mirroring existing trainer utilities; document tensor shapes in comments when not obvious.
- Keep YAML keys kebab-free and lower_snake_case for consistency with `RPeakConfig`.

## Testing & Evaluation
- Run `python scripts/debug.py` before training to confirm environment health; investigate failures before proceeding.
- After training, validate with `evaluate.py` and capture F1/precision/recall; attach generated plots from `outputs/plots/`.
- Target similar coverage across ECG and EEG channels; note any data subsets skipped in the PR.

## Commit & Pull Request Guidelines
- Use concise, imperative commit subjects (e.g., `Tune preprocessing window size`); keep body lines wrapped at 72 chars when context is needed.
- Squash noisy intermediate commits before opening a PR.
- PRs should: describe the change, list affected configs/data, link tracking issues, and include before/after metrics or plots when model quality shifts.

## Data & Configuration Notes
- Update `configs/base_config.yaml` only when defaults should change for everyone; otherwise add a new file under `configs/model_configs/`.
- Store large artifacts in `outputs/` and share via cloud storage rather than Git.
