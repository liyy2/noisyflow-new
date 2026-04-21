# NoisyFlow

NoisyFlow is a three-stage pipeline for federated synthetic data generation with optional differential privacy.
It trains a flow-matching generator per client, fits an optimal transport map to a target domain, and
then synthesizes target-like data for downstream classification tasks. The framework has two main goals: (1) enable domain adaptation by generating data that matches a target distribution, so that classifiers trained on the synthetic (domain-transferred) data perform better on the target domain than classifiers trained only on the original (source) domains; and (2) allow combining both original and transferred data to further improve classification accuracy on the desired target domain (3) all of these have privacy guarantee.

## Features
- Stage 1: flow matching generator with optional DP-SGD (Opacus).
- Stage 2: ICNN or CellOT transport (options A/B/C).
- Stage 3: server-side synthesis and classifier training.
- Optional privacy-utility sweeps and membership inference evaluations.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy
pip install pyyaml  # required for YAML configs
pip install opacus  # optional, DP-SGD in stage1/stage2
pip install matplotlib  # optional, privacy curve plots
```

## Quickstart 22
```bash
python run.py --config configs/default.yaml
```

For a smaller smoke test:
```bash
python run.py --config configs/quick_smoke.yaml
```
Note: `configs/quick_smoke.yaml` sets `device: cuda`; switch to `cpu` if you do not have a GPU.

To run the toy demo script:
```bash
python noisyflow_sketch.py
```

## Configuration
- Configs live under `configs/` and are loaded by `run.py`.
- `data.type` supports `federated_mixture_gaussians`, `mixture_gaussians`, `toy_federated_gaussians`, `federated_cell_dataset`, `pamap2` (PAMAP2 wearable time-series windows), `camelyon17`, `camelyon17_wilds`, `brainscope`, and the CellOT convenience wrappers `cellot_lupuspatients_kang_hvg`, `cellot_statefate_invitro_hvg`, `cellot_sciplex3_hvg`.
- Enabling DP (`stage1.dp` / `stage2.dp`) requires Opacus.
- `privacy_curve.enabled: true` runs a sweep and writes `privacy_utility.png` (requires matplotlib).

To fetch CellOT preprocessed datasets:
```bash
python scripts/fetch_cellot_datasets.py --dataset lupuspatients
python scripts/fetch_cellot_datasets.py --dataset statefate
python scripts/fetch_cellot_datasets.py --dataset sciplex3
```

To prepare the BrainSCOPE-style cohort dataset from the processed matrices in `liyy2/aging_YL`:
```bash
python scripts/prepare_brainscope_aging_yl.py
python run.py --config configs/brainscope_excitatory_smoke.yaml
python run.py --config configs/brainscope_excitatory_demo_best.yaml

# Optional: sweep label budgets (ref-only vs synth-only vs ref+synth)
python scripts/sweep_ref_sweet_spot.py --config configs/brainscope_excitatory_ref50_optionC.yaml \
  --ref-sizes 5,10,20,30,50,75,100,150,all --syn-sizes 100,200,500,1000,2000,5000,all \
  --output-json plots/brainscope_ref_sweep_seed0.json --plot-output plots/brainscope_ref_sweep_seed0.pdf
```

## Documentation
Start here: `docs/README.md`.
- `docs/overview.md`: Pipeline overview and stage summary.
- `docs/configuration.md`: Full config reference.
- `docs/data.md`: Synthetic data builders.
- `docs/experiments.md`: CLI usage and experiment configs.
- `docs/attacks.md`: Membership inference attack details.
- `docs/architecture.md`: Code map.

## Tests
```bash
python -m unittest
```
Tests that depend on optional packages (pyyaml, opacus, matplotlib) are skipped if those packages
are not installed.
