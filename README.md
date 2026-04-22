# NoisyFlow: Privacy-Preserving Federated Domain Adaptation with Synthetic Data

[Main schematic](assets/Noisyflow-Mar24th-schematics-updated.pdf) • [Additional schematic](assets/schematics.pdf) • [Documentation](docs/README.md) • [Citation](#citation)

[![ISMB 2026 / Bioinformatics](https://img.shields.io/badge/ISMB%202026-Bioinformatics-1f6feb)](#citation)
[![Python](https://img.shields.io/badge/Python-3.10%2B-informational)](#requirements)
[![PyTorch](https://img.shields.io/badge/PyTorch-required-red)](#requirements)
[![Differential Privacy](https://img.shields.io/badge/Differential%20Privacy-Opacus-6f42c1)](#configuration-guide)

NoisyFlow studies a federated domain adaptation setting in which source clients, such as patients, subjects, cohorts, or hospitals, hold private labeled source-domain data. The target domain provides a labeled target-reference split and a held-out target-test split. The goal is to train accurate target-domain classifiers without centralizing the private source data.

The method follows the paper's three-stage protocol. Each source client trains a label-conditional flow-matching generator on its private samples. Each client then learns a transport map from its source distribution to the target-reference distribution using ICNN/CellOT or flow-matching transport. The server samples from the client generators, applies the learned transports, and evaluates classifiers trained on transported synthetic samples (`Synth-only`), labeled target-reference data (`Ref-only`), or their union (`Ref+Synth`). Differential privacy is implemented with Opacus DP-SGD and reported through privacy-utility tradeoffs.

This repository accompanies the ISMB 2026 / *Bioinformatics* version of NoisyFlow.

## Method Overview

<p align="center">
  <a href="assets/Noisyflow-Mar24th-schematics-updated.pdf">
    <img src="assets/noisyflow-main-schematic-600dpi.png" width="900" alt="NoisyFlow pipeline schematic for private federated generation, transport, synthesis, and downstream evaluation." />
  </a>
</p>
<p align="center">
  <em>Figure 1. NoisyFlow protocol for federated domain adaptation. Source clients train label-conditional generators on private data, learn transports into the target-reference distribution, and send only the components needed for server-side synthesis. The server trains target-domain classifiers with transported synthetic labels, target-reference labels, or their union.</em>
</p>

<p align="center">
  <a href="assets/schematics.pdf">
    <img src="assets/noisyflow-schematic-600dpi.png" width="760" alt="Additional NoisyFlow schematic for the federated synthetic data generation workflow." />
  </a>
</p>
<p align="center">
  <em>Figure 2. Expanded experimental workflow. The paper reports Ref-only, Synth-only, and Ref+Synth target-test performance, distributional alignment between transported synthetic samples and target data, and privacy-utility tradeoffs under DP-SGD.</em>
</p>

## Repository Contents

| Path | Purpose |
|---|---|
| `noisyflow/` | Core package for configuration, metrics, utilities, data builders, attacks, and stage implementations. |
| `noisyflow/stage1/` | Client-side flow-matching generators with DP-SGD support. |
| `noisyflow/stage2/` | Target-reference transport modules, including ICNN/CellOT and flow-matching transport variants. |
| `noisyflow/stage3/` | Server-side synthesis and downstream classifier training. |
| `noisyflow/baselines/` | Baselines for domain adaptation, federated predictor training, and noise-then-transport comparisons. |
| `configs/publication/` | YAML files for publication experiments. |
| `scripts/` | Data preparation, experiment, plotting, sweep, and benchmarking utilities. |
| `tests/` | Unit tests for configuration, data, metrics, DP, training, and baselines. |
| `docs/` | Detailed documentation for architecture, configuration, data, experiments, and attacks. |
| `assets/` | Schematics and rendered README figures. |
| `run.py` | Main experiment entrypoint. |

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Pipeline](#pipeline)
- [Configuration guide](#configuration-guide)
- [Reproducible workflows](#reproducible-workflows)
- [Documentation](#documentation)
- [Tests](#tests)
- [Citation](#citation)

## Requirements

| Component | Version / Expectation |
|---|---|
| OS | Linux recommended; CUDA GPU recommended for nontrivial experiments. |
| Python | 3.10+ recommended. |
| PyTorch | Required for all training and inference paths. |
| PyYAML | Required for YAML configuration files. |
| Opacus | Required dependency; used for DP-SGD in Stage 1 and Stage 2 experiments. |
| Matplotlib | Optional; required for privacy-utility plots. |
| scikit-learn | Optional; required for PCA, standardization, and RandomForest baselines. |

## Installation

Create an environment and install the core dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dataset-specific extras are documented as commented entries in `requirements.txt`. CAMELYON17-WILDS preparation uses `wilds` and `torchvision`; the BrainSCOPE preparation script uses `pandas`.

## Pipeline

NoisyFlow follows the three-stage protocol used in the paper.

1. **Stage 1: per-client conditional generation.** Each source client trains a label-conditional flow-matching generator on private source-domain samples. Training supports both non-private optimization and DP-SGD through Opacus.
2. **Stage 2: transport to target reference.** Each client learns a map from its source distribution to the target-reference distribution. The implementation supports ICNN-based optimal transport, CellOT-style training, and flow-matching transport variants.
3. **Stage 3: synthesis and target evaluation.** The server samples from the client generators, pushes samples through the learned transports, and trains target-domain classifiers under the paper's `Synth-only`, `Ref-only`, and `Ref+Synth` protocols.

Experiments evaluate target-test accuracy, macro-F1 when applicable, distributional alignment metrics such as SW2 or MMD, and privacy-utility curves. Privacy budgets are reported per client and summarized by the maximum epsilon at the configured delta.

## Configuration Guide

Experiments are specified by YAML files and executed by `run.py`.

| Config block | Purpose |
|---|---|
| `seed`, `device` | Reproducibility and CPU/GPU selection. |
| `data` | Dataset type, preprocessing, source-client split, target-reference split, and target-test split. |
| `stage1` | Flow-matching generator architecture, optimization, and DP-SGD settings. |
| `stage2` | Transport option, ICNN/CellOT/flow-matching settings, target pairing, and DP settings. |
| `stage3` | Number of synthetic samples per client and downstream classifier configuration. |
| `privacy_curve` | Privacy-utility sweep settings. |

Supported `data.type` values include:

- synthetic builders: `federated_mixture_gaussians`, `mixture_gaussians`, `toy_federated_gaussians`;
- biological and wearable datasets: `federated_cell_dataset`, `brainscope`, `camelyon17`, `camelyon17_wilds`, `pamap2`;
- CellOT convenience wrappers: `cellot_lupuspatients_kang_hvg`, `cellot_statefate_invitro_hvg`, `cellot_sciplex3_hvg`.

Publication configurations live in `configs/publication/`.

## Reproducible Workflows

Fetch the preprocessed CellOT single-cell benchmarks:

```bash
python scripts/fetch_cellot_datasets.py --dataset lupuspatients
python scripts/fetch_cellot_datasets.py --dataset statefate
python scripts/fetch_cellot_datasets.py --dataset sciplex3
```

Prepare the BrainSCOPE-style multi-cohort pseudobulk dataset from processed matrices in `liyy2/aging_YL`:

```bash
python scripts/prepare_brainscope_aging_yl.py
python run.py --config configs/brainscope_excitatory_smoke.yaml
python run.py --config configs/brainscope_excitatory_demo_best.yaml
```

Sweep labeled target-reference and transported-synthetic budgets:

```bash
python scripts/sweep_ref_sweet_spot.py --config configs/brainscope_excitatory_ref50_optionC.yaml \
  --ref-sizes 5,10,20,30,50,75,100,150,all \
  --syn-sizes 100,200,500,1000,2000,5000,all \
  --output-json plots/brainscope_ref_sweep_seed0.json \
  --plot-output plots/brainscope_ref_sweep_seed0.pdf
```

## Documentation

Start with:

| Resource | Purpose |
|---|---|
| `docs/README.md` | Documentation index. |
| `docs/overview.md` | Pipeline overview and stage summary. |
| `docs/configuration.md` | Full configuration reference. |
| `docs/data.md` | Data builders and dataset preparation. |
| `docs/experiments.md` | CLI usage and experiment recipes. |
| `docs/attacks.md` | Membership inference attack details. |
| `docs/architecture.md` | Code map and module relationships. |

## Tests

Run the unit test suite:

```bash
python -m unittest discover -s tests
```

Tests that require plotting, dataset-specific, or baseline-only dependencies are skipped when those packages are not installed.

## Citation

If you use NoisyFlow in your work, please cite the ISMB 2026 / *Bioinformatics* paper. The final BibTeX entry should be added here once proceedings metadata is available.
