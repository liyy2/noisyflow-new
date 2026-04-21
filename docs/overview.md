# NoisyFlow Overview

## Pipeline summary
NoisyFlow is a three-stage pipeline for privacy-preserving federated synthetic data generation. Each client trains a flow-matching generator, learns a target-domain transport map, and then a server synthesizes labeled samples for downstream classification.

## Stage 1: Client flow matching
- Model: `VelocityField` in `noisyflow/stage1/networks.py`.
- Loss: flow matching over random time and Gaussian noise in `noisyflow/stage1/training.py`.
- DP-SGD is implemented with Opacus when `stage1.dp.enabled: true`.
- Label priors can be estimated with noisy counts through `stage1.label_prior`.
- Output per client: trained flow model and label prior, when configured.

## Stage 2: Client transport map
- Model options: ICNN-based optimal transport (`noisyflow/stage2/networks.py`), CellOT ICNN pairs, and flow-matching transport variants.
- Options in `stage2.option`:
  - A: real client data to target reference data.
  - B: synthetic data only (post-processing of stage 1).
  - C: mixed real + synthetic (concatenated batches).
- ICNN and flow-matching transport training live in `noisyflow/stage2/training.py`.
- CellOT training is enabled by `stage2.cellot.enabled: true` and supports option A.
- Stage 2 DP (from `stage2.dp`) requires CellOT with option A in the CLI entrypoint.

## Stage 3: Server synthesis and classifier
- Server samples labels from each client prior (or uniform), draws flow samples, then transports them with the OT map.
- Synthesis and classifier training live in `noisyflow/stage3/training.py`.
- Final classifier metrics are reported in `run.py`.

## Privacy curve and attacks
- `privacy_curve` runs multiple experiments across noise multipliers and writes a plot if Matplotlib is installed.
- Membership inference attacks live in `noisyflow/attacks/membership_inference.py`.
- See `docs/attacks.md` for the attack configuration and outputs.
