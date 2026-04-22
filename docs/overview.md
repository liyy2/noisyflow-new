# NoisyFlow Overview

## Pipeline summary
NoisyFlow is a three-stage pipeline for privacy-preserving federated domain adaptation. Source clients hold private labeled data, while the target domain provides a limited labeled reference split and a held-out test split. NoisyFlow synthesizes labeled target-like samples for target-domain classifier training without centralizing private source data.

## Stage 1: Per-client conditional generation
- Model: `VelocityField` in `noisyflow/stage1/networks.py`.
- Loss: label-conditional flow matching over random time and Gaussian noise in `noisyflow/stage1/training.py`.
- DP-SGD is implemented with Opacus when `stage1.dp.enabled: true`.
- Label priors can be estimated with noisy counts through `stage1.label_prior`.
- Output per client: trained flow model and label prior, when configured.

## Stage 2: Transport to target reference
- Model options: ICNN-based optimal transport (`noisyflow/stage2/networks.py`), CellOT ICNN pairs, and flow-matching transport variants.
- Options in `stage2.option`:
  - A: real client data to target reference data.
  - B: synthetic data only (post-processing of stage 1).
  - C: mixed real + synthetic (concatenated batches).
- ICNN and flow-matching transport training live in `noisyflow/stage2/training.py`.
- CellOT training is enabled by `stage2.cellot.enabled: true` and supports option A.
- Stage 2 DP (from `stage2.dp`) requires CellOT with option A in the CLI entrypoint.

## Stage 3: Synthesis and target evaluation
- The server samples labels from each client prior (or uniformly), draws flow samples, and transports them with the learned Stage 2 map.
- Classifiers are trained under the paper's `Synth-only`, `Ref-only`, and `Ref+Synth` reporting protocols.
- Synthesis and classifier training live in `noisyflow/stage3/training.py`; final target-test metrics are reported in `run.py`.

## Privacy curve and attacks
- `privacy_curve` runs multiple experiments across DP-SGD noise multipliers and reports privacy-utility curves.
- Membership inference attacks live in `noisyflow/attacks/membership_inference.py`.
- See `docs/attacks.md` for the attack configuration and outputs.
