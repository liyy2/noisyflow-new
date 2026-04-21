# Data Builders

All data builders return:
- `client_datasets`: list of `TensorDataset(x, label)` for each client.
- `target_ref`: `TensorDataset(y, label)` with public labeled target reference data.
- `target_test`: `TensorDataset(y, label)` for evaluation.

`run.py` supports the following `data.type` values:
- `federated_mixture_gaussians`
- `mixture_gaussians` (alias of `federated_mixture_gaussians`)
- `toy_federated_gaussians`
- `federated_cell_dataset` (generic `.h5ad` / `.npz` single-cell loader)
- `pamap2` / `federated_pamap2` (PAMAP2 Protocol wearable time-series windows)
- `brainscope` (BrainSCOPE-style cohort expression dataset)
- `cellot_lupuspatients_kang_hvg` (CellOT Kang lupuspatients convenience wrapper)
- `cellot_statefate_invitro_hvg` (CellOT statefate invitro convenience wrapper)
- `cellot_sciplex3_hvg` (CellOT sciplex3 convenience wrapper)
- `camelyon17` (CAMELYON17 embeddings loader with explicit source/target hospital sets)
- `camelyon17_wilds` (CAMELYON17-WILDS embeddings loader)

## `make_federated_mixture_gaussians`
Defined in `noisyflow/data/synthetic.py`.

Parameters:
- `K`: Number of clients.
- `n_per_client`: Samples per client.
- `n_target_ref`: Target reference samples.
- `n_target_test`: Target test samples.
- `d`: Feature dimension.
- `num_classes`: Number of classes.
- `component_scale`: Scale of class means.
- `component_cov`: Std of each Gaussian component.
- `class_probs`: Optional list of class probabilities.
- `scale_logstd`, `shift_scale`: Control random affine transforms.
- `seed`: RNG seed.

Notes:
- Each client applies a random affine transformation to the base mixture.
- The target domain uses a different affine transformation.

## `make_toy_federated_gaussians`
Defined in `noisyflow/data/toy.py`.

Parameters:
- `K`, `n_per_client`, `n_target_ref`, `n_target_test`, `d`, `num_classes`, `seed`.

Notes:
- Similar to `make_federated_mixture_gaussians` but with a simpler setup and fewer knobs.

## Example config
```yaml
data:
  type: federated_mixture_gaussians
  params:
    K: 3
    n_per_client: 1500
    n_target_ref: 2000
    n_target_test: 1000
    d: 2
    num_classes: 3
    component_scale: 3.0
    component_cov: 0.5
    seed: 0
```

## `make_federated_cell_dataset`
Defined in `noisyflow/data/cell.py`.

This loader supports CellOT-style data with a source and target condition (e.g., `ctrl`→`stim`)
and a client partition key (e.g., donor/patient id).

Input formats:
- `.h5ad`: reads `adata.X` and uses `adata.obs[label_key]`, `adata.obs[client_key]`, `adata.obs[condition_key]`
  (requires `anndata` + `h5py`)
- `.npz`: expects arrays: `X` (N,d), `label` (N,), `client` (N,), `condition` (N,)

Key parameters:
- `path`: dataset path
- `label_key`, `client_key`, `condition_key`: metadata keys (for `.h5ad`)
- `source_condition`, `target_condition`: condition values defining source/target
- `split_mode`: `ood` (use `holdout_client`) or `iid`
- `holdout_client`: client id reserved for target test split in `ood` mode
- `source_size_per_client`: optional per-client subsample size for source data (int or fraction)
- `target_ref_size`, `target_test_size`: optionally subsample target splits (int or fraction)
- `max_clients`, `min_cells_per_client`: control the federated client list
- `pca_dim`, `standardize`: optional global preprocessing (fit on source+target_ref)

## `make_federated_camelyon17_wilds`
Defined in `noisyflow/data/camelyon17.py`.

This loader expects an embedding table produced by `scripts/prepare_camelyon17_wilds.py`.
It treats each *hospital* as a federated client and uses one hospital as the target domain.

Input format:
- `.npz` with arrays: `X` (N,d), `label` (N,), `hospital` (N,), `split` (N,)

Key parameters:
- `path`: `.npz` path
- `source_splits`: WILDS split names/ids to use for source clients (default: `["train","id_val"]`)
- `target_split`: WILDS split name/id to use for target reference/test (default: `"test"`)
- `target_hospital`: hospital id for the target domain (default: `2` in CAMELYON17-WILDS)
- `target_hospitals`: optional list of hospital ids for a multi-hospital target domain (mutually exclusive with `target_hospital`)
- `n_per_client`: optional per-client subsample size (int or fraction)
- `target_ref_size`, `target_test_size`: sizes (int or fraction) for iid split within the target domain
- `pca_dim`, `standardize`: optional global preprocessing (fit on source+target_ref)

## `make_federated_camelyon17`
Defined in `noisyflow/data/camelyon17.py`.

This loader is similar to `make_federated_camelyon17_wilds`, but defines source/target domains
explicitly via hospital id lists (useful for custom "2 source hospitals / 3 target hospitals"
setups that don't align with WILDS's official train/val/test split).

Key parameters:
- `path`: `.npz` path with arrays: `X`, `label`, `hospital`, `split`
- `source_hospitals`: list of hospital ids used as federated clients (one client per hospital)
- `target_hospitals`: list of hospital ids used as the target domain (pooled)
- `source_splits`, `target_splits`: optional WILDS split names/ids to restrict each side (omit for all splits)
- `n_per_client`, `target_ref_size`, `target_test_size`: optional subsampling controls (int or fraction)
- `pca_dim`, `standardize`: optional global preprocessing (fit on source+target_ref)

## `make_federated_brainscope`
Defined in `noisyflow/data/brainscope.py`.

This loader expects a cohort-level expression table prepared by:
```bash
python scripts/prepare_brainscope_aging_yl.py
```

Input format:
- `.npz` with arrays: `X` (N,G), `cohort` (N,), `disorder` (N,), and label arrays.

Key parameters:
- `path`: `.npz` path
- `label_mode`: `case_control` or `neurodegenerative`
- `source_cohorts`: cohorts used as source clients (one client per cohort)
- `target_cohorts`: optional target cohort list (default: all non-source cohorts)
- `target_test_size`: target test fraction/size (default: 20% when omitted)
- `target_ref_size`: optional subsample size for target_ref
- `pca_dim`, `standardize`: optional global preprocessing (fit on source+target_ref)

## `make_federated_pamap2`
Defined in `noisyflow/data/pamap2.py`.

This loader expects a preprocessed PAMAP2 window table (Protocol subset) produced by:
```bash
python scripts/prepare_pamap2.py --output datasets/pamap2/pamap2_protocol_windows.npz
```

Input format:
- `.npz` with arrays: `X` (N,d), `label` (N,), `subject` (N,)

Domain adaptation setup:
- Each *source subject* is a federated client (one client per subject id).
- One held-out *target subject* is split iid into `target_ref` and `target_test`.

Key parameters:
- `path`: `.npz` path
- `target_subject`: subject id reserved as the target domain
- `source_subjects`: optional explicit source subject list (default: all non-target subjects)
- `n_per_client`, `target_ref_size`, `target_test_size`: optional subsampling controls (int or fraction)
- `pca_dim`, `standardize`: optional global preprocessing (fit on source+target_ref)
