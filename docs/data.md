# Data Builders and Dataset Reproduction

NoisyFlow experiments load datasets through the `data` block in a YAML config. The CLI entrypoint, `run.py`, maps `data.type` values to builders in `noisyflow/data/`.

Every builder returns the same three objects:

| Return value | Meaning |
|---|---|
| `client_datasets` | List of `TensorDataset(x, label)` objects, one private source client per list element. |
| `target_ref` | `TensorDataset(x, label)` for labeled target-reference data available during adaptation and classifier training. |
| `target_test` | `TensorDataset(x, label)` for held-out target-domain evaluation. |

The repository does not track raw or generated datasets. The `datasets/`, `.cache/`, `results/`, and most generated `plots/` files are local artifacts. Reproduce a dataset by running the commands below from the repository root, then run the matching YAML config.

## Supported Dataset Types

| `data.type` | Builder | Required artifact | How to create it | Example config |
|---|---|---|---|---|
| `federated_mixture_gaussians` | `make_federated_mixture_gaussians` | None | Generated in memory from `seed` and YAML parameters. | `configs/default.yaml` |
| `mixture_gaussians` | Alias of `federated_mixture_gaussians` | None | Generated in memory. | `configs/default.yaml` |
| `toy_federated_gaussians` | `make_toy_federated_gaussians` | None | Generated in memory from `seed` and YAML parameters. | Custom/toy configs |
| `federated_cell_dataset` | `make_federated_cell_dataset` | `.h5ad` or `.npz` single-cell table | Supply a compatible table or use one of the CellOT wrappers below. | Custom configs |
| `cellot_lupuspatients_kang_hvg` | `make_cellot_lupuspatients_kang_hvg` | `datasets/scrna-lupuspatients/kang-hvg.h5ad` | `python scripts/fetch_cellot_datasets.py --dataset lupuspatients` | `configs/cellot_lupus_kang_smoke.yaml` |
| `cellot_statefate_invitro_hvg` | `make_cellot_statefate_invitro_hvg` | `datasets/scrna-statefate/invitro-hvg.h5ad` | `python scripts/fetch_cellot_datasets.py --dataset statefate` | `configs/cellot_statefate_invitro_smoke.yaml` |
| `cellot_sciplex3_hvg` | `make_cellot_sciplex3_hvg` | `datasets/scrna-sciplex3/hvg.h5ad` | `python scripts/fetch_cellot_datasets.py --dataset sciplex3` | `configs/cellot_sciplex3_trametinib_cellot_ref50.yaml` |
| `brainscope` | `make_federated_brainscope` | `datasets/brainscope/brainscope_excitatory_neur.npz` | `python scripts/prepare_brainscope_aging_yl.py` | `configs/brainscope_excitatory_smoke.yaml` |
| `federated_brainscope` | Alias of `brainscope` builder | Same as `brainscope` | Same as `brainscope`. | Custom configs |
| `camelyon17_wilds` | `make_federated_camelyon17_wilds` | `datasets/camelyon17_wilds/camelyon17_resnet18.npz` | `python scripts/prepare_camelyon17_wilds.py ...` | `configs/camelyon17_wilds_quick.yaml` |
| `camelyon17` | `make_federated_camelyon17` | CAMELYON17 embedding `.npz` | Use `scripts/prepare_camelyon17_wilds.py`, usually with all splits. | `configs/camelyon17_source2_target3_stage_mia_nodp.yaml` |
| `pamap2` | `make_federated_pamap2` | `datasets/pamap2/pamap2_protocol_windows.npz` or variant | `python scripts/prepare_pamap2.py ...` | `configs/pamap2_protocol_smoke.yaml` |
| `federated_pamap2` | Alias of `pamap2` builder | Same as `pamap2` | Same as `pamap2`. | Custom configs |
| `federated_4i_proteomics` | `make_federated_4i_proteomics` | `datasets/4i/8h.h5ad` | Supply a compatible 4i `.h5ad`; no downloader is included. | `configs/4i_proteomics_control_to_dasatinib_smoke.yaml` |

## Reproducibility Checklist

1. Create the raw or prepared dataset artifact listed in the table.
2. Keep the generated file under `datasets/<dataset-name>/`.
3. Use a YAML config whose `data.params.path` points to that file.
4. Set `data.params.seed` and the top-level `seed`.
5. Record the exact preparation command, package versions, and config.
6. For downloaded public datasets, record any upstream version, branch, split scheme, or model backbone used by the preparation script.

## Shared Splitting and Preprocessing Rules

Most real-data builders share these conventions:

- Source domains become private clients. Examples: patient ids, cohorts, hospitals, subjects, or pseudo-clients.
- Target-domain samples are split into `target_ref` and `target_test`.
- Integer sizes such as `target_ref_size: 50` keep exactly that many samples when available.
- Fractional sizes such as `target_test_size: 0.2` keep that fraction of the available split.
- Subsampling is randomized with `data.params.seed`.
- `standardize: true` fits a `StandardScaler` on source plus `target_ref` unless the builder documents a different `preprocess_fit` option.
- `pca_dim` fits PCA on the same fit set used for standardization and then applies it to all splits.
- Optional preprocessing requires `scikit-learn`.

## Synthetic Mixture Datasets

Defined in:

- `noisyflow/data/synthetic.py`
- `noisyflow/data/toy.py`

These datasets require no external files. They are generated in memory from the YAML parameters and `seed`.

### `federated_mixture_gaussians` and `mixture_gaussians`

The builder samples a class-conditional Gaussian mixture in a shared base space, applies a random affine transformation to each source client, and applies a separate affine transformation to the target domain.

Key parameters:

| Parameter | Meaning |
|---|---|
| `K` | Number of source clients. |
| `n_per_client` | Samples per source client. |
| `n_target_ref` | Labeled target-reference samples. |
| `n_target_test` | Held-out target-test samples. |
| `d` | Feature dimension. |
| `num_classes` | Number of class labels. |
| `component_scale` | Scale of class means. |
| `component_cov` | Gaussian component standard deviation. |
| `class_probs` | Optional class probabilities. |
| `scale_logstd`, `shift_scale` | Strength of random affine domain shifts. |
| `seed` | Random seed for reproducible generation. |

Minimal config:

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

Smoke test:

```bash
python run.py --config configs/quick_smoke.yaml
```

### `toy_federated_gaussians`

This builder has the same high-level source/target affine-shift structure as `federated_mixture_gaussians`, with fewer knobs. Use it for small debugging examples or tests.

Key parameters:

- `K`
- `n_per_client`
- `n_target_ref`
- `n_target_test`
- `d`
- `num_classes`
- `seed`

## Generic Single-Cell Table

Defined in `noisyflow/data/cell.py` as `make_federated_cell_dataset`.

Use `federated_cell_dataset` when you already have a single `.h5ad` or `.npz` table and want to define source and target domains by metadata columns.

Install `anndata` and `h5py` when loading `.h5ad` files:

```bash
pip install anndata h5py
```

Supported input formats:

| Format | Required contents |
|---|---|
| `.h5ad` | `adata.X` plus `adata.obs[label_key]`, `adata.obs[client_key]`, and `adata.obs[condition_key]`. |
| `.npz` | Arrays `X` with shape `(N, d)`, `label` with shape `(N,)`, `client` with shape `(N,)`, and `condition` with shape `(N,)`. |

Split rules:

- Rows with `condition == source_condition` form private source data.
- Rows with `condition == target_condition` form target data.
- With `split_mode: ood` and `holdout_client`, target-condition rows from the held-out client become `target_test`; other target-condition rows become `target_ref`.
- With `split_mode: iid`, target-condition rows are shuffled and split into `target_ref` and `target_test`.
- Each source client is one distinct value of `client_key` after filtering.

Key parameters:

| Parameter | Meaning |
|---|---|
| `path` | Input `.h5ad` or `.npz`. |
| `label_key`, `client_key`, `condition_key` | Metadata keys for `.h5ad` files. |
| `source_condition`, `target_condition` | Values defining source and target domains. |
| `split_mode` | `ood` or `iid`. |
| `holdout_client` | Client id reserved for target-test in OOD mode. |
| `source_size_per_client` | Optional per-client source subsample size. |
| `target_ref_size`, `target_test_size` | Optional target split sizes. |
| `max_clients`, `min_cells_per_client` | Client filtering controls. |
| `standardize`, `pca_dim` | Optional global preprocessing. |
| `seed` | Split and subsampling seed. |

Minimal `.npz` creation contract:

```python
import numpy as np

np.savez_compressed(
    "datasets/my_cells/my_table.npz",
    X=X.astype("float32"),
    label=labels,
    client=client_ids,
    condition=conditions,
)
```

Example config:

```yaml
data:
  type: federated_cell_dataset
  params:
    path: datasets/my_cells/my_table.npz
    source_condition: ctrl
    target_condition: stim
    split_mode: ood
    holdout_client: patient_101
    target_ref_size: 500
    target_test_size: 1000
    standardize: true
    pca_dim: 100
    seed: 0
```

## CellOT Single-Cell Benchmarks

Defined in `noisyflow/data/cell.py` as wrappers around `make_federated_cell_dataset`.

The preparation script downloads the CellOT preprocessed dataset ZIP from the URL hard-coded in `scripts/fetch_cellot_datasets.py` and extracts only the requested subfolder.

Install dataset I/O dependencies:

```bash
pip install anndata h5py
```

Fetch each benchmark:

```bash
python scripts/fetch_cellot_datasets.py --dataset lupuspatients
python scripts/fetch_cellot_datasets.py --dataset statefate
python scripts/fetch_cellot_datasets.py --dataset sciplex3
```

Expected outputs:

| Dataset | Output file | Wrapper defaults |
|---|---|---|
| Kang lupuspatients | `datasets/scrna-lupuspatients/kang-hvg.h5ad` | `label_key: cell_type`, `client_key: sample_id`, `condition_key: condition`, `source_condition: ctrl`, `target_condition: stim`, `split_mode: ood`, `holdout_client: 101` |
| Statefate invitro | `datasets/scrna-statefate/invitro-hvg.h5ad` | `label_key: annotation`, `client_key: library`, `condition_key: condition`, `source_condition: control`, `target_condition: developed`, `split_mode: iid` |
| SciPlex3 | `datasets/scrna-sciplex3/hvg.h5ad` | `label_key: cell_type`, `client_key: replicate`, `condition_key: drug-dose`, `source_condition: control-0`, `target_condition: trametinib-1000`, `split_mode: iid` |

Run smoke or representative experiments:

```bash
python run.py --config configs/cellot_lupus_kang_smoke.yaml
python run.py --config configs/cellot_statefate_invitro_smoke.yaml
python run.py --config configs/cellot_sciplex3_trametinib_cellot_ref50.yaml
```

Reproduction notes:

- The downloader caches `processed_datasets_all.zip` under `.cache/cellot_data/`.
- Delete the extracted dataset folder and rerun the command to recreate the local artifact.
- Use `target_ref_size`, `target_test_size`, `source_size_per_client`, and `seed` in YAML to reproduce the NoisyFlow split.

## BrainSCOPE-Style Cohort Expression Data

Defined in `noisyflow/data/brainscope.py` as `make_federated_brainscope`.

The preparation script downloads metadata and an excitatory-neuron expression matrix from the `liyy2/aging_YL` GitHub repository, aligns samples to metadata, encodes labels, and writes a compact `.npz`.

Install dependency:

```bash
pip install pandas
```

Create the prepared file:

```bash
python scripts/prepare_brainscope_aging_yl.py
```

Default output:

```text
datasets/brainscope/brainscope_excitatory_neur.npz
```

The `.npz` contains:

| Key | Meaning |
|---|---|
| `X` | Expression matrix with shape `(samples, genes)`. |
| `genes` | Gene names after duplicate-gene aggregation. |
| `individual_id` | Sample ids aligned to metadata. |
| `cohort` | Cohort id used for source/target domains. |
| `disorder` | Disorder label from metadata. |
| `label_case_control` | Binary label: non-control vs. control. |
| `label_neurodegenerative` | Binary label for neurodegenerative disorders used by the script. |
| `sex`, `age_death`, `ancestry` | Additional metadata copied from the source table. |
| `source_repo`, `source_branch`, `source_expr_path`, `source_meta_path` | Provenance fields for the download. |

Default source files:

| Script argument | Default |
|---|---|
| `--repo` | `liyy2/aging_YL` |
| `--branch` | `master` |
| `--metadata-path` | `PEC2_sample_metadata_processed.csv` |
| `--expr-path` | `expression_matrix_9celltypes_07072023/Excitatory_Neur.expr.bed.gz` |
| `--cache-dir` | `.cache/brainscope_aging_yl` |

Builder split rules:

- `source_cohorts` become private clients, one cohort per client.
- `target_cohorts`, if provided, become the target domain.
- If `target_cohorts` is omitted, all non-source cohorts become the target domain.
- Target data are split iid into `target_ref` and `target_test`.
- `label_mode: case_control` uses `label_case_control`.
- `label_mode: neurodegenerative` uses `label_neurodegenerative`.

Example config:

```yaml
data:
  type: brainscope
  params:
    path: datasets/brainscope/brainscope_excitatory_neur.npz
    label_mode: case_control
    source_cohorts: [CMC]
    target_ref_size: 50
    standardize: true
    pca_dim: 100
    seed: 0
```

Run:

```bash
python run.py --config configs/brainscope_excitatory_smoke.yaml
python run.py --config configs/brainscope_excitatory_demo_best.yaml
```

## CAMELYON17-WILDS Embeddings

Defined in `noisyflow/data/camelyon17.py` as:

- `make_federated_camelyon17_wilds`
- `make_federated_camelyon17`

The preparation script downloads CAMELYON17 through WILDS, embeds image patches with an ImageNet-pretrained torchvision ResNet, and writes a feature table.

Install dependencies:

```bash
pip install wilds torchvision
```

Create the default ResNet18 feature table:

```bash
python scripts/prepare_camelyon17_wilds.py \
  --splits train,id_val,val,test \
  --max-per-hospital 20000 \
  --device cuda \
  --amp
```

Default outputs:

```text
datasets/camelyon17_wilds/camelyon17_resnet18.npz
datasets/camelyon17_wilds/camelyon17_resnet18.meta.json
```

Use CPU by changing `--device cpu` and omitting `--amp`.

For configs that reference `camelyon17_resnet18_all_splits.npz`, recreate that artifact by choosing an explicit output path:

```bash
python scripts/prepare_camelyon17_wilds.py \
  --splits train,id_val,val,test \
  --max-per-hospital 20000 \
  --output datasets/camelyon17_wilds/camelyon17_resnet18_all_splits.npz \
  --device cuda \
  --amp
```

The `.npz` contains:

| Key | Meaning |
|---|---|
| `X` | ResNet embedding matrix with shape `(N, d)`. |
| `label` | CAMELYON17 binary labels. |
| `hospital` | Hospital id metadata; used as source/target domain id. |
| `split` | WILDS split id. |

The `.meta.json` records the WILDS split scheme, selected splits, backbone, feature dimension, maximum samples per hospital, and seed.

### `camelyon17_wilds`

This builder follows the WILDS split convention:

- Source clients are hospitals from `source_splits`, excluding the target hospital.
- Target data are rows from `target_split` and `target_hospital` or `target_hospitals`.
- Target data are split iid into `target_ref` and `target_test`.

Example:

```yaml
data:
  type: camelyon17_wilds
  params:
    path: datasets/camelyon17_wilds/camelyon17_resnet18.npz
    source_splits: [train, id_val]
    target_split: test
    target_hospital: 2
    n_per_client: 20000
    target_ref_size: 50
    target_test_size: 10000
    standardize: true
    seed: 0
```

Run:

```bash
python run.py --config configs/camelyon17_wilds_quick.yaml
```

### `camelyon17`

This builder uses explicit hospital lists instead of the default WILDS source/target convention.

Example:

```yaml
data:
  type: camelyon17
  params:
    path: datasets/camelyon17_wilds/camelyon17_resnet18_all_splits.npz
    source_hospitals: [0, 3]
    target_hospitals: [1, 2, 4]
    source_splits: [train, id_val]
    target_splits: [test]
    target_ref_size: 2000
    target_test_size: 10000
    standardize: true
    seed: 0
```

Run:

```bash
python run.py --config configs/camelyon17_source2_target3_stage_mia_nodp.yaml
```

## PAMAP2 Wearable Activity Windows

Defined in `noisyflow/data/pamap2.py` as `make_federated_pamap2`.

The preparation script downloads the PAMAP2 Protocol dataset from the UCI Machine Learning Repository, extracts subject `.dat` files, converts time-series segments into window-level feature vectors, and writes a compact `.npz`.

Create the default six-activity table:

```bash
python scripts/prepare_pamap2.py \
  --output datasets/pamap2/pamap2_protocol_windows.npz
```

Create the 11-activity and 12-activity variants used by configs:

```bash
python scripts/prepare_pamap2.py \
  --activities 1,2,3,4,5,6,7,12,13,16,17 \
  --output datasets/pamap2/pamap2_protocol_windows_11act.npz

python scripts/prepare_pamap2.py \
  --activities 1,2,3,4,5,6,7,12,13,16,17,24 \
  --output datasets/pamap2/pamap2_protocol_windows_12act.npz
```

Default raw locations:

| Path | Meaning |
|---|---|
| `datasets/pamap2/raw/PAMAP2_Dataset.zip` | Downloaded UCI archive. |
| `datasets/pamap2/raw/PAMAP2_Dataset/Protocol/` | Extracted subject files. |

The prepared `.npz` contains:

| Key | Meaning |
|---|---|
| `X` | Window feature matrix. |
| `label` | Activity label encoded as `0..C-1`. |
| `subject` | Subject id; each source subject becomes one client. |
| `meta` | JSON string with preparation parameters. |

Important preparation options:

| Option | Meaning |
|---|---|
| `--subjects` | Comma-separated subject ids. Default is `101,102,103,104,105,106,107,108,109`. |
| `--activities` | Comma-separated PAMAP2 activity ids to keep. |
| `--include-gyro` | Include gyroscope channels. |
| `--no-heart-rate` | Drop heart-rate channel. |
| `--downsample` | Keep every Nth row. Default is `4`. |
| `--window-size`, `--stride` | Sliding-window length and stride. |
| `--label-purity` | Minimum majority-label fraction per window. |
| `--representation` | `stats` or `flat`; default is `stats`. |

Builder split rules:

- One held-out `target_subject` forms the target domain.
- All other subjects, or explicit `source_subjects`, become private clients.
- Target-subject windows are split iid into `target_ref` and `target_test`.
- `standardize: true` is the default for this builder.
- `preprocess_fit` can be `source_target_ref` or `source_only`.

Example config:

```yaml
data:
  type: pamap2
  params:
    path: datasets/pamap2/pamap2_protocol_windows.npz
    target_subject: 104
    target_ref_size: 20
    target_test_size: 0.5
    standardize: true
    seed: 0
```

Run:

```bash
python run.py --config configs/pamap2_protocol_smoke.yaml
python run.py --config configs/pamap2_protocol_best.yaml
```

## 4i Single-Cell Proteomics

Defined in `noisyflow/data/proteomics.py` as `make_federated_4i_proteomics`.

This builder expects a compatible `.h5ad` at:

```text
datasets/4i/8h.h5ad
```

No downloader is included for this dataset. To reproduce the NoisyFlow artifact, obtain or construct an `.h5ad` with:

```bash
pip install anndata h5py scikit-learn
```

| Required field | Meaning |
|---|---|
| `adata.X` | Cell-level protein/morphology feature matrix with shape `(N, d)`. |
| `adata.obs["drug"]` | Drug or condition label per cell. |

The local helper files, when present, are:

| File | Meaning |
|---|---|
| `datasets/4i/drugs.txt` | Drug names observed in the prepared 4i table. |
| `datasets/4i/features.txt` | Feature names used in the prepared 4i table. |

Split rules:

- Rows with `drug == source_drug` are private source cells.
- Source cells are randomly partitioned into `n_source_clients` pseudo-clients.
- Rows with `drug == target_drug` are split iid into `target_ref` and `target_test`.
- `label_mode: kmeans` fits `MiniBatchKMeans` on source plus `target_ref` after optional preprocessing and uses cluster ids as labels.
- `label_mode: none` assigns a single label to all cells and is appropriate only for distributional metrics.

Recreate a compatible `.h5ad` from your own 4i table:

```python
import anndata as ad
import numpy as np
import pandas as pd

adata = ad.AnnData(
    X=np.asarray(X, dtype="float32"),
    obs=pd.DataFrame({"drug": drug_labels.astype(str)}),
)
adata.write_h5ad("datasets/4i/8h.h5ad")
```

Example config:

```yaml
data:
  type: federated_4i_proteomics
  params:
    path: datasets/4i/8h.h5ad
    source_drug: control
    target_drug: dasatinib
    n_source_clients: 5
    source_size_per_client: 2000
    target_ref_size: 1000
    target_test_size: 1000
    standardize: true
    pca_dim: 32
    label_mode: kmeans
    num_labels: 12
    seed: 0
```

Run:

```bash
python run.py --config configs/4i_proteomics_control_to_dasatinib_smoke.yaml
```

## Dataset Sanity Checks

Use these small checks after preparing a dataset.

Check whether expected files exist:

```bash
test -f datasets/scrna-lupuspatients/kang-hvg.h5ad
test -f datasets/brainscope/brainscope_excitatory_neur.npz
test -f datasets/camelyon17_wilds/camelyon17_resnet18.npz
test -f datasets/pamap2/pamap2_protocol_windows.npz
test -f datasets/4i/8h.h5ad
```

Inspect a prepared `.npz`:

```bash
python - <<'PY'
import numpy as np

path = "datasets/pamap2/pamap2_protocol_windows.npz"
data = np.load(path, allow_pickle=True)
print(path)
for key in data.files:
    value = data[key]
    print(key, value.shape, value.dtype)
PY
```

Run the fastest matching smoke config before launching a sweep:

```bash
python run.py --config configs/quick_smoke.yaml
python run.py --config configs/cellot_lupus_kang_smoke.yaml
python run.py --config configs/brainscope_excitatory_smoke.yaml
python run.py --config configs/camelyon17_wilds_quick.yaml
python run.py --config configs/pamap2_protocol_smoke.yaml
python run.py --config configs/4i_proteomics_control_to_dasatinib_smoke.yaml
```
