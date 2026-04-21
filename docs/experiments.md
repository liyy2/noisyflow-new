# Experiments and CLI

## Common commands
Run the default experiment:
```bash
python run.py --config configs/default.yaml
```

Run the small smoke test:
```bash
python run.py --config configs/quick_smoke.yaml
```
Note: `configs/quick_smoke.yaml` defaults to `device: cuda`. Switch to `cpu` if you do not have a GPU.

Run the DP config on GPU:
```bash
python run.py --config configs/dp_gpu.yaml
```

Run the option A OT config:
```bash
python run.py --config configs/ot_option_a.yaml
```

Run the stage-level MIA demo:
```bash
python run.py --config configs/stage_mia_demo.yaml
```

Run the stage shadow MIA demo:
```bash
python run.py --config configs/stage_shadow_mia_demo.yaml
```

## BrainSCOPE-style cohort data (BrainSCOPE / aging_YL)
Prepare the cohort-level expression matrix from `liyy2/aging_YL` (downloads a processed `.bed.gz` and writes a compact `.npz`):
```bash
python scripts/prepare_brainscope_aging_yl.py
```

Run a quick end-to-end sanity check:
```bash
python run.py --config configs/brainscope_excitatory_smoke.yaml
```

Run the tuned setting (CMC as source, all other cohorts as target; case/control label; RectifiedFlow OT option C):
```bash
python run.py --config configs/brainscope_excitatory_ref50_optionC.yaml
```

Run the “best demo” label budget (large ref+synth gain over ref-only at fixed seed):
```bash
python run.py --config configs/brainscope_excitatory_demo_best.yaml
```

Run the Stage I generator ablation on the fixed Option B BrainSCOPE pipeline
(same Stage II/III settings, swap only `stage1.model` between flow and VAE):
```bash
python scripts/run_stage1_generator_ablation.py \
  --config configs/brainscope_excitatory_stage1_ablation_optionB_dp.yaml \
  --seeds 0,1,2 \
  --output-json plots/brainscope_stage1_ablation_optionB_dp.json
```

To sweep label budgets (ref-only vs synth-only vs ref+synth) and write a plot:
```bash
python scripts/sweep_ref_sweet_spot.py --config configs/brainscope_excitatory_ref50_optionC.yaml \
  --ref-sizes 5,10,20,30,50,75,100,150,all --syn-sizes 100,200,500,1000,2000,5000,all \
  --output-json plots/brainscope_ref_sweep_seed0.json --plot-output plots/brainscope_ref_sweep_seed0.pdf
```

## CellOT lupuspatients (Kang)
Install dataset I/O deps:
```bash
python -m pip install anndata h5py
```

Download the preprocessed dataset ZIP and extract the lupuspatients subset:
```bash
python scripts/fetch_cellot_datasets.py --dataset lupuspatients
```

Run an end-to-end NoisyFlow experiment (ctrl→stim transport; stimulated cell-type classification on OOD holdout patient `101`):
```bash
python run.py --config configs/cellot_lupus_kang_smoke.yaml
python run.py --config configs/cellot_lupus_kang_rectifiedflow_ref50.yaml
```

Tuned config (more clients, longer training, more synthesis) with an ``acceptable'' labeled target budget:
```bash
python run.py --config configs/cellot_lupus_kang_rectifiedflow_ref50_tuned.yaml
```

To check that transport improves donor alignment without collapsing cell-type structure, run the
structure-preservation analysis on the tuned Kang setup:
```bash
python scripts/evaluate_cell_structure_preservation.py \
  --config configs/cellot_lupus_kang_rectifiedflow_ref50_tuned.yaml \
  --seeds 0,1,2 \
  --output-json plots/kang_structure_preservation.json \
  --plot-output plots/kang_structure_preservation.pdf
```

Privacy-utility curve (plots `acc_ref_plus_synth` vs ε for a stage-1-only DP pipeline; OT is post-processing):
```bash
python run.py --config configs/cellot_lupus_kang_privacy_curve_ref50_stage1only.yaml
```

## CellOT statefate (invitro)
Install dataset I/O deps:
```bash
python -m pip install anndata h5py
```

Download the preprocessed dataset ZIP and extract the statefate subset:
```bash
python scripts/fetch_cellot_datasets.py --dataset statefate
```

Run a tuned, low-label target regime where `ref+synth` improves over `ref-only`:
```bash
python run.py --config configs/cellot_statefate_invitro_rectifiedflow_ref75_sweetspot.yaml
```

To sweep ref/synth label budgets on a fixed trained model:
```bash
python scripts/sweep_ref_sweet_spot.py --config configs/cellot_statefate_invitro_rectifiedflow_ref50.yaml
```

To also save a PDF plot + JSON table:
```bash
python scripts/sweep_ref_sweet_spot.py --config configs/cellot_statefate_invitro_rectifiedflow_ref50.yaml \
  --ref-sizes 25,50,75,100,200 --syn-sizes 500,1000,2000,4000,all \
  --plot-output plots/statefate_ref_sweep_seed0.pdf --output-json plots/statefate_ref_sweep_seed0.json \
  --plot-syn-size 500
```

## CAMELYON17-WILDS (Koh et al., 2021)
Install dataset deps:
```bash
python -m pip install wilds torchvision
```

Download the dataset and precompute embedding features (ResNet18):
```bash
python scripts/prepare_camelyon17_wilds.py --splits train,id_val,test --max-per-hospital 12000 --device cuda --amp
```

Run NoisyFlow with the standard `syn` / `ref` / `syn+ref` reporting:
```bash
python run.py --config configs/camelyon17_wilds_quick.yaml
python run.py --config configs/camelyon17_wilds_ref50.yaml
```

Run DP domain-adaptation baselines (Opacus DP-SGD on embeddings):
```bash
python scripts/run_camelyon17_baselines.py --config configs/camelyon17_wilds_dp_stage1only_ref50_pair_ot_nm2.yaml
```

Run the rebuttal federated classifier baseline on the same embedding split:
```bash
python scripts/run_camelyon17_fedavg_baseline.py --config configs/camelyon17_wilds_ref50.yaml
python scripts/run_camelyon17_fedavg_baseline.py --config configs/camelyon17_wilds_dp_stage1only_ref50_pair_ot_nm2.yaml --dp --noise-multiplier 2
```

## Toy demo
A compact demo exists in `noisyflow/demo.py` and is exposed by `noisyflow_sketch.py`:
```bash
python noisyflow_sketch.py
```

## Dependencies
Required:
- `torch`
- `numpy`
- `pyyaml` (required for config loading)

Optional:
- `opacus` (required for DP-SGD)
- `matplotlib` (required for privacy curve plots)

## Tests
```bash
python -m unittest discover -s tests
```
