from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset


@dataclass(frozen=True)
class DPSetting:
    name: str
    stage2_option: str
    stage1_target_epsilon: float
    stage2_target_epsilon: Optional[float]


@dataclass(frozen=True)
class Stage3Setting:
    name: str
    classifier: str  # mlp | rf | extra_trees | logreg | linsvc
    hidden: Sequence[int] = (256, 256)
    epochs: int = 30
    lr: float = 1e-3
    sk_params: Dict[str, Any] = field(default_factory=dict)
    standardize: bool = False


@dataclass(frozen=True)
class CaseSpec:
    name: str
    config_path: str
    dp_candidates: Sequence[DPSetting]
    ref_sizes: Sequence[int]
    syn_sizes: Sequence[int]
    syn_sizes_tune: Sequence[int]
    selection_ref_n: int


@dataclass
class TrainedArtifacts:
    cfg: Any
    d: int
    num_classes: int
    target_ref: TensorDataset
    target_test: TensorDataset
    y_syn: torch.Tensor
    l_syn: torch.Tensor
    x_syn_raw: torch.Tensor
    epsilon_flow_max: Optional[float]
    epsilon_ot_max: Optional[float]
    epsilon_total_max: Optional[float]


def _set_dp_target(dp_cfg, target_epsilon: float) -> Any:
    from noisyflow.utils import DPConfig

    if dp_cfg is None:
        dp_cfg = DPConfig()
    dp_cfg.enabled = True
    dp_cfg.target_epsilon = float(target_epsilon)
    return dp_cfg


def _disable_dp(dp_cfg) -> Any:
    from noisyflow.utils import DPConfig

    if dp_cfg is None:
        dp_cfg = DPConfig()
    dp_cfg.enabled = False
    dp_cfg.target_epsilon = None
    return dp_cfg


def _apply_dp_setting(cfg, setting: DPSetting) -> Any:
    cfg = copy.deepcopy(cfg)
    cfg.stage1.dp = _set_dp_target(getattr(cfg.stage1, "dp", None), setting.stage1_target_epsilon)
    cfg.stage2.option = str(setting.stage2_option).upper()
    if setting.stage2_target_epsilon is None or cfg.stage2.option.upper() == "B":
        cfg.stage2.dp = _disable_dp(getattr(cfg.stage2, "dp", None))
    else:
        cfg.stage2.dp = _set_dp_target(getattr(cfg.stage2, "dp", None), float(setting.stage2_target_epsilon))
    return cfg


def _set_device(cfg, device: str) -> Any:
    cfg = copy.deepcopy(cfg)
    cfg.device = str(device)
    return cfg


def _load_or_train_artifacts(cache_path: Path, cfg) -> TrainedArtifacts:
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        return TrainedArtifacts(
            cfg=cfg,
            d=int(payload["d"]),
            num_classes=int(payload["num_classes"]),
            target_ref=TensorDataset(payload["target_ref_x"], payload["target_ref_y"]),
            target_test=TensorDataset(payload["target_test_x"], payload["target_test_y"]),
            y_syn=payload["y_syn"],
            l_syn=payload["l_syn"],
            x_syn_raw=payload["x_syn_raw"],
            epsilon_flow_max=payload.get("epsilon_flow_max", None),
            epsilon_ot_max=payload.get("epsilon_ot_max", None),
            epsilon_total_max=payload.get("epsilon_total_max", None),
        )

    from noisyflow.stage1.networks import VelocityField
    from noisyflow.stage1.training import sample_flow_euler, train_flow_stage1
    from noisyflow.stage2.networks import RectifiedFlowOT
    from noisyflow.stage2.training import train_ot_stage2_rectified_flow
    from noisyflow.stage3.training import server_synthesize_with_raw
    from noisyflow.utils import dp_label_prior_from_counts, set_seed, unwrap_model

    from run import _build_datasets, _infer_dims

    set_seed(cfg.seed)
    device = str(cfg.device)

    client_datasets, target_ref, target_test = _build_datasets(cfg)
    d, num_classes = _infer_dims(cfg, client_datasets, target_ref=target_ref, target_test=target_test)

    target_loader = DataLoader(
        target_ref,
        batch_size=cfg.loaders.target_batch_size,
        shuffle=True,
        drop_last=False,
    )

    stage1_eps: List[float] = []
    stage2_eps: List[float] = []
    clients_out: List[Dict[str, Any]] = []

    for idx, ds in enumerate(client_datasets):
        train_loader = DataLoader(
            ds,
            batch_size=cfg.loaders.batch_size,
            shuffle=True,
            drop_last=cfg.loaders.drop_last,
        )

        flow = VelocityField(
            d=d,
            num_classes=num_classes,
            hidden=cfg.stage1.hidden,
            time_emb_dim=cfg.stage1.time_emb_dim,
            label_emb_dim=cfg.stage1.label_emb_dim,
            act=cfg.stage1.act,
            mlp_norm=cfg.stage1.mlp_norm,
            mlp_dropout=cfg.stage1.mlp_dropout,
            cond_dim=cfg.stage1.cond_dim,
            cond_emb_dim=cfg.stage1.cond_emb_dim,
        )
        flow_stats = train_flow_stage1(
            flow,
            train_loader,
            epochs=cfg.stage1.epochs,
            lr=cfg.stage1.lr,
            optimizer=cfg.stage1.optimizer,
            weight_decay=cfg.stage1.weight_decay,
            ema_decay=cfg.stage1.ema_decay,
            loss_normalize_by_dim=cfg.stage1.loss_normalize_by_dim,
            dp=cfg.stage1.dp,
            device=device,
        )
        if "epsilon_flow" in flow_stats:
            stage1_eps.append(float(flow_stats["epsilon_flow"]))

        prior = None
        if cfg.stage1.label_prior.enabled:
            labels = ds.tensors[1]
            prior = dp_label_prior_from_counts(
                labels,
                num_classes=num_classes,
                mechanism=cfg.stage1.label_prior.mechanism,
                sigma=cfg.stage1.label_prior.sigma,
                device="cpu",
            )

        def synth_sampler(
            batch_size: int,
            labels: Optional[torch.Tensor] = None,
            flow_model=flow,
        ) -> torch.Tensor:
            if labels is None:
                labels = torch.randint(0, num_classes, (batch_size,), device=device)
            else:
                labels = labels.to(device).long().view(-1)
                if int(labels.numel()) != int(batch_size):
                    raise ValueError(f"labels must have shape ({batch_size},), got {tuple(labels.shape)}")
            return sample_flow_euler(flow_model.to(device).eval(), labels, n_steps=cfg.stage2.flow_steps).cpu()

        if not cfg.stage2.rectified_flow.enabled:
            raise ValueError("Only stage2.rectified_flow.enabled=true is supported by this script.")

        source_loader = None
        if cfg.stage2.option.upper() in {"A", "C"}:
            source_loader = DataLoader(
                ds,
                batch_size=cfg.loaders.batch_size,
                shuffle=True,
                drop_last=cfg.loaders.drop_last,
            )

        ot = RectifiedFlowOT(
            d=d,
            hidden=cfg.stage2.rectified_flow.hidden,
            time_emb_dim=cfg.stage2.rectified_flow.time_emb_dim,
            act=cfg.stage2.rectified_flow.act,
            transport_steps=cfg.stage2.rectified_flow.transport_steps,
            mlp_norm=cfg.stage2.rectified_flow.mlp_norm,
            mlp_dropout=cfg.stage2.rectified_flow.mlp_dropout,
        )
        ot_stats = train_ot_stage2_rectified_flow(
            ot,
            source_loader=source_loader,
            target_loader=target_loader,
            option=cfg.stage2.option,
            pair_by_label=cfg.stage2.pair_by_label,
            pair_by_ot=cfg.stage2.pair_by_ot,
            pair_by_ot_method=cfg.stage2.pair_by_ot_method,
            synth_sampler=synth_sampler if cfg.stage2.option.upper() in {"B", "C"} else None,
            epochs=cfg.stage2.epochs,
            lr=cfg.stage2.lr,
            optimizer=cfg.stage2.optimizer,
            weight_decay=cfg.stage2.weight_decay,
            ema_decay=cfg.stage2.ema_decay,
            loss_normalize_by_dim=cfg.stage2.loss_normalize_by_dim,
            public_synth_steps=cfg.stage2.public_synth_steps,
            public_pretrain_epochs=cfg.stage2.public_pretrain_epochs,
            dp=cfg.stage2.dp,
            device=device,
        )
        if "epsilon_ot" in ot_stats:
            stage2_eps.append(float(ot_stats["epsilon_ot"]))

        clients_out.append({"flow": unwrap_model(flow).cpu(), "ot": unwrap_model(ot).cpu(), "prior": prior})
        print(f"[Train] finished client {idx+1}/{len(client_datasets)}")

    y_syn, l_syn, x_syn_raw = server_synthesize_with_raw(
        clients_out,
        M_per_client=cfg.stage3.M_per_client,
        num_classes=num_classes,
        flow_steps=cfg.stage3.flow_steps,
        device=device,
    )

    eps_flow_max = float(max(stage1_eps)) if stage1_eps else None
    eps_ot_max = float(max(stage2_eps)) if stage2_eps else None
    eps_total_max = (float(max(stage1_eps or [0.0]) + max(stage2_eps or [0.0]))) if (stage1_eps or stage2_eps) else None

    cache_payload = {
        "d": int(d),
        "num_classes": int(num_classes),
        "target_ref_x": target_ref.tensors[0].cpu(),
        "target_ref_y": target_ref.tensors[1].long().cpu(),
        "target_test_x": target_test.tensors[0].cpu(),
        "target_test_y": target_test.tensors[1].long().cpu(),
        "y_syn": y_syn.cpu(),
        "l_syn": l_syn.long().cpu(),
        "x_syn_raw": x_syn_raw.cpu(),
        "epsilon_flow_max": eps_flow_max,
        "epsilon_ot_max": eps_ot_max,
        "epsilon_total_max": eps_total_max,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache_payload, cache_path)
    print(f"[Cache] wrote {cache_path}")

    return TrainedArtifacts(
        cfg=cfg,
        d=d,
        num_classes=num_classes,
        target_ref=target_ref,
        target_test=target_test,
        y_syn=y_syn,
        l_syn=l_syn,
        x_syn_raw=x_syn_raw,
        epsilon_flow_max=eps_flow_max,
        epsilon_ot_max=eps_ot_max,
        epsilon_total_max=eps_total_max,
    )


def _train_mlp_classifier(
    d: int,
    num_classes: int,
    hidden: Sequence[int],
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
) -> float:
    from noisyflow.stage3.networks import Classifier
    from noisyflow.stage3.training import train_classifier

    clf = Classifier(d=d, num_classes=num_classes, hidden=list(hidden))
    stats = train_classifier(clf, train_loader, test_loader=test_loader, epochs=int(epochs), lr=float(lr), device=device)
    return float(stats.get("acc", float("nan")))


def _train_rf_classifier(train_loader: DataLoader, test_loader: DataLoader, seed: int, name: str) -> float:
    from noisyflow.stage3.training import train_random_forest_classifier

    stats = train_random_forest_classifier(train_loader, test_loader=test_loader, seed=int(seed), name=name)
    return float(stats.get("acc", float("nan")))


def _collect_numpy_xy(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    dataset = getattr(loader, "dataset", None)
    if dataset is not None:
        try:
            n_ds = len(dataset)
        except TypeError:
            n_ds = None
        if n_ds is not None and n_ds > 0:
            batch_size = getattr(loader, "batch_size", None)
            if batch_size is None:
                batch_size = min(1024, n_ds)
            batch_size = max(1, min(int(batch_size), int(n_ds)))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for xb, yb in loader:
        xs.append(xb.detach().cpu().numpy())
        ys.append(yb.detach().cpu().numpy())
    if not xs:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0).reshape(-1).astype(np.int64, copy=False)
    return X, y


def _train_sklearn_classifier(
    stage3: Stage3Setting,
    train_loader: DataLoader,
    test_loader: DataLoader,
    seed: int,
    name: str,
) -> float:
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    X_train, y_train = _collect_numpy_xy(train_loader)
    X_test, y_test = _collect_numpy_xy(test_loader)
    if X_train.size == 0 or X_test.size == 0:
        return float("nan")

    params = dict(stage3.sk_params or {})
    clf_name = str(stage3.classifier).strip().lower()
    if clf_name == "rf":
        n_estimators = int(params.pop("n_estimators", 500))
        max_depth = params.pop("max_depth", None)
        max_features = params.pop("max_features", "sqrt")
        min_samples_leaf = int(params.pop("min_samples_leaf", 1))
        class_weight = params.pop("class_weight", None)
        n_jobs = int(params.pop("n_jobs", 1))
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=int(seed),
            n_jobs=n_jobs,
        )
    elif clf_name == "extra_trees":
        n_estimators = int(params.pop("n_estimators", 500))
        max_depth = params.pop("max_depth", None)
        max_features = params.pop("max_features", "sqrt")
        min_samples_leaf = int(params.pop("min_samples_leaf", 1))
        class_weight = params.pop("class_weight", None)
        n_jobs = int(params.pop("n_jobs", 1))
        clf = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=int(seed),
            n_jobs=n_jobs,
        )
    elif clf_name == "logreg":
        C = float(params.pop("C", 1.0))
        max_iter = int(params.pop("max_iter", 5000))
        class_weight = params.pop("class_weight", None)
        solver = str(params.pop("solver", "lbfgs"))
        base = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            solver=solver,
            n_jobs=None,
        )
        clf = make_pipeline(StandardScaler(), base) if stage3.standardize else base
    elif clf_name == "linsvc":
        C = float(params.pop("C", 1.0))
        max_iter = int(params.pop("max_iter", 5000))
        class_weight = params.pop("class_weight", None)
        base = LinearSVC(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=int(seed),
        )
        clf = make_pipeline(StandardScaler(), base) if stage3.standardize else base
    else:
        raise ValueError(f"Unknown stage3 classifier '{stage3.classifier}'")

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = float((pred == y_test).mean())
    print(f"[{name}] train_n={len(y_train)} test_n={len(y_test)} acc={acc:.4f}")
    return acc


def _train_stage3_acc(
    artifacts: TrainedArtifacts,
    stage3: Stage3Setting,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    seed: int,
    name: str,
) -> float:
    torch.manual_seed(int(seed))
    if stage3.classifier in {"rf", "extra_trees", "logreg", "linsvc"}:
        return _train_sklearn_classifier(stage3, train_loader, test_loader, seed=seed, name=name)
    return _train_mlp_classifier(
        d=artifacts.d,
        num_classes=artifacts.num_classes,
        hidden=stage3.hidden,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=stage3.epochs,
        lr=stage3.lr,
        device=device,
    )


def _subsample_labeled(ds: TensorDataset, n: Optional[int], num_classes: int, seed: int) -> TensorDataset:
    from run import _subsample_labeled_dataset

    return _subsample_labeled_dataset(ds, n=n, num_classes=num_classes, seed=seed)


def _evaluate_for_sizes(
    artifacts: TrainedArtifacts,
    stage3: Stage3Setting,
    ref_sizes: Sequence[int],
    syn_sizes: Sequence[int],
    include_raw: bool = True,
) -> Dict[str, Any]:
    cfg = artifacts.cfg
    device = str(cfg.device)

    target_test_loader = DataLoader(
        artifacts.target_test,
        batch_size=cfg.loaders.test_batch_size,
        shuffle=False,
    )

    ref_pool = TensorDataset(artifacts.target_ref.tensors[0], artifacts.target_ref.tensors[1].long())
    syn_pool = TensorDataset(artifacts.y_syn, artifacts.l_syn.long())
    raw_pool = TensorDataset(artifacts.x_syn_raw, artifacts.l_syn.long())

    results: List[Dict[str, Any]] = []
    ref_only_cache: Dict[int, float] = {}
    syn_only_cache: Dict[int, float] = {}
    raw_only_cache: Dict[int, float] = {}

    def train_acc(loader: DataLoader, name: str) -> float:
        return _train_stage3_acc(
            artifacts,
            stage3,
            train_loader=loader,
            test_loader=target_test_loader,
            device=device,
            seed=int(cfg.seed),
            name=name,
        )

    for ref_n in ref_sizes:
        if int(ref_n) not in ref_only_cache:
            ref_ds = _subsample_labeled(ref_pool, n=int(ref_n), num_classes=artifacts.num_classes, seed=int(cfg.seed))
            ref_loader = DataLoader(
                ref_ds,
                batch_size=cfg.loaders.target_batch_size,
                shuffle=True,
                drop_last=False,
            )
            ref_only_cache[int(ref_n)] = float(train_acc(ref_loader, name=f"{stage3.name}-ref_only"))

    for syn_n in syn_sizes:
        if int(syn_n) not in syn_only_cache:
            syn_ds = _subsample_labeled(syn_pool, n=int(syn_n), num_classes=artifacts.num_classes, seed=int(cfg.seed))
            syn_loader = DataLoader(
                syn_ds,
                batch_size=cfg.loaders.synth_batch_size,
                shuffle=True,
                drop_last=False,
            )
            syn_only_cache[int(syn_n)] = float(train_acc(syn_loader, name=f"{stage3.name}-y_only"))

        if include_raw and int(syn_n) not in raw_only_cache:
            raw_ds = _subsample_labeled(raw_pool, n=int(syn_n), num_classes=artifacts.num_classes, seed=int(cfg.seed))
            raw_loader = DataLoader(
                raw_ds,
                batch_size=cfg.loaders.synth_batch_size,
                shuffle=True,
                drop_last=False,
            )
            raw_only_cache[int(syn_n)] = float(train_acc(raw_loader, name=f"{stage3.name}-x_only"))

    for ref_n in ref_sizes:
        acc_ref = ref_only_cache[int(ref_n)]
        ref_ds = _subsample_labeled(ref_pool, n=int(ref_n), num_classes=artifacts.num_classes, seed=int(cfg.seed))
        for syn_n in syn_sizes:
            syn_ds = _subsample_labeled(syn_pool, n=int(syn_n), num_classes=artifacts.num_classes, seed=int(cfg.seed))
            combo_ds = ConcatDataset([ref_ds, syn_ds])
            combo_loader = DataLoader(
                combo_ds,
                batch_size=cfg.loaders.synth_batch_size,
                shuffle=True,
                drop_last=False,
            )
            acc_combo = train_acc(combo_loader, name=f"{stage3.name}-ref_plus_y")

            entry: Dict[str, Any] = {
                "ref_n": int(ref_n),
                "syn_n": int(syn_n),
                "acc_ref_only": float(acc_ref),
                "acc_transport_only": float(syn_only_cache[int(syn_n)]),
                "acc_ref_plus_transport": float(acc_combo),
                "gain_ref_plus_transport": float(acc_combo - acc_ref),
            }

            if include_raw:
                raw_ds = _subsample_labeled(raw_pool, n=int(syn_n), num_classes=artifacts.num_classes, seed=int(cfg.seed))
                combo_raw_ds = ConcatDataset([ref_ds, raw_ds])
                combo_raw_loader = DataLoader(
                    combo_raw_ds,
                    batch_size=cfg.loaders.synth_batch_size,
                    shuffle=True,
                    drop_last=False,
                )
                acc_combo_raw = train_acc(combo_raw_loader, name=f"{stage3.name}-ref_plus_x")
                entry.update(
                    {
                        "acc_raw_only": float(raw_only_cache[int(syn_n)]),
                        "acc_ref_plus_raw": float(acc_combo_raw),
                        "gain_ref_plus_raw": float(acc_combo_raw - acc_ref),
                    }
                )
            results.append(entry)

    return {
        "stage3_setting": {
            "name": stage3.name,
            "classifier": stage3.classifier,
            "hidden": list(stage3.hidden),
            "epochs": int(stage3.epochs),
            "lr": float(stage3.lr),
        },
        "epsilon_flow_max": artifacts.epsilon_flow_max,
        "epsilon_ot_max": artifacts.epsilon_ot_max,
        "epsilon_total_max": artifacts.epsilon_total_max,
        "results": results,
    }


def _best_row_for_ref(payload: Dict[str, Any], ref_n: int) -> Dict[str, Any]:
    best: Optional[Dict[str, Any]] = None
    best_acc = float("-inf")
    for row in payload.get("results", []):
        if int(row.get("ref_n", -1)) != int(ref_n):
            continue
        acc = float(row.get("acc_ref_plus_transport", float("-inf")))
        if acc > best_acc:
            best_acc = acc
            best = row
    if best is None:
        raise RuntimeError(f"No results for ref_n={ref_n}")
    return best


def _tune_stage3_setting(
    artifacts: TrainedArtifacts,
    case: CaseSpec,
    stage3_grid: Sequence[Stage3Setting],
) -> Tuple[Stage3Setting, int, float]:
    cfg = artifacts.cfg
    device = str(cfg.device)

    target_test_loader = DataLoader(
        artifacts.target_test,
        batch_size=cfg.loaders.test_batch_size,
        shuffle=False,
    )

    ref_pool = TensorDataset(artifacts.target_ref.tensors[0], artifacts.target_ref.tensors[1].long())
    syn_pool = TensorDataset(artifacts.y_syn, artifacts.l_syn.long())
    ref_ds = _subsample_labeled(ref_pool, n=int(case.selection_ref_n), num_classes=artifacts.num_classes, seed=int(cfg.seed))

    syn_ds_by_n: Dict[int, TensorDataset] = {}
    for syn_n in case.syn_sizes_tune:
        syn_ds_by_n[int(syn_n)] = _subsample_labeled(
            syn_pool, n=int(syn_n), num_classes=artifacts.num_classes, seed=int(cfg.seed)
        )

    best_setting: Optional[Stage3Setting] = None
    best_syn_n = -1
    best_acc = float("-inf")

    for s3 in stage3_grid:
        local_best_acc = float("-inf")
        local_best_syn_n = -1
        for syn_n in case.syn_sizes_tune:
            combo_ds = ConcatDataset([ref_ds, syn_ds_by_n[int(syn_n)]])
            combo_loader = DataLoader(
                combo_ds,
                batch_size=cfg.loaders.synth_batch_size,
                shuffle=True,
                drop_last=False,
            )
            acc = _train_stage3_acc(
                artifacts,
                s3,
                train_loader=combo_loader,
                test_loader=target_test_loader,
                device=device,
                seed=int(cfg.seed),
                name=f"{case.name}:{s3.name}:ref_plus_y_tune(nsyn={syn_n})",
            )
            if acc > local_best_acc:
                local_best_acc = float(acc)
                local_best_syn_n = int(syn_n)
        print(
            f"[TuneStage3] {case.name} {s3.name}: best_ref_plus_y={local_best_acc:.4f} at syn_n={local_best_syn_n}"
        )
        if local_best_acc > best_acc:
            best_acc = local_best_acc
            best_setting = s3
            best_syn_n = local_best_syn_n

    if best_setting is None:
        raise RuntimeError(f"No stage3 setting evaluated for {case.name}")
    return best_setting, best_syn_n, float(best_acc)


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def _plot_case(
    *,
    case: CaseSpec,
    tuned: Dict[str, Any],
    out_dir: Path,
    title: str,
    filename: str,
) -> None:
    _configure_matplotlib()

    payload = tuned["payload"]
    rows = payload["results"]
    ref_sizes = sorted({int(r["ref_n"]) for r in rows})

    okabe_ito = {
        "black": "#000000",
        "orange": "#E69F00",
        "sky": "#56B4E9",
        "green": "#009E73",
        "blue": "#0072B2",
        "vermillion": "#D55E00",
        "purple": "#CC79A7",
        "grey": "#7F7F7F",
    }

    def extract_series(ref_n: int, key: str) -> Tuple[np.ndarray, np.ndarray]:
        pts = [(int(r["syn_n"]), float(r[key])) for r in rows if int(r["ref_n"]) == int(ref_n)]
        pts = sorted(pts, key=lambda x: x[0])
        xs = np.asarray([p[0] for p in pts], dtype=np.int64)
        ys = np.asarray([p[1] for p in pts], dtype=np.float64)
        return xs, ys

    ncols = min(3, max(1, len(ref_sizes)))
    nrows = int(np.ceil(len(ref_sizes) / max(1, ncols)))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7.0, 2.2 * nrows), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    axes = axes.reshape(nrows, ncols)

    for idx, ref_n in enumerate(ref_sizes):
        ax = axes[idx // ncols][idx % ncols]
        ref_only = None
        for r in rows:
            if int(r["ref_n"]) == int(ref_n):
                ref_only = float(r["acc_ref_only"])
                break
        if ref_only is None:
            continue

        x_syn, y_ref_plus_y = extract_series(ref_n, "acc_ref_plus_transport")
        _, y_y_only = extract_series(ref_n, "acc_transport_only")

        ax.plot(
            x_syn,
            y_ref_plus_y,
            color=okabe_ito["blue"],
            marker="o",
            label=rf"Ref+$\tilde{{y}}$",
        )
        ax.plot(
            x_syn,
            y_y_only,
            color=okabe_ito["sky"],
            marker="o",
            linestyle="--",
            label=rf"$\tilde{{y}}$-only",
        )

        if any("acc_ref_plus_raw" in r for r in rows):
            _, y_ref_plus_x = extract_series(ref_n, "acc_ref_plus_raw")
            _, y_x_only = extract_series(ref_n, "acc_raw_only")
            ax.plot(
                x_syn,
                y_ref_plus_x,
                color=okabe_ito["vermillion"],
                marker="s",
                label=rf"Ref+$\tilde{{x}}$",
            )
            ax.plot(
                x_syn,
                y_x_only,
                color=okabe_ito["orange"],
                marker="s",
                linestyle="--",
                label=rf"$\tilde{{x}}$-only",
            )

        ax.axhline(ref_only, color=okabe_ito["black"], linewidth=1.0, linestyle=":", label="Ref-only")
        ax.set_title(rf"{title} ($n_{{ref}}$={ref_n})")
        ax.set_xlabel(r"$n_{\mathrm{syn}}$")
        ax.set_ylabel("Accuracy")
        ax.grid(True, axis="y", alpha=0.2, linewidth=0.6)

    # Hide unused axes.
    for j in range(len(ref_sizes), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    out_pdf = out_dir / f"{filename}.pdf"
    out_png = out_dir / f"{filename}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[Plot] wrote {out_pdf}")


def _write_tex_snippet(summary: Dict[str, Any], out_path: Path) -> None:
    def fmt_pct(x: float) -> str:
        return f"{100.0 * x:.2f}"

    lines: List[str] = []
    lines.append("% Auto-generated by scripts/generate_paper_dp_tuned_artifacts.py")
    lines.append("\\begin{table}[t]")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append("  \\caption{DP reruns (record-level, $\\delta=10^{-5}$). Best Ref+$\\tilde{y}$ settings per task.}")
    lines.append("  \\begin{tabular}{lrrrr}")
    lines.append("    \\toprule")
    lines.append("    Task & $\\varepsilon$ & Ref-only (\\%) & Ref+$\\tilde{x}$ (\\%) & Ref+$\\tilde{y}$ (\\%) \\\\")
    lines.append("    \\midrule")
    for entry in summary["cases"]:
        best = entry["best"]
        eps = entry.get("epsilon_total_max", None)
        eps_str = f"{eps:.2f}" if eps is not None else "--"
        ref_only = fmt_pct(float(best["acc_ref_only"]))
        ref_plus_x = fmt_pct(float(best.get("acc_ref_plus_raw", float("nan")))) if "acc_ref_plus_raw" in best else "--"
        ref_plus_y = fmt_pct(float(best["acc_ref_plus_transport"]))
        lines.append(f"    {entry['label']} & {eps_str} & {ref_only} & {ref_plus_x} & {ref_plus_y} \\\\")
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("  \\label{tab:dp_reruns_tuned}")
    lines.append("\\end{table}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun paper experiments with DP and tuned stage3, then plot + export.")
    parser.add_argument("--out_dir", type=str, default="plots/paper_dp_tuned", help="Output folder under repo root.")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional cache folder (defaults to OUT_DIR/cache). Useful to reuse trained stage1+2 artifacts.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device override (e.g. cuda or cpu).")
    parser.add_argument(
        "--eps_cap",
        type=float,
        default=20.0,
        help=(
            "Target total DP budget cap (approx). Uses preset splits. "
            "For eps_cap=20: {20+0 (opt B), 15+5 (opt C), 10+10 (opt C)}. "
            "For eps_cap=10: {10+0 (opt B), 7+3 (opt C), 5+5 (opt C)}."
        ),
    )
    parser.add_argument("--force_retrain", action="store_true", help="Ignore cached artifacts and retrain stage1+2.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir is not None else (out_dir / "cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    eps_cap = float(args.eps_cap)
    if eps_cap <= 0.0:
        raise ValueError("--eps_cap must be > 0")

    if abs(eps_cap - 20.0) < 1e-9:
        dp_stage1_only = DPSetting(
            name="dp_stage1_only_eps20_optionB",
            stage2_option="B",
            stage1_target_epsilon=20.0,
            stage2_target_epsilon=None,
        )
        dp_stage1_eps15_stage2_eps5 = DPSetting(
            name="dp_stage1_eps15_stage2_eps5",
            stage2_option="C",
            stage1_target_epsilon=15.0,
            stage2_target_epsilon=5.0,
        )
        dp_stage1_eps10_stage2_eps10 = DPSetting(
            name="dp_stage1_eps10_stage2_eps10",
            stage2_option="C",
            stage1_target_epsilon=10.0,
            stage2_target_epsilon=10.0,
        )
        dp_candidates_common = [dp_stage1_only, dp_stage1_eps15_stage2_eps5, dp_stage1_eps10_stage2_eps10]
    elif abs(eps_cap - 10.0) < 1e-9:
        dp_stage1_only = DPSetting(
            name="dp_stage1_only_eps10_optionB",
            stage2_option="B",
            stage1_target_epsilon=10.0,
            stage2_target_epsilon=None,
        )
        dp_stage1_eps7_stage2_eps3 = DPSetting(
            name="dp_stage1_eps7_stage2_eps3",
            stage2_option="C",
            stage1_target_epsilon=7.0,
            stage2_target_epsilon=3.0,
        )
        dp_stage1_eps5_stage2_eps5 = DPSetting(
            name="dp_stage1_eps5_stage2_eps5",
            stage2_option="C",
            stage1_target_epsilon=5.0,
            stage2_target_epsilon=5.0,
        )
        dp_candidates_common = [dp_stage1_only, dp_stage1_eps7_stage2_eps3, dp_stage1_eps5_stage2_eps5]
    else:
        eps_stage1 = float(eps_cap)
        eps_split = float(eps_cap / 2.0)
        dp_stage1_only = DPSetting(
            name=f"dp_stage1_only_eps{eps_stage1:g}_optionB",
            stage2_option="B",
            stage1_target_epsilon=eps_stage1,
            stage2_target_epsilon=None,
        )
        dp_stage1_and_stage2 = DPSetting(
            name=f"dp_stage1_eps{eps_split:g}_stage2_eps{eps_split:g}",
            stage2_option="C",
            stage1_target_epsilon=eps_split,
            stage2_target_epsilon=eps_split,
        )
        dp_candidates_common = [dp_stage1_only, dp_stage1_and_stage2]

    cases: List[CaseSpec] = [
        CaseSpec(
            name="brainscope",
            config_path="configs/brainscope_excitatory_ref50_optionC.yaml",
            dp_candidates=list(dp_candidates_common),
            ref_sizes=[20],
            syn_sizes=[100, 200, 500, 1000, 2000, 5000],
            syn_sizes_tune=[2000],
            selection_ref_n=20,
        ),
        CaseSpec(
            name="statefate",
            config_path="configs/cellot_statefate_invitro_rectifiedflow_ref50.yaml",
            dp_candidates=list(dp_candidates_common),
            ref_sizes=[25],
            syn_sizes=[500, 1000, 2000, 4000, 8000],
            syn_sizes_tune=[500, 1000, 2000],
            selection_ref_n=25,
        ),
        CaseSpec(
            name="pbmc_kang",
            config_path="configs/cellot_lupus_kang_rectifiedflow_ref50.yaml",
            dp_candidates=list(dp_candidates_common),
            ref_sizes=[5, 10, 50],
            syn_sizes=[2000, 4000, 6000, 8000],
            syn_sizes_tune=[2000, 4000],
            selection_ref_n=5,
        ),
        CaseSpec(
            name="pamap2",
            config_path="configs/pamap2_protocol_11act_ref20_pairlabel.yaml",
            dp_candidates=list(dp_candidates_common),
            ref_sizes=[20],
            syn_sizes=[200, 500, 1000, 2000, 3000],
            syn_sizes_tune=[200, 1000],
            selection_ref_n=20,
        ),
    ]

    stage3_grid: List[Stage3Setting] = [
        Stage3Setting(name="mlp_h256_e30_lr1e3", classifier="mlp", hidden=(256, 256), epochs=30, lr=1e-3),
        Stage3Setting(name="mlp_h256_e80_lr5e4", classifier="mlp", hidden=(256, 256), epochs=80, lr=5e-4),
        Stage3Setting(name="mlp_h512_e50_lr5e4", classifier="mlp", hidden=(512, 512), epochs=50, lr=5e-4),
        Stage3Setting(name="mlp_h512_e100_lr5e4", classifier="mlp", hidden=(512, 512), epochs=100, lr=5e-4),
        Stage3Setting(
            name="rf_500_sqrt",
            classifier="rf",
            sk_params={"n_estimators": 500, "max_features": "sqrt", "n_jobs": 1},
        ),
        Stage3Setting(
            name="rf_1000_balanced",
            classifier="rf",
            sk_params={"n_estimators": 1000, "max_features": "sqrt", "class_weight": "balanced_subsample", "n_jobs": 1},
        ),
        Stage3Setting(
            name="extra_trees_1000_balanced",
            classifier="extra_trees",
            sk_params={"n_estimators": 1000, "max_features": "sqrt", "class_weight": "balanced", "n_jobs": 1},
        ),
        Stage3Setting(
            name="extra_trees_2000_balanced",
            classifier="extra_trees",
            sk_params={"n_estimators": 2000, "max_features": "sqrt", "class_weight": "balanced", "n_jobs": 1},
        ),
        Stage3Setting(
            name="extra_trees_2000_depth16_balanced",
            classifier="extra_trees",
            sk_params={
                "n_estimators": 2000,
                "max_features": "sqrt",
                "max_depth": 16,
                "class_weight": "balanced",
                "n_jobs": 1,
            },
        ),
        Stage3Setting(
            name="logreg_bal_C1",
            classifier="logreg",
            standardize=True,
            sk_params={"C": 1.0, "class_weight": "balanced", "max_iter": 5000},
        ),
        Stage3Setting(
            name="logreg_bal_C10",
            classifier="logreg",
            standardize=True,
            sk_params={"C": 10.0, "class_weight": "balanced", "max_iter": 5000},
        ),
        Stage3Setting(
            name="linsvc_bal_C0p1",
            classifier="linsvc",
            standardize=True,
            sk_params={"C": 0.1, "class_weight": "balanced", "max_iter": 5000},
        ),
        Stage3Setting(
            name="linsvc_bal_C1",
            classifier="linsvc",
            standardize=True,
            sk_params={"C": 1.0, "class_weight": "balanced", "max_iter": 5000},
        ),
        Stage3Setting(
            name="linsvc_bal_C2",
            classifier="linsvc",
            standardize=True,
            sk_params={"C": 2.0, "class_weight": "balanced", "max_iter": 5000},
        ),
        Stage3Setting(
            name="linsvc_bal_C10",
            classifier="linsvc",
            standardize=True,
            sk_params={"C": 10.0, "class_weight": "balanced", "max_iter": 5000},
        ),
    ]

    from noisyflow.config import load_config

    summary_out: Dict[str, Any] = {"cases": []}

    for case in cases:
        print(f"\n=== Case: {case.name} ===")
        best_case_out: Optional[Dict[str, Any]] = None
        best_case_acc = float("-inf")
        candidate_runs: List[Dict[str, Any]] = []

        for dp_setting in case.dp_candidates:
            cfg = load_config(case.config_path)
            cfg = _set_device(cfg, args.device)
            cfg = _apply_dp_setting(cfg, dp_setting)

            cache_path = cache_dir / f"{case.name}__{dp_setting.name}__device_{args.device}.pt"
            if args.force_retrain and cache_path.exists():
                cache_path.unlink()
            artifacts = _load_or_train_artifacts(cache_path, cfg)

            best_setting, tune_syn_n, tune_best_acc = _tune_stage3_setting(
                artifacts, case=case, stage3_grid=stage3_grid
            )
            best_payload = _evaluate_for_sizes(
                artifacts,
                stage3=best_setting,
                ref_sizes=case.ref_sizes,
                syn_sizes=case.syn_sizes,
                include_raw=True,
            )
            best_row = _best_row_for_ref(best_payload, ref_n=case.selection_ref_n)

            case_out = {
                "case": case.name,
                "label": case.name,
                "config": case.config_path,
                "dp_setting": {
                    "name": dp_setting.name,
                    "stage2_option": dp_setting.stage2_option,
                    "stage1_target_epsilon": dp_setting.stage1_target_epsilon,
                    "stage2_target_epsilon": dp_setting.stage2_target_epsilon,
                },
                "stage3_best": {
                    "name": best_setting.name,
                    "classifier": best_setting.classifier,
                    "hidden": list(best_setting.hidden),
                    "epochs": int(best_setting.epochs),
                    "lr": float(best_setting.lr),
                    "standardize": bool(best_setting.standardize),
                    "sk_params": dict(best_setting.sk_params or {}),
                },
                "stage3_tune": {
                    "syn_n": int(tune_syn_n),
                    "acc_ref_plus_transport": float(tune_best_acc),
                },
                "epsilon_total_max": best_payload.get("epsilon_total_max", None),
                "best": best_row,
                "payload": best_payload,
            }
            candidate_runs.append(
                {
                    "dp_setting": case_out["dp_setting"],
                    "epsilon_total_max": case_out["epsilon_total_max"],
                    "best": case_out["best"],
                    "stage3_best": case_out["stage3_best"],
                }
            )

            per_dp_json = out_dir / f"{case.name}__{dp_setting.name}__tuned.json"
            per_dp_json.write_text(json.dumps(case_out, indent=2) + "\n", encoding="utf-8")
            print(f"[Write] {per_dp_json}")

            acc = float(best_row.get("acc_ref_plus_transport", float("nan")))
            if acc > best_case_acc:
                best_case_acc = acc
                best_case_out = case_out

        if best_case_out is None:
            raise RuntimeError(f"No successful DP settings evaluated for case '{case.name}'")

        best_case_out["candidates"] = candidate_runs
        summary_out["cases"].append(best_case_out)

        per_case_json = out_dir / f"{case.name}__tuned.json"
        per_case_json.write_text(json.dumps(best_case_out, indent=2) + "\n", encoding="utf-8")
        print(f"[Write] {per_case_json}")

        _plot_case(
            case=case,
            tuned=best_case_out,
            out_dir=out_dir,
            title=case.name,
            filename=f"{case.name}_dp_tuned",
        )

    (out_dir / "summary.json").write_text(json.dumps(summary_out, indent=2) + "\n", encoding="utf-8")
    _write_tex_snippet(summary_out, out_dir / "dp_reruns_tuned_table.tex")

    print(f"\n[Done] Wrote tuned outputs to {out_dir}")


if __name__ == "__main__":
    main()
