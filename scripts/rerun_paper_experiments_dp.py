from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset


@dataclass(frozen=True)
class DPSetting:
    name: str
    stage2_option: str
    stage1_target_epsilon: float
    stage2_target_epsilon: Optional[float]


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


def _train_once(cfg) -> TrainedArtifacts:
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
        train_ds = ds
        train_loader = DataLoader(
            train_ds,
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
            labels = train_ds.tensors[1]
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
            raise ValueError("This rerun script currently supports stage2.rectified_flow.enabled=true only.")

        source_loader = None
        if cfg.stage2.option.upper() in {"A", "C"}:
            source_loader = DataLoader(
                train_ds,
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

        flow_cpu = unwrap_model(flow).cpu()
        ot_cpu = unwrap_model(ot).cpu()
        clients_out.append({"flow": flow_cpu, "ot": ot_cpu, "prior": prior})

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
    stats = train_classifier(clf, train_loader, test_loader=test_loader, epochs=epochs, lr=lr, device=device)
    return float(stats.get("acc", float("nan")))


def _train_rf_classifier(train_loader: DataLoader, test_loader: DataLoader, seed: int, name: str) -> float:
    from noisyflow.stage3.training import train_random_forest_classifier

    stats = train_random_forest_classifier(train_loader, test_loader=test_loader, seed=seed, name=name)
    return float(stats.get("acc", float("nan")))


def _should_use_rf(cfg) -> bool:
    choice = str(getattr(cfg.stage3, "classifier", "auto")).strip().lower()
    if choice in {"auto", "rf", "random_forest"}:
        return choice != "mlp"
    if choice in {"mlp"}:
        return False
    raise ValueError(f"stage3.classifier must be one of: auto, rf, mlp (got '{choice}')")


def _subsample_labeled(ds: TensorDataset, n: Optional[int], num_classes: int, seed: int) -> TensorDataset:
    from run import _subsample_labeled_dataset

    return _subsample_labeled_dataset(ds, n=n, num_classes=num_classes, seed=seed)


def _evaluate_for_sizes(
    artifacts: TrainedArtifacts,
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

    # Pre-build labeled pools.
    ref_pool = TensorDataset(artifacts.target_ref.tensors[0], artifacts.target_ref.tensors[1].long())
    syn_pool = TensorDataset(artifacts.y_syn, artifacts.l_syn.long())
    raw_pool = TensorDataset(artifacts.x_syn_raw, artifacts.l_syn.long())

    results: List[Dict[str, Any]] = []
    ref_only_cache: Dict[int, float] = {}
    syn_only_cache: Dict[int, float] = {}
    raw_only_cache: Dict[int, float] = {}

    for ref_n in ref_sizes:
        if ref_n not in ref_only_cache:
            torch.manual_seed(int(cfg.seed))
            ref_ds = _subsample_labeled(ref_pool, n=int(ref_n), num_classes=artifacts.num_classes, seed=int(cfg.seed))
            ref_loader = DataLoader(
                ref_ds,
                batch_size=cfg.loaders.target_batch_size,
                shuffle=True,
                drop_last=False,
            )
            if _should_use_rf(cfg):
                try:
                    acc_ref = _train_rf_classifier(ref_loader, target_test_loader, seed=int(cfg.seed), name="RF-ref_only")
                except RuntimeError:
                    acc_ref = _train_mlp_classifier(
                        d=artifacts.d,
                        num_classes=artifacts.num_classes,
                        hidden=cfg.stage3.hidden,
                        train_loader=ref_loader,
                        test_loader=target_test_loader,
                        epochs=cfg.stage3.epochs,
                        lr=cfg.stage3.lr,
                        device=device,
                    )
            else:
                acc_ref = _train_mlp_classifier(
                    d=artifacts.d,
                    num_classes=artifacts.num_classes,
                    hidden=cfg.stage3.hidden,
                    train_loader=ref_loader,
                    test_loader=target_test_loader,
                    epochs=cfg.stage3.epochs,
                    lr=cfg.stage3.lr,
                    device=device,
                )
            ref_only_cache[int(ref_n)] = float(acc_ref)

    for syn_n in syn_sizes:
        if syn_n not in syn_only_cache:
            torch.manual_seed(int(cfg.seed))
            syn_ds = _subsample_labeled(syn_pool, n=int(syn_n), num_classes=artifacts.num_classes, seed=int(cfg.seed))
            syn_loader = DataLoader(
                syn_ds,
                batch_size=cfg.loaders.synth_batch_size,
                shuffle=True,
                drop_last=False,
            )
            if _should_use_rf(cfg):
                try:
                    acc_syn = _train_rf_classifier(syn_loader, target_test_loader, seed=int(cfg.seed), name="RF-syn_only")
                except RuntimeError:
                    acc_syn = _train_mlp_classifier(
                        d=artifacts.d,
                        num_classes=artifacts.num_classes,
                        hidden=cfg.stage3.hidden,
                        train_loader=syn_loader,
                        test_loader=target_test_loader,
                        epochs=cfg.stage3.epochs,
                        lr=cfg.stage3.lr,
                        device=device,
                    )
            else:
                acc_syn = _train_mlp_classifier(
                    d=artifacts.d,
                    num_classes=artifacts.num_classes,
                    hidden=cfg.stage3.hidden,
                    train_loader=syn_loader,
                    test_loader=target_test_loader,
                    epochs=cfg.stage3.epochs,
                    lr=cfg.stage3.lr,
                    device=device,
                )
            syn_only_cache[int(syn_n)] = float(acc_syn)

        if include_raw and syn_n not in raw_only_cache:
            torch.manual_seed(int(cfg.seed))
            raw_ds = _subsample_labeled(raw_pool, n=int(syn_n), num_classes=artifacts.num_classes, seed=int(cfg.seed))
            raw_loader = DataLoader(
                raw_ds,
                batch_size=cfg.loaders.synth_batch_size,
                shuffle=True,
                drop_last=False,
            )
            if _should_use_rf(cfg):
                try:
                    acc_raw = _train_rf_classifier(raw_loader, target_test_loader, seed=int(cfg.seed), name="RF-raw_only")
                except RuntimeError:
                    acc_raw = _train_mlp_classifier(
                        d=artifacts.d,
                        num_classes=artifacts.num_classes,
                        hidden=cfg.stage3.hidden,
                        train_loader=raw_loader,
                        test_loader=target_test_loader,
                        epochs=cfg.stage3.epochs,
                        lr=cfg.stage3.lr,
                        device=device,
                    )
            else:
                acc_raw = _train_mlp_classifier(
                    d=artifacts.d,
                    num_classes=artifacts.num_classes,
                    hidden=cfg.stage3.hidden,
                    train_loader=raw_loader,
                    test_loader=target_test_loader,
                    epochs=cfg.stage3.epochs,
                    lr=cfg.stage3.lr,
                    device=device,
                )
            raw_only_cache[int(syn_n)] = float(acc_raw)

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
            torch.manual_seed(int(cfg.seed))
            if _should_use_rf(cfg):
                try:
                    acc_combo = _train_rf_classifier(
                        combo_loader, target_test_loader, seed=int(cfg.seed), name="RF-ref_plus_transport"
                    )
                except RuntimeError:
                    acc_combo = _train_mlp_classifier(
                        d=artifacts.d,
                        num_classes=artifacts.num_classes,
                        hidden=cfg.stage3.hidden,
                        train_loader=combo_loader,
                        test_loader=target_test_loader,
                        epochs=cfg.stage3.epochs,
                        lr=cfg.stage3.lr,
                        device=device,
                    )
            else:
                acc_combo = _train_mlp_classifier(
                    d=artifacts.d,
                    num_classes=artifacts.num_classes,
                    hidden=cfg.stage3.hidden,
                    train_loader=combo_loader,
                    test_loader=target_test_loader,
                    epochs=cfg.stage3.epochs,
                    lr=cfg.stage3.lr,
                    device=device,
                )

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
                torch.manual_seed(int(cfg.seed))
                if _should_use_rf(cfg):
                    try:
                        acc_combo_raw = _train_rf_classifier(
                            combo_raw_loader, target_test_loader, seed=int(cfg.seed), name="RF-ref_plus_raw"
                        )
                    except RuntimeError:
                        acc_combo_raw = _train_mlp_classifier(
                            d=artifacts.d,
                            num_classes=artifacts.num_classes,
                            hidden=cfg.stage3.hidden,
                            train_loader=combo_raw_loader,
                            test_loader=target_test_loader,
                            epochs=cfg.stage3.epochs,
                            lr=cfg.stage3.lr,
                            device=device,
                        )
                else:
                    acc_combo_raw = _train_mlp_classifier(
                        d=artifacts.d,
                        num_classes=artifacts.num_classes,
                        hidden=cfg.stage3.hidden,
                        train_loader=combo_raw_loader,
                        test_loader=target_test_loader,
                        epochs=cfg.stage3.epochs,
                        lr=cfg.stage3.lr,
                        device=device,
                    )
                entry.update(
                    {
                        "acc_raw_only": float(raw_only_cache[int(syn_n)]),
                        "acc_ref_plus_raw": float(acc_combo_raw),
                        "gain_ref_plus_raw": float(acc_combo_raw - acc_ref),
                    }
                )
            results.append(entry)

    payload: Dict[str, Any] = {
        "config": str(getattr(cfg, "source_path", "")) if hasattr(cfg, "source_path") else None,
        "epsilon_flow_max": artifacts.epsilon_flow_max,
        "epsilon_ot_max": artifacts.epsilon_ot_max,
        "epsilon_total_max": artifacts.epsilon_total_max,
        "results": results,
    }
    return payload


def _max_gain(payload: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    best = None
    best_gain = float("-inf")
    for row in payload.get("results", []):
        g = float(row.get("gain_ref_plus_transport", float("-inf")))
        if g > best_gain:
            best_gain = g
            best = row
    if best is None:
        raise RuntimeError("No results produced")
    return best_gain, best


def _run_case(
    base_cfg_path: str,
    setting: DPSetting,
    ref_sizes: Sequence[int],
    syn_sizes: Sequence[int],
    out_path: Path,
    include_raw: bool = True,
    device_override: Optional[str] = None,
) -> Dict[str, Any]:
    from noisyflow.config import load_config

    cfg = load_config(base_cfg_path)
    cfg.source_path = base_cfg_path  # type: ignore[attr-defined]
    if device_override is not None:
        cfg.device = str(device_override)
    dp_cfg = _apply_dp_setting(cfg, setting)
    artifacts = _train_once(dp_cfg)
    payload = _evaluate_for_sizes(artifacts, ref_sizes=ref_sizes, syn_sizes=syn_sizes, include_raw=include_raw)
    payload["dp_setting"] = {
        "name": setting.name,
        "stage2_option": setting.stage2_option,
        "stage1_target_epsilon": setting.stage1_target_epsilon,
        "stage2_target_epsilon": setting.stage2_target_epsilon,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    parser = argparse.ArgumentParser(
        description="Rerun paper experiments with DP and summarize Ref+ŷ vs Ref-only gains."
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device override for runs (default: cuda). Use 'cpu' if needed.",
    )
    parser.add_argument(
        "--out_dir",
        default="plots/paper_dp_reruns",
        help="Output directory for JSON results.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    device = str(args.device)

    settings = [
        DPSetting(
            name="dp_stage1_only_eps20_optionB",
            stage2_option="B",
            stage1_target_epsilon=20.0,
            stage2_target_epsilon=None,
        ),
        DPSetting(
            name="dp_stage1_eps10_stage2_eps10",
            stage2_option="C",
            stage1_target_epsilon=10.0,
            stage2_target_epsilon=10.0,
        ),
    ]

    cases = [
        {
            "name": "brainscope_ref20",
            "config": "configs/brainscope_excitatory_ref50_optionC.yaml",
            "ref_sizes": [20],
            "syn_sizes": [100, 200, 500, 1000, 2000],
            "include_raw": True,
        },
        {
            "name": "statefate_ref25",
            "config": "configs/cellot_statefate_invitro_rectifiedflow_ref50.yaml",
            "ref_sizes": [25],
            "syn_sizes": [500, 1000, 2000, 4000],
            "include_raw": False,
        },
        {
            "name": "pbmc_kang_fewshot",
            "config": "configs/cellot_lupus_kang_rectifiedflow_ref50.yaml",
            "ref_sizes": [5, 10, 50],
            "syn_sizes": [2000, 4000, 6000],
            "include_raw": False,
        },
        {
            "name": "pamap2_ref20",
            "config": "configs/pamap2_protocol_11act_ref20_pairlabel.yaml",
            "ref_sizes": [20],
            "syn_sizes": [200, 500, 1000, 2000],
            "include_raw": True,
        },
    ]

    summary: Dict[str, Any] = {"cases": []}
    for case in cases:
        best_overall: Optional[Dict[str, Any]] = None
        best_gain = float("-inf")
        for setting in settings:
            out_path = out_dir / f"{case['name']}__{setting.name}.json"
            payload = _run_case(
                base_cfg_path=str(case["config"]),
                setting=setting,
                ref_sizes=case["ref_sizes"],
                syn_sizes=case["syn_sizes"],
                out_path=out_path,
                include_raw=bool(case["include_raw"]),
                device_override=device,
            )
            gain, best_row = _max_gain(payload)
            eps = payload.get("epsilon_total_max", None)
            record = {
                "case": case["name"],
                "config": case["config"],
                "dp_setting": payload["dp_setting"],
                "epsilon_total_max": eps,
                "best_row": best_row,
                "best_gain": gain,
                "json": str(out_path),
            }
            if gain > best_gain:
                best_gain = gain
                best_overall = record
        assert best_overall is not None
        summary["cases"].append(best_overall)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote summary to {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
