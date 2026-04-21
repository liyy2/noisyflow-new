from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noisyflow.config import load_config
from run import _subsample_labeled_dataset
from scripts.rerun_paper_experiments_dp import _train_once


def _acc_sklearn(clf: Any, train_ds: TensorDataset | ConcatDataset, test_ds: TensorDataset) -> float:
    def collect(ds: TensorDataset | ConcatDataset):
        loader = DataLoader(ds, batch_size=512, shuffle=False, drop_last=False)
        xs, ys = [], []
        for xb, yb in loader:
            xs.append(xb.numpy())
            ys.append(yb.numpy())
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0).reshape(-1)

    x_train, y_train = collect(train_ds)
    x_test, y_test = collect(test_ds)
    clf.fit(x_train, y_train)
    return float((clf.predict(x_test) == y_test).mean())


def main() -> None:
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    cfg = load_config("configs/publication/pamap2_table_seed0.yaml")
    cfg.device = "cpu"
    cfg.stage3.M_per_client = 40
    cfg.stage3.combined_synth_train_size = 200

    art = _train_once(cfg)
    num_classes = int(max(int(art.l_syn.max().item()), int(art.target_ref.tensors[1].max().item())) + 1)
    syn_y = TensorDataset(art.y_syn, art.l_syn)
    syn_x = TensorDataset(art.x_syn_raw, art.l_syn)
    ref = TensorDataset(art.target_ref.tensors[0], art.target_ref.tensors[1].long())
    ref = _subsample_labeled_dataset(ref, n=20, num_classes=num_classes, seed=cfg.seed)
    combo_y = ConcatDataset([ref, syn_y])

    candidates: list[tuple[str, Any]] = []
    for n in [10, 25, 50, 100, 200, 500, 1000]:
        for max_depth in [None, 2, 4, 8, 16]:
            candidates.append(
                (
                    f"rf_n{n}_d{max_depth}",
                    RandomForestClassifier(n_estimators=n, max_depth=max_depth, random_state=0, n_jobs=1),
                )
            )
            candidates.append(
                (
                    f"et_n{n}_d{max_depth}",
                    ExtraTreesClassifier(n_estimators=n, max_depth=max_depth, random_state=0, n_jobs=1),
                )
            )
    for c in [0.1, 1.0, 10.0]:
        candidates.append((f"logreg_C{c}", LogisticRegression(C=c, max_iter=5000, class_weight=None)))
        candidates.append((f"linsvc_C{c}", LinearSVC(C=c, max_iter=5000)))

    rows = []
    for name, clf in candidates:
        raw = _acc_sklearn(clf, syn_x, art.target_test)
        # Re-instantiate classifier to avoid fitted state reuse.
        import copy

        trans = _acc_sklearn(copy.deepcopy(clf), syn_y, art.target_test)
        combo = _acc_sklearn(copy.deepcopy(clf), combo_y, art.target_test)
        row = {"name": name, "raw": raw, "transport": trans, "combo": combo}
        rows.append(row)
        print(row, flush=True)

    out = Path("results/publication_repro/pamap2_classifier_tune_syn200.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
