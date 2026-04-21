from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


DEFAULT_REPO = "liyy2/aging_YL"
DEFAULT_BRANCH = "master"
DEFAULT_META_PATH = "PEC2_sample_metadata_processed.csv"
DEFAULT_EXPR_PATH = "expression_matrix_9celltypes_07072023/Excitatory_Neur.expr.bed.gz"


def _raw_url(repo: str, branch: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[brainscope] Using cached file: {dest}")
        return

    print(f"[brainscope] Downloading {url}")
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
        total = resp.headers.get("Content-Length")
        total_int = int(total) if total is not None and total.isdigit() else None
        read = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            read += len(chunk)
            if total_int:
                pct = 100.0 * float(read) / float(total_int)
                print(f"\r[brainscope] {pct:5.1f}% ({read}/{total_int} bytes)", end="", file=sys.stderr)
        if total_int:
            print(file=sys.stderr)
    print(f"[brainscope] Saved to {dest}")


def _load_metadata_csv(path: Path):
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("Preparing BrainSCOPE requires pandas. Install with `pip install pandas`.") from exc

    df = pd.read_csv(path)
    required = {"Cohort", "Individual_ID", "Biological_Sex", "Age_death", "Disorder", "1000G_ancestry"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Metadata missing required columns {missing}. Found: {sorted(df.columns)}")
    df["Individual_ID"] = df["Individual_ID"].astype(str)
    df = df.set_index("Individual_ID", drop=False)
    return df


def _read_bed_expression(path: Path) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Reads a BED-like expression matrix from aging_YL.

    Returns:
      sample_ids: list of sample/individual ids (column names after the first 6 columns)
      genes:      (G,) array of gene ids/names (unique after aggregation)
      X:          (N,G) float32 expression matrix (samples x genes)
    """
    import gzip

    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("Preparing BrainSCOPE requires pandas. Install with `pip install pandas`.") from exc

    with gzip.open(path, "rt", encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
    cols = header.split("\t")
    if len(cols) < 7:
        raise ValueError(f"Unexpected header with {len(cols)} columns in {path}")
    sample_ids = cols[6:]
    if not sample_ids:
        raise ValueError(f"No sample columns detected in {path}")

    dtype_map: Dict[str, object] = {"gene": str}
    dtype_map.update({sid: np.float32 for sid in sample_ids})
    usecols = ["gene", *sample_ids]
    df = pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        usecols=usecols,
        dtype=dtype_map,
        engine="c",
    )

    if df["gene"].isna().any():
        raise ValueError("Found NaN gene names in expression file")
    if df["gene"].duplicated().any():
        df = df.groupby("gene", sort=False).mean(numeric_only=True).reset_index()
    df = df.set_index("gene", drop=True)

    df = df[sample_ids]
    genes = df.index.to_numpy()
    X = df.to_numpy(dtype=np.float32, copy=False).T
    return sample_ids, genes, X


def _encode_labels(disorders: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    disorders = np.asarray(disorders).astype(str)
    disorder_norm = np.char.lower(np.char.strip(disorders))
    control_mask = (disorder_norm == "control").astype(np.int64)
    label_case_control = (1 - control_mask).astype(np.int64)
    label_neurodegenerative = np.isin(
        disorders,
        np.array(["Alzheimers/dementia", "cognitive impairment"], dtype=object),
    ).astype(np.int64)
    return label_case_control, label_neurodegenerative


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download and prepare a BrainSCOPE-like cohort expression dataset from aging_YL for NoisyFlow.\n"
            "This produces a compact .npz with X (samples x genes) and aligned metadata."
        )
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/brainscope/brainscope_excitatory_neur.npz",
        help="Output .npz path (default: datasets/brainscope/brainscope_excitatory_neur.npz).",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=DEFAULT_REPO,
        help=f"GitHub repo in owner/name format (default: {DEFAULT_REPO}).",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=DEFAULT_BRANCH,
        help=f"Git branch/tag (default: {DEFAULT_BRANCH}).",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=DEFAULT_META_PATH,
        help=f"Path within repo to metadata CSV (default: {DEFAULT_META_PATH}).",
    )
    parser.add_argument(
        "--expr-path",
        type=str,
        default=DEFAULT_EXPR_PATH,
        help=f"Path within repo to expression .bed.gz (default: {DEFAULT_EXPR_PATH}).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".cache/brainscope_aging_yl",
        help="Where to cache downloads before conversion (default: .cache/brainscope_aging_yl).",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta_url = _raw_url(str(args.repo), str(args.branch), str(args.metadata_path))
    expr_url = _raw_url(str(args.repo), str(args.branch), str(args.expr_path))
    meta_cache = cache_dir / Path(str(args.metadata_path)).name
    expr_cache = cache_dir / Path(str(args.expr_path)).name

    _download(meta_url, meta_cache)
    _download(expr_url, expr_cache)

    meta = _load_metadata_csv(meta_cache)
    sample_ids, genes, X = _read_bed_expression(expr_cache)
    missing = [sid for sid in sample_ids if sid not in meta.index]
    if missing:
        raise RuntimeError(
            f"Expression file has {len(missing)} sample ids missing from metadata, e.g. {missing[:10]}"
        )

    meta_sel = meta.loc[sample_ids]
    cohort = meta_sel["Cohort"].astype(str).to_numpy(dtype=object)
    disorder = meta_sel["Disorder"].astype(str).to_numpy(dtype=object)
    sex = meta_sel["Biological_Sex"].astype(str).to_numpy(dtype=object)
    age_death = meta_sel["Age_death"].astype(str).to_numpy(dtype=object)
    ancestry = meta_sel["1000G_ancestry"].astype(str).to_numpy(dtype=object)

    label_case_control, label_neurodegenerative = _encode_labels(disorder)

    payload = {
        "X": X.astype(np.float32, copy=False),
        "genes": genes.astype(object, copy=False),
        "individual_id": np.asarray(sample_ids, dtype=object),
        "cohort": cohort,
        "disorder": disorder,
        "label_case_control": label_case_control,
        "label_neurodegenerative": label_neurodegenerative,
        "sex": sex,
        "age_death": age_death,
        "ancestry": ancestry,
        "source_repo": np.asarray([str(args.repo)], dtype=object),
        "source_branch": np.asarray([str(args.branch)], dtype=object),
        "source_expr_path": np.asarray([str(args.expr_path)], dtype=object),
        "source_meta_path": np.asarray([str(args.metadata_path)], dtype=object),
    }
    np.savez_compressed(out_path, **payload)
    print(f"[brainscope] Wrote {out_path}")
    print(f"[brainscope] X shape: {X.shape} (samples x genes)")
    print(f"[brainscope] cohorts: {sorted(set(cohort.tolist()))}")
    print(f"[brainscope] case/control positives: {int(label_case_control.sum())}/{int(label_case_control.shape[0])}")
    print(
        f"[brainscope] neurodegenerative positives: {int(label_neurodegenerative.sum())}/{int(label_neurodegenerative.shape[0])}"
    )


if __name__ == "__main__":
    main()
