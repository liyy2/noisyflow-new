from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _run_command(cmd: List[str]) -> Tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError:
        return 127, ""
    return int(proc.returncode), (proc.stdout or "").strip()


def _detect_cpu() -> Dict[str, Any]:
    logical = os.cpu_count() or 0
    physical: Optional[int] = None

    rc, out = _run_command(["lscpu", "-p=core,socket"])
    if rc == 0 and out:
        cores = set()
        for line in out.splitlines():
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            try:
                core_id = int(parts[0])
                socket_id = int(parts[1])
            except ValueError:
                continue
            cores.add((socket_id, core_id))
        if cores:
            physical = len(cores)

    if physical is None:
        physical = logical

    return {
        "physical_cores": int(physical),
        "logical_cores": int(logical),
        "architecture": platform.machine(),
        "processor": platform.processor() or None,
    }


def _parse_meminfo_bytes() -> Tuple[Optional[int], Optional[int]]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None, None
    total_kb: Optional[int] = None
    available_kb: Optional[int] = None
    for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    total_kb = int(parts[1])
                except ValueError:
                    pass
        elif line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    available_kb = int(parts[1])
                except ValueError:
                    pass
    total_b = total_kb * 1024 if total_kb is not None else None
    available_b = available_kb * 1024 if available_kb is not None else None
    return total_b, available_b


def _detect_memory() -> Dict[str, Any]:
    total_b, available_b = _parse_meminfo_bytes()
    if total_b is None:
        return {"total_gb": None, "available_gb": None, "percent_used": None}
    if available_b is None:
        available_b = 0
    total_gb = total_b / (1024**3)
    available_gb = available_b / (1024**3)
    used_gb = max(0.0, total_gb - available_gb)
    percent_used = 100.0 * used_gb / total_gb if total_gb > 0 else None
    return {"total_gb": total_gb, "available_gb": available_gb, "percent_used": percent_used}


def _detect_disk(path: Path) -> Dict[str, Any]:
    usage = shutil.disk_usage(str(path))
    total_gb = usage.total / (1024**3)
    available_gb = usage.free / (1024**3)
    used_gb = usage.used / (1024**3)
    percent_used = 100.0 * used_gb / total_gb if total_gb > 0 else None
    return {"total_gb": total_gb, "available_gb": available_gb, "percent_used": percent_used}


def _detect_nvidia_gpus() -> List[Dict[str, Any]]:
    rc, _ = _run_command(["nvidia-smi", "-L"])
    if rc != 0:
        return []

    rc, out = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if rc != 0 or not out:
        return [{"name": "NVIDIA GPU", "memory_total_mb": None, "driver_version": None}]

    gpus: List[Dict[str, Any]] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        name = parts[0] if parts else "NVIDIA GPU"
        mem_mb: Optional[float] = None
        driver: Optional[str] = None
        if len(parts) >= 2:
            try:
                mem_mb = float(parts[1])
            except ValueError:
                mem_mb = None
        if len(parts) >= 3:
            driver = parts[2] or None
        gpus.append({"name": name, "memory_total_mb": mem_mb, "driver_version": driver})
    return gpus


def _detect_amd_gpus() -> List[Dict[str, Any]]:
    rc, out = _run_command(["rocm-smi", "--showproductname"])
    if rc != 0 or not out:
        return []
    # Best-effort parsing; rocm-smi formats vary.
    gpus: List[Dict[str, Any]] = []
    for line in out.splitlines():
        if "GPU" not in line:
            continue
        gpus.append({"name": line.strip()})
    return gpus


def _detect_gpu() -> Dict[str, Any]:
    nvidia = _detect_nvidia_gpus()
    amd = _detect_amd_gpus()

    backends: List[str] = []
    if nvidia:
        backends.append("CUDA")
    if amd:
        backends.append("ROCm")

    return {
        "nvidia_gpus": nvidia,
        "amd_gpus": amd,
        "apple_silicon": None,
        "total_gpus": int(len(nvidia) + len(amd)),
        "available_backends": backends,
    }


def _recommendations(cpu: Dict[str, Any], memory: Dict[str, Any], disk: Dict[str, Any], gpu: Dict[str, Any]) -> Dict[str, Any]:
    logical = int(cpu.get("logical_cores") or 0)
    if logical >= 8:
        parallel_strategy = "high_parallelism"
        workers = max(1, logical - 2)
    elif logical >= 4:
        parallel_strategy = "moderate_parallelism"
        workers = max(1, logical - 1)
    else:
        parallel_strategy = "sequential"
        workers = 1

    available_gb = memory.get("available_gb")
    if available_gb is None:
        memory_strategy = "unknown"
        memory_note = "Could not detect available RAM."
        memory_libs: List[str] = []
    elif available_gb < 4.0:
        memory_strategy = "memory_constrained"
        memory_note = "Prefer chunking/out-of-core for datasets > ~1GB."
        memory_libs = ["dask", "zarr"]
    elif available_gb < 16.0:
        memory_strategy = "moderate_memory"
        memory_note = "Consider chunking for datasets > ~2GB."
        memory_libs = ["dask", "zarr"]
    else:
        memory_strategy = "memory_abundant"
        memory_note = "Most datasets can be loaded in-memory."
        memory_libs = []

    backends = list(gpu.get("available_backends") or [])
    gpu_available = bool(backends)
    suggested_gpu_libs: List[str] = []
    if "CUDA" in backends:
        suggested_gpu_libs.extend(["torch", "jax", "cupy"])
    if "ROCm" in backends:
        suggested_gpu_libs.extend(["torch"])

    disk_available_gb = disk.get("available_gb")
    if disk_available_gb is None:
        disk_strategy = "unknown"
        disk_note = "Could not detect disk availability."
    elif disk_available_gb < 10.0:
        disk_strategy = "disk_constrained"
        disk_note = "Avoid large intermediate files; prefer streaming and compression."
    elif disk_available_gb < 100.0:
        disk_strategy = "moderate_disk"
        disk_note = "Some room for intermediate files; clean up between runs."
    else:
        disk_strategy = "disk_abundant"
        disk_note = "Sufficient space for large intermediate files and sweeps."

    return {
        "parallel_processing": {
            "strategy": parallel_strategy,
            "suggested_workers": workers,
            "libraries": ["joblib", "multiprocessing"] + (["dask"] if logical >= 8 else []),
        },
        "memory_strategy": {
            "strategy": memory_strategy,
            "libraries": memory_libs,
            "note": memory_note,
        },
        "gpu_acceleration": {
            "available": gpu_available,
            "backends": backends,
            "suggested_libraries": suggested_gpu_libs,
        },
        "large_data_handling": {
            "strategy": disk_strategy,
            "note": disk_note,
        },
    }


def detect_resources(*, workdir: Path) -> Dict[str, Any]:
    """Detect basic system resources and emit strategy recommendations."""
    os_info = {"system": platform.system(), "release": platform.release(), "machine": platform.machine()}
    cpu = _detect_cpu()
    memory = _detect_memory()
    disk = _detect_disk(workdir)
    gpu = _detect_gpu()
    recs = _recommendations(cpu=cpu, memory=memory, disk=disk, gpu=gpu)
    return {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "os": os_info,
        "python": {"version": platform.python_version()},
        "cpu": cpu,
        "memory": memory,
        "disk": disk,
        "gpu": gpu,
        "recommendations": recs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect local compute resources for NoisyFlow runs.")
    parser.add_argument(
        "-o",
        "--output",
        default=".claude_resources.json",
        help="Output JSON path (default: .claude_resources.json).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detected resources to stdout.")
    args = parser.parse_args()

    payload = detect_resources(workdir=Path.cwd())
    out_path = Path(args.output)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.verbose:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Wrote resource report to {out_path}")


if __name__ == "__main__":
    main()
