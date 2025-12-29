#!/usr/bin/env python3
"""
Dedicated parser for hw2b Nsight Systems CSV exports (nvtx_sum).
Same behavior as parse_nsys.py but kept separate per request, and expects a
baseline run at N=1 (n1c1) to compute proper speedup.
Usage:
  python scripts/parse_nsys_hw2b.py --case slow01_hw2b \
      --root ./logs/nsys_reports --out ./logs/plots --img-dpi 600
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Dict, List, Tuple

import polars as pl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

NVTX_RANGE_TO_BUCKET = {":CPU": "CPU", ":IO": "IO", ":Comm": "Comm"}
TIME_COL = "Total Time (ms)"
RANGE_COL = "Range"
HEADER_PREFIX = "Time (%),"


def read_nvtx_sum_csv(csv_path: Path) -> pl.DataFrame:
    text = csv_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines(True)
    start = 0
    for i, line in enumerate(lines):
        if line.startswith(HEADER_PREFIX):
            start = i
            break
    data_str = "".join(lines[start:])
    if not data_str:
        raise ValueError(f"Could not find CSV header in {csv_path}")
    return pl.read_csv(io.StringIO(data_str))


def extract_bucket_times_seconds(df: pl.DataFrame) -> Dict[str, float]:
    for col in (TIME_COL, RANGE_COL):
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}' in CSV")
    bucket_sums_ms: Dict[str, float] = {"CPU": 0.0, "IO": 0.0, "Comm": 0.0}
    for key, bucket in NVTX_RANGE_TO_BUCKET.items():
        ms = (
            df.filter(pl.col(RANGE_COL) == key)
              .select(pl.col(TIME_COL).cast(pl.Float64))
              .sum()
              .item()
        )
        if ms is None:
            ms = 0.0
        bucket_sums_ms[bucket] = float(ms)
    return {k: v / 1000.0 for k, v in bucket_sums_ms.items()}


def find_rank_with_max_runtime(csv_files: List[Path]):
    best_path = None
    best_buckets = None
    best_total = -1.0
    for p in csv_files:
        try:
            df = read_nvtx_sum_csv(p)
            buckets = extract_bucket_times_seconds(df)
            total = buckets["CPU"] + buckets["IO"] + buckets["Comm"]
            if total > best_total:
                best_total = total
                best_path = p
                best_buckets = buckets
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")
    if best_path is None or best_buckets is None:
        raise RuntimeError("No valid CSVs found to determine max runtime rank.")
    return best_path, best_buckets


def discover_runs(case_dir: Path) -> List[Tuple[int, Path]]:
    runs: List[Tuple[int, Path]] = []
    for child in sorted(case_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        try:
            n = int(name)
        except ValueError:
            try:
                n = int(name.lstrip("0") or "0")
            except ValueError:
                continue
        runs.append((n, child))
    runs.sort(key=lambda x: x[0])
    return runs


def collect_summary(root: Path, case: str) -> pl.DataFrame:
    case_dir = root / case
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    rows = []
    for N, run_dir in discover_runs(case_dir):
        csvs = list(run_dir.rglob("*.csv"))
        if not csvs:
            print(f"[INFO] No CSVs in {run_dir}, skipping.")
            continue
        best_path, best_buckets = find_rank_with_max_runtime(csvs)
        cpu_s = best_buckets["CPU"]
        io_s = best_buckets["IO"]
        comm_s = best_buckets["Comm"]
        total_s = cpu_s + io_s + comm_s
        rows.append({
            "N": N,
            "cpu_s": cpu_s,
            "io_s": io_s,
            "comm_s": comm_s,
            "total_s": total_s,
            "rank_file": str(best_path),
        })
        print(f"[OK] N={N}: max-rank CSV -> {best_path} (total {total_s:.3f}s)")

    if not rows:
        raise RuntimeError("No runs with valid CSVs found.")

    df = pl.DataFrame(rows).sort("N")

    # Enforce baseline N=1 for hw2b speedup
    Ns = df["N"].to_list()
    if 1 not in Ns:
        raise RuntimeError("Baseline N=1 missing for hw2b; ensure n1c1 run exists.")
    baseline = float(df.filter(pl.col("N") == 1)["total_s"].item())

    df = df.with_columns(pl.lit(baseline).alias("baseline_total_s")) \
           .with_columns((pl.col("baseline_total_s") / pl.col("total_s")).alias("speedup")) \
           .drop("baseline_total_s")
    return df


def plot_runtime_breakdown(df: pl.DataFrame, out_path: Path, case: str, dpi: int = 150) -> None:
    x = df["N"].to_list()
    io = df["io_s"].to_list()
    comm = df["comm_s"].to_list()
    cpu = df["cpu_s"].to_list()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
    p1 = ax.bar(x, io, label="I/O", color="#377eb8")
    p2 = ax.bar(x, comm, bottom=io, label="Comm", color="#e41a1c")
    bottom_cpu = [i + c for i, c in zip(io, comm)]
    p3 = ax.bar(x, cpu, bottom=bottom_cpu, label="CPU", color="#4daf4a")

    ax.set_xlabel("# of cores (12 cores/node)")
    ax.set_ylabel("runtime (seconds)")
    ax.set_title(f"Runtime breakdown by cores – case {case}")
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_speedup(df: pl.DataFrame, out_path: Path, case: str, dpi: int = 150) -> None:
    x = df["N"].to_list()
    y = df["speedup"].to_list()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
    ax.plot(x, y, marker="s")
    ax.set_xlabel("# cores (12 cores/node)")
    ax.set_ylabel("speedup")
    ax.set_title(f"Speedup vs cores – case {case}")
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse nsys nvtx_sum CSVs for hw2b and plot with N=1 baseline.")
    ap.add_argument("--case", required=True)
    ap.add_argument("--root", default="./logs/nsys_reports")
    ap.add_argument("--out", default="./logs/plots")
    ap.add_argument("--img-dpi", type=int, default=150)
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out) / args.case
    out_root.mkdir(parents=True, exist_ok=True)

    df = collect_summary(root, args.case)
    summary_csv = out_root / "summary.csv"
    df.write_csv(summary_csv)

    plot_runtime_breakdown(df, out_root / "runtime_breakdown.png", args.case, dpi=args.img_dpi)
    plot_speedup(df, out_root / "speedup.png", args.case, dpi=args.img_dpi)

    print(f"\n[DONE] Summary: {summary_csv}")
    print(f"[DONE] Plots: {out_root/'runtime_breakdown.png'} and {out_root/'speedup.png'}")


if __name__ == "__main__":
    main()