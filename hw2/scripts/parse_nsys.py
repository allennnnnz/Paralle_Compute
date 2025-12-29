#!/usr/bin/env python3
"""
Parse Nsight Systems CSV exports (nvtx_sum) for multiple runs and plot:
1) Stacked runtime breakdown (I/O, Comm, CPU) vs number of processes (12 cores/node)
2) Speedup vs number of processes using T(1) as baseline
Expected directory layout produced by your SLURM script:
  ./logs/nsys_reports/<CASE>/<RANKS_PADDED>/<rank>/*.csv
where <RANKS_PADDED> is e.g. 01..48.
For each run (a directory like 01, 02, ...), this script finds the rank with the
longest total runtime (Total Time over the three NVTX ranges :CPU, :IO, :Comm),
then uses only that rank's breakdown for plotting.
Dependencies (PyPI/Conda package names):
  - polars
  - matplotlib
Usage:
  python parse_nsys.py --case 33 \
      --root ./logs/nsys_reports \
      --out ./logs/plots \
      [--img-dpi 150]
Outputs (under --out/<CASE>/):
  - runtime_breakdown.png
  - speedup.png
  - summary.csv (N, cpu_s, comm_s, io_s, total_s, speedup)
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Dict, List, Tuple

import polars as pl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# --- Constants ---
NVTX_RANGE_TO_BUCKET = {
    ":CPU": "CPU",
    ":IO": "IO",
    ":Comm": "Comm",
}

TIME_COL = "Total Time (ms)"
RANGE_COL = "Range"
HEADER_PREFIX = "Time (%),"  # line that starts the real CSV header


def read_nvtx_sum_csv(csv_path: Path) -> pl.DataFrame:
    """Read an nsys nvtx_sum CSV, skipping the two preamble lines.
    The file may start with lines like:
      Generating SQLite file ...
      Processing [...] ...
    followed by the actual CSV header starting with 'Time (%),'.
    """
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
    df = pl.read_csv(io.StringIO(data_str))
    return df


def extract_bucket_times_seconds(df: pl.DataFrame) -> Dict[str, float]:
    """Return per-bucket seconds for CPU, IO, Comm from a single-rank CSV."""
    # Ensure needed columns exist
    for col in (TIME_COL, RANGE_COL):
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}' in CSV")

    # Map :CPU/:IO/:Comm -> CPU/IO/Comm and sum Total Time (ms) for each
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

    # Convert to seconds
    return {k: v / 1000.0 for k, v in bucket_sums_ms.items()}


def find_rank_with_max_runtime(csv_files: List[Path]) -> Tuple[Path, Dict[str, float]]:
    """Among CSVs for one run (N ranks), return the file and bucket times for the
    rank with maximum total runtime.
    """
    best_path: Path | None = None
    best_buckets: Dict[str, float] | None = None
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
            # Skip malformed files but log the issue
            print(f"[WARN] Skipping {p}: {e}")

    if best_path is None or best_buckets is None:
        raise RuntimeError("No valid CSVs found to determine max runtime rank.")

    return best_path, best_buckets


def discover_runs(case_dir: Path) -> List[Tuple[int, Path]]:
    """Discover run directories inside the case directory.
    Returns a list of tuples (N_ranks, run_dir) sorted by N.
    Accept directories whose names are numeric (e.g., '01', '4', '16').
    """
    runs: List[Tuple[int, Path]] = []
    for child in sorted(case_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        try:
            n = int(name)
        except ValueError:
            # also allow zero-padded names like 01, 04, etc.
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
        # Find all CSVs under this run dir (any nested directories)
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

    # Speedup: T(1)/T(N). If N=1 not present, fall back to smallest N and warn.
    Ns = df["N"].to_list()
    if 1 in Ns:
        baseline = float(df.filter(pl.col("N") == 1)["total_s"].item())
        baseline_N = 1
    else:
        baseline = float(df.select(pl.col("total_s")).row(0)[0])
        baseline_N = int(df.select(pl.col("N")).row(0)[0])
        print(f"[WARN] N=1 not found; using N={baseline_N} as baseline for speedup.")

    df = df.with_columns(
        pl.lit(baseline).alias("baseline_total_s"),
    ).with_columns(
        (pl.col("baseline_total_s") / pl.col("total_s")).alias("speedup")
    ).drop("baseline_total_s")

    return df


def plot_runtime_breakdown(df: pl.DataFrame, out_path: Path, case: str, dpi: int = 150) -> None:
    x = df["N"].to_list()
    io = df["io_s"].to_list()
    comm = df["comm_s"].to_list()
    cpu = df["cpu_s"].to_list()

    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
    # Stack order: IO (bottom), Comm (middle), CPU (top)
    p1 = ax.bar(x, io, label="I/O", color="#377eb8")
    p2 = ax.bar(x, comm, bottom=io, label="Comm", color="#e41a1c")
    bottom_cpu = [i + c for i, c in zip(io, comm)]
    p3 = ax.bar(x, cpu, bottom=bottom_cpu, label="CPU", color="#4daf4a")

    ax.set_xlabel("# of cores (12 cores/node)")
    ax.set_ylabel("runtime (seconds)")
    ax.set_title(f"Runtime breakdown by cores – case {case}")
    # ax.set_xticks(x)
    ax.xaxis.set_major_locator(MultipleLocator(6))
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_speedup(df: pl.DataFrame, out_path: Path, case: str, dpi: int = 150) -> None:
    x = df["N"].to_list()
    y = df["speedup"].to_list()

    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
    ax.plot(x, y, marker="s")
    ax.set_xlabel("# cores (12 cores/node)")
    ax.set_ylabel("speedup")
    ax.set_title(f"Speedup vs cores – case {case}")
    # ax.set_xticks(x)
    ax.xaxis.set_major_locator(MultipleLocator(6))
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse nsys nvtx_sum CSVs and plot runtime breakdown & speedup.")
    ap.add_argument("--case", required=True, help="Testcase name (directory under --root)")
    ap.add_argument("--root", default="./logs/nsys_reports", help="Root directory containing case subdirectories")
    ap.add_argument("--out", default="./logs/plots", help="Output directory for plots and summary")
    ap.add_argument("--img-dpi", type=int, default=150, help="Image DPI for saved figures")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out) / args.case
    out_root.mkdir(parents=True, exist_ok=True)

    df = collect_summary(root, args.case)

    # Save summary CSV
    summary_csv = out_root / "summary.csv"
    df.write_csv(summary_csv)

    # Plots
    plot_runtime_breakdown(df, out_root / "runtime_breakdown.png", args.case, dpi=args.img_dpi)
    plot_speedup(df, out_root / "speedup.png", args.case, dpi=args.img_dpi)

    print(f"\n[DONE] Summary: {summary_csv}")
    print(f"[DONE] Plots: {out_root/'runtime_breakdown.png'} and {out_root/'speedup.png'}")


if __name__ == "__main__":
    main()