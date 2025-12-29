#!/usr/bin/env python3
"""
Compare two runs (baseline vs. experiment) and plot SPEEDUP using Polars + Matplotlib.
Inputs:
  - Two CSVs (e.g., the 'summary.csv' produced by your pipeline)
  - Each must contain:
        N           (int)  number of processes
    AND either
        total_s     (float) total runtime seconds (script will compute speedup), OR
        speedup     (float) precomputed speedup
Behavior:
  - If 'speedup' is missing, compute it as T(1)/T(N). If N=1 is missing,
    uses the smallest N as the baseline (and prints a warning).
  - Plots two series labeled "baseline" and "experiment".
  - X axis: "# processes (12 cores/node)"
  - Y axis: "speedup"
  - X major ticks every 6 (MultipleLocator(6)).
Usage:
  python compare_speedup.py \
      --baseline ./logs/plots/CASE_BASELINE/summary.csv \
      --experiment ./logs/plots/CASE_EXPERIMENT/summary.csv \
      --out ./logs/plots/compare_speedup.png \
      [--title "Speedup – baseline vs experiment"] \
      [--img-dpi 150]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import polars as pl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def _ensure_speedup(df: pl.DataFrame, label: str) -> pl.DataFrame:
    """
    Ensure the DataFrame has columns: N (int), speedup (float).
    If 'speedup' is absent, compute it from 'total_s' using T(1) (or min N).
    """
    cols = set(df.columns)

    if "N" not in cols:
        raise KeyError(f"[{label}] CSV must contain column 'N'.")

    # If 'speedup' provided, keep it and return trimmed columns.
    if "speedup" in cols:
        out = df.select(
            pl.col("N").cast(pl.Int64).alias("N"),
            pl.col("speedup").cast(pl.Float64).alias("speedup"),
        ).sort("N")
        return out

    # Otherwise compute from total_s
    if "total_s" not in cols:
        raise KeyError(f"[{label}] CSV must contain 'total_s' if 'speedup' is not present.")

    df = df.select(
        pl.col("N").cast(pl.Int64).alias("N"),
        pl.col("total_s").cast(pl.Float64).alias("total_s"),
    ).sort("N")

    Ns = df["N"].to_list()
    if 1 in Ns:
        baseline = float(df.filter(pl.col("N") == 1)["total_s"].item())
        baseline_N = 1
    else:
        baseline = float(df.select("total_s").row(0)[0])
        baseline_N = int(df.select("N").row(0)[0])
        print(f"[WARN] [{label}] N=1 not found; using N={baseline_N} as baseline for speedup.")

    out = df.with_columns(
        (pl.lit(baseline) / pl.col("total_s")).alias("speedup")
    ).select("N", "speedup")
    return out


def _load_speedup(csv_path: Path, label: str) -> pl.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"[{label}] CSV not found: {csv_path}")
    df = pl.read_csv(csv_path)
    return _ensure_speedup(df, label)


def plot_speedup_compare(
    df_base: pl.DataFrame,
    df_exp: pl.DataFrame,
    out_path: Path,
    title: str | None = None,
    dpi: int = 300,
) -> None:
    # Convert to Python lists for plotting
    x_base = df_base["N"].to_list()
    y_base = df_base["speedup"].to_list()
    x_exp  = df_exp["N"].to_list()
    y_exp  = df_exp["speedup"].to_list()

    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
    ax.plot(x_base, y_base, marker="o", label="baseline")
    ax.plot(x_exp,  y_exp,  marker="s", label="experiment")

    ax.set_xlabel("# cores (12 cores/node)")
    ax.set_ylabel("speedup")
    ax.set_title(title or "Speedup comparison: baseline vs experiment")
    ax.xaxis.set_major_locator(MultipleLocator(6))
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved plot -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot speedup comparison (baseline vs experiment).")
    ap.add_argument("--baseline", required=True, help="Path to baseline summary CSV")
    ap.add_argument("--experiment", required=True, help="Path to experiment summary CSV")
    ap.add_argument("--out", default="./logs/plots/speedup_compare.png", help="Output image path")
    ap.add_argument("--title", default=None, help="Optional plot title")
    ap.add_argument("--img-dpi", type=int, default=150, help="Image DPI for saved figure")
    args = ap.parse_args()

    base_df = _load_speedup(Path(args.baseline), "baseline")
    exp_df  = _load_speedup(Path(args.experiment), "experiment")

    # If the two runs have different N sets, that's fine—each speedup is relative to its own baseline.
    # We plot both series as-is.
    plot_speedup_compare(base_df, exp_df, Path(args.out), args.title, dpi=args.img_dpi)


if __name__ == "__main__":
    main()