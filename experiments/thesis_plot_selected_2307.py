"""
Render rich 7-panel plots for a small set of interesting (cohort, seed)
combinations from the batch run. Re-runs each case to get full time series
(batch only stored summary metrics).

Cases are chosen based on batch_2307_pairwise_diff.csv:
  - Most hypo-prone (baseline TBR high)
  - Most hyper-prone (baseline TAR high)
  - Median / easy for context
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[2])
PATCH_DIR = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if PATCH_DIR not in sys.path:
    sys.path.insert(0, PATCH_DIR)

# Reuse the v2 machinery
from thesis_closed_loop_trial_2307_v2 import (
    build_hr_eda_series, jitter_meals, run_arm, plot_three_arm,
)


# Selected cases: mix of hypo-prone and hyper-prone
SELECTED_CASES = [
    # Hypo-prone (baseline struggles with hypo)
    ("adolescent#003", 1, "hypo_prone"),   # baseline TBR=41.2%
    ("adolescent#003", 3, "hypo_prone"),   # baseline TBR=35.6%
    ("child#005", 1, "hypo_prone"),         # baseline TBR
    # Hyper-prone (baseline struggles with hyper)
    ("adult#005", 7, "hyper_prone"),        # baseline TAR=37.9%
    ("adolescent#005", 42, "hyper_prone"),  # baseline TAR=22.9%, max 323
    # Balanced / easy (reference)
    ("adolescent#001", 7, "balanced"),     # closer to 100% TIR
    ("adult#001", 3, "balanced"),
]

BASE_MEALS = [
    (datetime(2025, 1, 1, 7, 30), 50),
    (datetime(2025, 1, 1, 12, 0), 70),
    (datetime(2025, 1, 1, 18, 0), 60),
    (datetime(2025, 1, 1, 10, 45), 15),
    (datetime(2025, 1, 1, 21, 15), 12),
]


def main():
    import thesis_closed_loop_trial_2307_v2 as v2

    out_dir = Path(__file__).parent / "results" / "selected_cases"
    out_dir.mkdir(exist_ok=True, parents=True)
    start = datetime(2025, 1, 1, 0, 0)

    for cohort, seed, tag in SELECTED_CASES:
        print(f"\n=== {cohort} seed={seed} ({tag}) ===")
        # Temporarily override v2's COHORT/SEED globals for plot title
        v2.COHORT = cohort
        v2.SEED = seed

        hr, eda = build_hr_eda_series(start, seed)
        meals = jitter_meals(BASE_MEALS, seed)

        df_A = run_arm("A blind", "baseline", hr, eda, meals, start)
        df_B = run_arm("B patch", "patch", hr, eda, meals, start)
        df_C = run_arm("C placebo", "placebo", hr, eda, meals, start)

        # Quick per-case metrics
        def m(df):
            g = df["glucose"]
            return (g.mean(), g.min(), g.max(),
                    ((g >= 70) & (g <= 180)).mean()*100,
                    (g < 70).mean()*100, (g > 180).mean()*100,
                    df["insulin"].sum() * 3)
        mA, mB, mC = m(df_A), m(df_B), m(df_C)
        print(f"  {'arm':>6} {'mean':>6} {'min':>6} {'max':>6} {'TIR':>5} {'TBR':>5} {'TAR':>5} {'ins':>5}")
        for label, mm in [("A", mA), ("B", mB), ("C", mC)]:
            print(f"  {label:>6} {mm[0]:>6.1f} {mm[1]:>6.1f} {mm[2]:>6.1f} "
                  f"{mm[3]:>5.1f} {mm[4]:>5.1f} {mm[5]:>5.1f} {mm[6]:>5.1f}")

        out_path = out_dir / f"case_{tag}_{cohort.replace('#','')}_seed{seed}.png"
        plot_three_arm(df_A, df_B, df_C, meals, out_path,
                       title_extra=f" [{tag}]")


if __name__ == "__main__":
    main()
