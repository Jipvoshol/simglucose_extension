"""
Plot the bolus-before-exercise cases where the patch made the clearest
hypo-prevention improvement after the oref0-style prediction fix.

Selected from bolus_before_exercise_pairwise.csv post-fix:
  adult#003 seed=7:  crit_TBR 4.92% -> 0% (patch eliminates hypo entirely)
  adult#003 seed=42: crit_TBR 9.84% -> 4.92% (patch halves hypo time)
  adolescent#005 seed=3: crit_TBR 34.43% -> 31.15% (large absolute reduction)
  adolescent#005 seed=7: crit_TBR 27.87% -> 24.59%
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from thesis_bolus_before_exercise_2307 import (
    build_hr_eda_prebolus_exercise, meals_prebolus, run_arm,
)
from thesis_closed_loop_trial_2307_v2 import plot_three_arm
import thesis_closed_loop_trial_2307_v2 as v2


CASES = [
    ("adult#003", 7, "hypo_eliminated"),
    ("adult#003", 42, "hypo_halved"),
    ("adolescent#005", 3, "hypo_reduced_adol"),
    ("adolescent#005", 7, "hypo_reduced_adol2"),
]


def main():
    out_dir = Path(__file__).parent / "results" / "best_bolus_cases_fixed_controller"
    out_dir.mkdir(exist_ok=True, parents=True)
    start = datetime(2025, 1, 1, 0, 0)

    for cohort, seed, tag in CASES:
        print(f"\n=== {cohort} seed={seed} ({tag}) ===")
        v2.COHORT = cohort
        v2.SEED = seed

        hr, eda = build_hr_eda_prebolus_exercise(start, seed)
        meals = meals_prebolus(seed)

        df_A = run_arm(cohort, seed, "baseline", hr, eda, meals, start)
        df_B = run_arm(cohort, seed, "patch", hr, eda, meals, start)
        df_C = run_arm(cohort, seed, "placebo", hr, eda, meals, start)

        def crit_metrics(df):
            crit = df.between_time("09:00", "12:00")["glucose"]
            tbr = (crit < 70).mean() * 100
            return crit.min(), tbr, crit.mean()

        mA = crit_metrics(df_A)
        mB = crit_metrics(df_B)
        mC = crit_metrics(df_C)
        print(f"  crit window (9-12h) min / TBR% / mean:")
        print(f"    A baseline:  min={mA[0]:.1f}  TBR={mA[1]:.1f}%  mean={mA[2]:.1f}")
        print(f"    B patch:     min={mB[0]:.1f}  TBR={mB[1]:.1f}%  mean={mB[2]:.1f}  (Δ_TBR={mB[1]-mA[1]:+.1f})")
        print(f"    C placebo:   min={mC[0]:.1f}  TBR={mC[1]:.1f}%  mean={mC[2]:.1f}")

        out_path = out_dir / f"case_{tag}_{cohort.replace('#','')}_seed{seed}.png"
        plot_three_arm(df_A, df_B, df_C, meals, out_path,
                       title_extra=f" [{tag}, oref0-style controller]")


if __name__ == "__main__":
    main()
