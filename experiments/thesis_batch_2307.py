"""
Batch closed-loop trial for the 2307 patch across multiple virtual patients
and random seeds. Looks for runs where baseline controller struggles
(hypo or hyper events) and measures whether the patch prevents/worsens them.

Reuses the v2 scenario + controller setup; just loops over (cohort, seed).
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

from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario import CustomScenario

from simglucose_ctx.env_wrapper import ContextAwareT1DSimEnv
from simglucose_ctx.context import ContextConfig, ContextStream
from simglucose_ctx.context_aware_openaps import ContextAwareOpenAPSController


HR_REST_PATIENT = 68.0
HR_REST_PATCH = 68.0
PATCH_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../manchester/F9_cap25_patch/patch/isf_patch_2307.pkl"))

COHORTS = ["adolescent#001", "adolescent#003", "adolescent#005",
           "adult#001", "adult#003", "adult#005",
           "child#001", "child#005"]
SEEDS = [1, 3, 7, 13, 42]


def build_hr_eda(start, seed):
    idx = pd.date_range(start, periods=480, freq="3min")
    rng = np.random.RandomState(seed)
    t = np.arange(480)
    circ = 68 + 6 * np.sin(2 * np.pi * t / 480 - 1.2)
    hr = circ + rng.randn(480) * 3.0
    hr[180:220] += 75   # exercise
    hr[220:260] += 25   # cooldown
    hr[280:300] += 18   # stress
    hr[360:390] += 25   # evening walk
    eda = np.full(480, 0.12)
    eda[280:300] = 0.7
    eda[220:240] = 0.45
    eda[360:390] = 0.2
    for _ in range(8):
        c = rng.randint(12, 468)
        w = rng.randint(2, 6)
        hr[max(0, c-w):c+w] += rng.uniform(8, 22)
        eda[max(0, c-w):c+w] += rng.uniform(0.0, 0.25)
    for _ in range(4):
        c = rng.randint(12, 468)
        w = rng.randint(3, 7)
        hr[max(0, c-w):c+w] -= rng.uniform(5, 12)
    hr += rng.randn(len(hr)) * 1.5
    eda += rng.randn(len(eda)) * 0.05
    eda = np.clip(eda, 0.05, 0.9)
    return pd.Series(hr, index=idx), pd.Series(eda, index=idx)


def ctx_cfg():
    return ContextConfig(dt_minutes=3, alpha=1.25, beta=0.3, mmax=2.5, mmin=0.7,
                         hr_rest=HR_REST_PATIENT, hr_max=180.0, ema_half_life_min=8.0,
                         stress_hr_supp_pow=2.0, stress_hr_off_threshold=0.75,
                         use_asymmetric_kinetics=True, tau_onset_min=8.0, tau_offset_min=40.0)


def jitter_meals(base, seed, max_min=15, max_g=15):
    rng = np.random.RandomState(seed)
    return [(t + timedelta(minutes=int(rng.randint(-max_min, max_min+1))),
             max(0, g + int(rng.randint(-max_g, max_g+1)))) for t, g in base]


def make_profile():
    return {
        "current_basal": 1.0, "sens": 50.0, "min_bg": 90.0, "max_bg": 120.0,
        "max_iob": 4.0, "bolus_increment": 0.05, "maxSMBBasalMinutes": 30, "SMBInterval": 3,
        "enableSMB_always": False, "enableSMB_with_COB": True, "enableSMB_high_bg": True,
        "enableSMB_after_carbs": True, "enableSMB_with_temptarget": False,
        "enableSMB_high_bg_target": 140, "temp_basal_duration_min": 30, "max_safe_basal": 3.0,
    }


def run_arm(cohort, seed, mode, hr, eda, meals, start):
    patient = T1DPatient.withName(cohort, seed=1)
    sensor = CGMSensor.withName("Dexcom", seed=1)
    pump = InsulinPump.withName("Insulet")
    scenario = CustomScenario(start_time=start, scenario=meals)

    ctx = ContextStream(hr, eda, ctx_cfg(), preprocess=False)
    env = ContextAwareT1DSimEnv(patient, sensor, pump, scenario,
                                 context_stream=ctx, modulate_vm0=True,
                                 modulate_vmx=True, modulate_p2u=True)

    profile = make_profile()
    if mode == "baseline":
        ctrl = ContextAwareOpenAPSController(profile, model_path=None, hr_rest=HR_REST_PATCH)
    elif mode == "patch":
        ctrl = ContextAwareOpenAPSController(profile, model_path=PATCH_PATH,
                                              placebo_mode=False, hr_rest=HR_REST_PATCH)
    elif mode == "placebo":
        ctrl = ContextAwareOpenAPSController(profile, model_path=PATCH_PATH,
                                              placebo_mode=True, hr_rest=HR_REST_PATCH)

    reset = env.reset()
    obs = reset.observation if hasattr(reset, "observation") else reset
    last_meal = 0.0
    rows = []
    for _ in range(480):
        st = env.env.sample_time if hasattr(env, "env") else env.sample_time
        info = {"sample_time": st, "meal": last_meal}
        current_time = env.env.time if hasattr(env, "env") else env.time
        ts = pd.Timestamp(current_time)
        pos = hr.index.searchsorted(ts, side="right")
        idx = hr.index[0] if pos == 0 else hr.index[pos-1]
        info["context_hr"] = float(hr.loc[idx])
        info["context_eda"] = float(eda.loc[idx]) * 100.0
        info["context_eda_is_stress_level"] = True
        info["context_hr_is_relative"] = False

        action = ctrl.policy(obs, reward=0.0, done=False, **info)
        step = env.step(action)
        obs = step.observation
        last_meal = step.info.get("meal", 0.0)
        rows.append({
            "time": current_time,
            "glucose": float(obs.CGM if hasattr(obs, "CGM") else obs),
            "basal": step.info.get("action_basal", action.basal),
            "bolus": step.info.get("action_bolus", action.bolus),
        })
    df = pd.DataFrame(rows).set_index("time")
    df["insulin"] = df["basal"] + df["bolus"]
    return df


def compute_metrics(df):
    g = df["glucose"]
    mask_exer = (df.index.hour >= 9) & (df.index.hour < 11)
    mask_cool = (df.index.hour >= 11) & (df.index.hour < 13)
    mask_stress = (df.index.hour >= 14) & (df.index.hour < 15)
    return dict(
        mean=g.mean(), std=g.std(), min=g.min(), max=g.max(),
        tir=((g >= 70) & (g <= 180)).mean() * 100,
        tbr_70=(g < 70).mean() * 100,
        tbr_54=(g < 54).mean() * 100,
        tar_180=(g > 180).mean() * 100,
        tar_250=(g > 250).mean() * 100,
        total_insulin=df["insulin"].sum() * 3.0,
        insulin_exercise=df.loc[mask_exer, "insulin"].sum() * 3.0,
        insulin_cooldown=df.loc[mask_cool, "insulin"].sum() * 3.0,
        insulin_stress=df.loc[mask_stress, "insulin"].sum() * 3.0,
        glucose_exercise_mean=df.loc[mask_exer, "glucose"].mean(),
        glucose_cooldown_mean=df.loc[mask_cool, "glucose"].mean(),
        glucose_stress_mean=df.loc[mask_stress, "glucose"].mean(),
        glucose_min_exercise=df.loc[mask_exer, "glucose"].min(),
        glucose_min_cooldown=df.loc[mask_cool, "glucose"].min(),
    )


def main():
    base_meals = [
        (datetime(2025, 1, 1, 7, 30), 50),
        (datetime(2025, 1, 1, 12, 0), 70),
        (datetime(2025, 1, 1, 18, 0), 60),
        (datetime(2025, 1, 1, 10, 45), 15),
        (datetime(2025, 1, 1, 21, 15), 12),
    ]
    start = datetime(2025, 1, 1, 0, 0)

    all_rows = []
    n_total = len(COHORTS) * len(SEEDS) * 3
    done = 0
    for cohort in COHORTS:
        for seed in SEEDS:
            hr, eda = build_hr_eda(start, seed)
            meals = jitter_meals(base_meals, seed)
            for mode in ("baseline", "patch", "placebo"):
                try:
                    df = run_arm(cohort, seed, mode, hr, eda, meals, start)
                    m = compute_metrics(df)
                    row = dict(cohort=cohort, seed=seed, mode=mode, **m)
                    all_rows.append(row)
                    done += 1
                    print(f"[{done:3d}/{n_total}] {cohort:>18} seed={seed:>2} {mode:>8} | "
                          f"TIR={m['tir']:>5.1f}  TBR<70={m['tbr_70']:>4.1f}  "
                          f"min={m['min']:>5.1f}  max={m['max']:>5.1f}  "
                          f"ins_exer={m['insulin_exercise']:>4.2f}")
                except Exception as e:
                    print(f"[FAIL] {cohort} seed={seed} {mode}: {e}")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True, parents=True)
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(out_dir / "batch_2307_metrics.csv", index=False)
    print(f"\nSaved: {out_dir/'batch_2307_metrics.csv'}")

    print_analysis(df_all, out_dir)


def print_analysis(df_all, out_dir):
    # Pivot: per (cohort, seed) per mode
    print("\n" + "=" * 90)
    print("BATCH SUMMARY — mean metrics per arm across all (cohort, seed) pairs")
    print("=" * 90)
    for mode in ("baseline", "patch", "placebo"):
        sub = df_all[df_all["mode"] == mode]
        print(f"\n{mode:>9}: n={len(sub)}  "
              f"TIR={sub['tir'].mean():.1f}±{sub['tir'].std():.1f}  "
              f"TBR<70={sub['tbr_70'].mean():.2f}±{sub['tbr_70'].std():.2f}  "
              f"TAR>180={sub['tar_180'].mean():.1f}±{sub['tar_180'].std():.1f}  "
              f"min_glu={sub['min'].mean():.1f}  max_glu={sub['max'].mean():.1f}")

    # Pair differences: patch vs baseline within same (cohort, seed)
    print("\n" + "=" * 90)
    print("PATCH vs BASELINE: paired differences (patch - baseline)")
    print("=" * 90)

    pairs = []
    for (cohort, seed), grp in df_all.groupby(["cohort", "seed"]):
        if set(grp["mode"]) >= {"baseline", "patch"}:
            a = grp[grp["mode"] == "baseline"].iloc[0]
            b = grp[grp["mode"] == "patch"].iloc[0]
            pairs.append({
                "cohort": cohort, "seed": seed,
                "tir_diff": b["tir"] - a["tir"],
                "tbr_diff": b["tbr_70"] - a["tbr_70"],
                "tar_diff": b["tar_180"] - a["tar_180"],
                "min_diff": b["min"] - a["min"],
                "max_diff": b["max"] - a["max"],
                "ins_exer_diff": b["insulin_exercise"] - a["insulin_exercise"],
                "ins_cool_diff": b["insulin_cooldown"] - a["insulin_cooldown"],
                "ins_stress_diff": b["insulin_stress"] - a["insulin_stress"],
                "baseline_tbr": a["tbr_70"], "baseline_tar": a["tar_180"],
                "baseline_min": a["min"], "baseline_max": a["max"],
            })
    df_pair = pd.DataFrame(pairs)
    df_pair.to_csv(out_dir / "batch_2307_pairwise_diff.csv", index=False)

    print(f"\nPaired summary (n={len(df_pair)}):")
    print(f"  TIR diff:     mean={df_pair['tir_diff'].mean():+.2f}  "
          f"std={df_pair['tir_diff'].std():.2f}  "
          f"#positive={(df_pair['tir_diff']>0).sum()}/{len(df_pair)}")
    print(f"  TBR<70 diff:  mean={df_pair['tbr_diff'].mean():+.3f}  "
          f"(negative = patch prevents hypo)")
    print(f"  TAR>180 diff: mean={df_pair['tar_diff'].mean():+.3f}")
    print(f"  Insulin during exercise (B-A): mean={df_pair['ins_exer_diff'].mean():+.3f} U  "
          f"std={df_pair['ins_exer_diff'].std():.3f}")
    print(f"  Insulin during cooldown (B-A): mean={df_pair['ins_cool_diff'].mean():+.3f} U")
    print(f"  Insulin during stress (B-A):   mean={df_pair['ins_stress_diff'].mean():+.3f} U")

    # Highlight challenging cases
    print("\n" + "=" * 90)
    print("CHALLENGING CASES — where baseline had hypo (TBR<70 > 0)")
    print("=" * 90)
    hypo_cases = df_pair[df_pair["baseline_tbr"] > 0].sort_values("baseline_tbr", ascending=False)
    if len(hypo_cases) == 0:
        print("  No runs with baseline hypo events. Scenario may be too easy.")
    else:
        print(f"{'cohort':>18} {'seed':>5} {'base_TBR':>10} {'patch_TBR':>11} "
              f"{'Δtbr':>8} {'base_min':>9} {'patch_min':>10} {'Δmin':>7}")
        for _, r in hypo_cases.iterrows():
            patch_tbr = r["baseline_tbr"] + r["tbr_diff"]
            patch_min = r["baseline_min"] + r["min_diff"]
            print(f"{r['cohort']:>18} {int(r['seed']):>5} {r['baseline_tbr']:>9.2f}% "
                  f"{patch_tbr:>10.2f}% {r['tbr_diff']:>+7.2f} "
                  f"{r['baseline_min']:>9.1f} {patch_min:>10.1f} {r['min_diff']:>+6.1f}")

    print("\n" + "=" * 90)
    print("CHALLENGING CASES — where baseline had hyper (TAR>180 > 0)")
    print("=" * 90)
    hyper_cases = df_pair[df_pair["baseline_tar"] > 0].sort_values("baseline_tar", ascending=False)
    if len(hyper_cases) == 0:
        print("  No runs with baseline hyper events.")
    else:
        print(f"{'cohort':>18} {'seed':>5} {'base_TAR':>10} {'patch_TAR':>11} "
              f"{'Δtar':>8} {'base_max':>9} {'patch_max':>10} {'Δmax':>7}")
        for _, r in hyper_cases.iterrows():
            patch_tar = r["baseline_tar"] + r["tar_diff"]
            patch_max = r["baseline_max"] + r["max_diff"]
            print(f"{r['cohort']:>18} {int(r['seed']):>5} {r['baseline_tar']:>9.2f}% "
                  f"{patch_tar:>10.2f}% {r['tar_diff']:>+7.2f} "
                  f"{r['baseline_max']:>9.1f} {patch_max:>10.1f} {r['min_diff']:>+6.1f}")


if __name__ == "__main__":
    main()
