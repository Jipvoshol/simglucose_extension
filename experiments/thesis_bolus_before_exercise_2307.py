"""
Bolus-before-exercise hypo-risk scenario for the 2307 patch.

Clinical pattern this tests:
  - Patient eats breakfast at 7:30 (50g carbs) → large pre-bolus
  - IOB peaks 1-2h later
  - Patient exercises 9:00-10:30h — during peak IOB
  - WITHOUT patch: OpenAPS is already dosing for the meal, can't anticipate the
    HR spike; glucose drops fast → hypo risk.
  - WITH patch: patch raises m(t) ≈ 1.8 as HR climbs → OpenAPS reduces basal
    aggressively (or suspends) → less net insulin → less hypo.

Runs across multiple (cohort, seed) combinations to see which scenarios
produce hypo-risk and how the patch behaves in each.
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


HR_REST = 68.0
PATCH_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../manchester/F9_cap25_patch/patch/isf_patch_2307.pkl"))

# Test cohorts: focus on adolescent+adult, child models are oversensitive
COHORTS = ["adolescent#001", "adolescent#003", "adolescent#005",
           "adolescent#007", "adult#001", "adult#003"]
SEEDS = [1, 3, 7, 13, 42]


def build_hr_eda_prebolus_exercise(start, seed):
    """
    HR/EDA scenario specifically designed for hypo-risk:
    - Big breakfast at 7:30 → pre-bolus
    - Exercise starts 9:00 (1.5h after meal, peak IOB moment)
    - Runs 1.5h until 10:30
    - Cooldown 10:30-11:30
    - Quiet rest of the day (no stress event)
    """
    idx = pd.date_range(start, periods=480, freq="3min")
    rng = np.random.RandomState(seed)
    t = np.arange(480)
    circ = HR_REST + 5 * np.sin(2 * np.pi * t / 480 - 1.2)
    hr = circ + rng.randn(480) * 2.5

    # Exercise 9:00-10:30h (bins 180-210) — DURING peak IOB
    hr[180:210] += 70   # vigorous run
    hr[210:230] += 30   # cooldown

    # Small micro-events elsewhere (no big stress)
    for _ in range(4):
        c = rng.randint(12, 468)
        if abs(c - 195) < 30:
            continue  # don't add noise during the main exercise event
        w = rng.randint(2, 5)
        hr[max(0, c-w):c+w] += rng.uniform(5, 15)

    hr += rng.randn(len(hr)) * 1.5

    # EDA: low baseline, small bump during exercise (physical stress only)
    eda = np.full(480, 0.12)
    eda[180:210] = 0.35
    eda[210:230] = 0.22
    eda += rng.randn(len(eda)) * 0.03
    eda = np.clip(eda, 0.05, 0.9)

    return pd.Series(hr, index=idx), pd.Series(eda, index=idx)


def meals_prebolus(seed):
    """Big breakfast at 7:30 (before the run at 9:00), normal lunch, dinner."""
    rng = np.random.RandomState(seed)
    # 60g breakfast (larger than v2's 50g → more bolus → more IOB at exercise)
    dt_break = int(rng.randint(-10, 10))
    dt_lunch = int(rng.randint(-15, 15))
    dt_dinner = int(rng.randint(-15, 15))
    return [
        (datetime(2025, 1, 1, 7, 30) + timedelta(minutes=dt_break), 60),
        (datetime(2025, 1, 1, 12, 30) + timedelta(minutes=dt_lunch), 55),
        (datetime(2025, 1, 1, 18, 30) + timedelta(minutes=dt_dinner), 55),
    ]


def ctx_cfg():
    return ContextConfig(dt_minutes=3, alpha=1.25, beta=0.3, mmax=2.5, mmin=0.7,
                         hr_rest=HR_REST, hr_max=180.0, ema_half_life_min=8.0,
                         stress_hr_supp_pow=2.0, stress_hr_off_threshold=0.75,
                         use_asymmetric_kinetics=True, tau_onset_min=8.0, tau_offset_min=40.0)


def profile():
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

    p = profile()
    if mode == "baseline":
        ctrl = ContextAwareOpenAPSController(p, model_path=None, hr_rest=HR_REST)
    elif mode == "patch":
        ctrl = ContextAwareOpenAPSController(p, model_path=PATCH_PATH,
                                              placebo_mode=False, hr_rest=HR_REST)
    elif mode == "placebo":
        ctrl = ContextAwareOpenAPSController(p, model_path=PATCH_PATH,
                                              placebo_mode=True, hr_rest=HR_REST)

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
        idx = hr.index[0] if pos == 0 else hr.index[pos - 1]
        info["context_hr"] = float(hr.loc[idx])
        info["context_eda"] = float(eda.loc[idx]) * 100.0
        info["context_eda_is_stress_level"] = True
        info["context_hr_is_relative"] = False

        action = ctrl.policy(obs, reward=0.0, done=False, **info)
        step = env.step(action)
        obs = step.observation
        last_meal = step.info.get("meal", 0.0)
        m_t_val = getattr(ctrl, "last_m_t", 1.0) if mode != "baseline" else 1.0
        rows.append({
            "time": current_time,
            "glucose": float(obs.CGM if hasattr(obs, "CGM") else obs),
            "basal": step.info.get("action_basal", action.basal),
            "bolus": step.info.get("action_bolus", action.bolus),
            "m_t": m_t_val,
            "m_t_physio": step.info.get("context_m", 1.0),
            "hr": info["context_hr"],
            "eda": eda.loc[idx],
        })
    df = pd.DataFrame(rows).set_index("time")
    df["insulin"] = df["basal"] + df["bolus"]
    lam = np.exp(-3.0 / (360.0 / 3.0))
    iob = []
    r = 0.0
    for x in df["insulin"].to_numpy(dtype=float):
        r = r * lam + x * 3.0
        iob.append(r)
    df["IOB"] = iob
    return df


def metrics(df):
    g = df["glucose"]
    # Critical window: 9:00-12:00 (exercise + cooldown + peak IOB)
    crit = df.between_time("09:00", "12:00")["glucose"]
    return dict(
        tir=((g >= 70) & (g <= 180)).mean() * 100,
        tbr_70=(g < 70).mean() * 100,
        tbr_54=(g < 54).mean() * 100,
        tar_180=(g > 180).mean() * 100,
        min_glu=g.min(),
        max_glu=g.max(),
        mean_glu=g.mean(),
        crit_min=crit.min() if len(crit) else np.nan,
        crit_tbr=(crit < 70).mean() * 100 if len(crit) else np.nan,
        crit_mean=crit.mean() if len(crit) else np.nan,
        total_insulin=df["insulin"].sum() * 3.0,
        ins_preexer=df.between_time("07:30", "09:00")["insulin"].sum() * 3.0,
        ins_exer=df.between_time("09:00", "11:00")["insulin"].sum() * 3.0,
    )


def main():
    start = datetime(2025, 1, 1, 0, 0)
    rows = []
    n_total = len(COHORTS) * len(SEEDS) * 3
    done = 0
    for cohort in COHORTS:
        for seed in SEEDS:
            hr, eda = build_hr_eda_prebolus_exercise(start, seed)
            meals = meals_prebolus(seed)
            for mode in ("baseline", "patch", "placebo"):
                try:
                    df = run_arm(cohort, seed, mode, hr, eda, meals, start)
                    m = metrics(df)
                    rows.append(dict(cohort=cohort, seed=seed, mode=mode, **m))
                    done += 1
                    print(f"[{done:3d}/{n_total}] {cohort:>18} seed={seed:>2} {mode:>8} | "
                          f"TIR={m['tir']:>5.1f}  TBR={m['tbr_70']:>4.1f}  "
                          f"min={m['min_glu']:>5.1f}  crit_min={m['crit_min']:>5.1f}  "
                          f"crit_TBR={m['crit_tbr']:>4.1f}  ins_exer={m['ins_exer']:.2f}")
                except Exception as e:
                    print(f"FAIL {cohort} seed={seed} {mode}: {e}")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True, parents=True)
    df_all = pd.DataFrame(rows)
    df_all.to_csv(out_dir / "bolus_before_exercise_metrics.csv", index=False)
    print(f"\nSaved: {out_dir / 'bolus_before_exercise_metrics.csv'}")

    print_analysis(df_all, out_dir)


def print_analysis(df_all, out_dir):
    print("\n" + "=" * 90)
    print("BOLUS-BEFORE-EXERCISE SCENARIO — mean metrics per arm")
    print("=" * 90)
    for mode in ("baseline", "patch", "placebo"):
        sub = df_all[df_all["mode"] == mode]
        print(f"  {mode:>9}: n={len(sub)}  "
              f"TIR={sub['tir'].mean():.1f}  TBR<70={sub['tbr_70'].mean():.2f}  "
              f"TAR>180={sub['tar_180'].mean():.1f}  min={sub['min_glu'].mean():.1f}  "
              f"crit_min={sub['crit_min'].mean():.1f}  crit_TBR={sub['crit_tbr'].mean():.2f}")

    # Pairwise patch vs baseline
    pairs = []
    for (cohort, seed), grp in df_all.groupby(["cohort", "seed"]):
        if set(grp["mode"]) >= {"baseline", "patch"}:
            a = grp[grp["mode"] == "baseline"].iloc[0]
            b = grp[grp["mode"] == "patch"].iloc[0]
            c = grp[grp["mode"] == "placebo"].iloc[0] if "placebo" in grp["mode"].values else None
            pairs.append({
                "cohort": cohort, "seed": seed,
                "base_tbr": a["tbr_70"], "patch_tbr": b["tbr_70"],
                "tbr_diff_patch": b["tbr_70"] - a["tbr_70"],
                "tbr_diff_placebo": (c["tbr_70"] - a["tbr_70"]) if c is not None else np.nan,
                "crit_tbr_base": a["crit_tbr"], "crit_tbr_patch": b["crit_tbr"],
                "crit_tbr_diff_patch": b["crit_tbr"] - a["crit_tbr"],
                "crit_min_base": a["crit_min"], "crit_min_patch": b["crit_min"],
                "ins_exer_diff": b["ins_exer"] - a["ins_exer"],
            })
    df_pair = pd.DataFrame(pairs)
    df_pair.to_csv(out_dir / "bolus_before_exercise_pairwise.csv", index=False)

    print("\n" + "=" * 90)
    print(f"PAIRED (patch − baseline) n={len(df_pair)}:")
    print("=" * 90)
    print(f"  TBR<70 change (full day):    mean={df_pair['tbr_diff_patch'].mean():+.2f}  "
          f"std={df_pair['tbr_diff_patch'].std():.2f}  "
          f"#reduced={(df_pair['tbr_diff_patch']<0).sum()}/{len(df_pair)}")
    print(f"  crit_TBR change (9-12h):     mean={df_pair['crit_tbr_diff_patch'].mean():+.2f}  "
          f"#reduced={(df_pair['crit_tbr_diff_patch']<0).sum()}/{len(df_pair)}")
    print(f"  Insulin during exercise (B-A): mean={df_pair['ins_exer_diff'].mean():+.3f} U")
    print(f"  Placebo TBR change (control):  mean={df_pair['tbr_diff_placebo'].mean():+.2f}")

    # Which cases had baseline hypo?
    print("\n" + "=" * 90)
    print("CASES WITH BASELINE HYPO (crit_TBR > 0 in exercise window 9-12h):")
    print("=" * 90)
    hypo = df_pair[df_pair["crit_tbr_base"] > 0].sort_values("crit_tbr_base", ascending=False)
    if len(hypo) == 0:
        print("  No exercise-window hypo events in baseline. Scenario not aggressive enough.")
    else:
        print(f"{'cohort':>18} {'seed':>5} {'base_crTBR':>11} {'patch_crTBR':>12} "
              f"{'Δ':>6} {'min_A':>7} {'min_B':>7} {'ins_exer_Δ':>11}")
        for _, r in hypo.iterrows():
            print(f"{r['cohort']:>18} {int(r['seed']):>5} "
                  f"{r['crit_tbr_base']:>10.2f}% {r['crit_tbr_patch']:>11.2f}% "
                  f"{r['crit_tbr_diff_patch']:>+6.2f} "
                  f"{r['crit_min_base']:>7.1f} {r['crit_min_patch']:>7.1f} "
                  f"{r['ins_exer_diff']:>+11.3f}")


if __name__ == "__main__":
    main()
