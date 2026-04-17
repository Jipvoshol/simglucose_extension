"""
Closed-loop trial v2: tests the 2307 XGBoost ISF patch under realistic OpenAPS setup.

Combines:
- Rich HR/EDA scenario from generate_batch_plots.py (circadian + exercise +
  cooldown + stress + evening walk + micro-events, jittered meals)
- Full OpenAPS oref0 profile with SMB flags enabled
- 3-arm comparison from thesis_closed_loop_trial.py (blind / patch / placebo)
- Rich 6-panel plot format showing basal-per-timestep and SMB events

This lets us see WHERE the controller reacts differently, not only how
glucose looks in the end.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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


COHORT = "adolescent#001"
SEED = 3
HR_REST_PATIENT = 68.0  # adolescent#001 physiological baseline (matches
                        # ContextStream config in generate_batch_plots.py)
HR_REST_PATCH = 68.0    # HR_REST passed to the XGBoost patch controller.
                        # 2307's actual resting HR was 81.5 bpm. We use 68 here
                        # so patch input HR_norm matches the simulated patient's
                        # relative HR range. The patch then sees the same
                        # HR_norm distribution it was trained on, even though
                        # the absolute HR values differ.
PATCH_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../manchester/F9_cap25_patch/patch/isf_patch_2307.pkl"))


# ---------------------------------------------------------------------------
# Scenario builders (adapted from generate_batch_plots.py)
# ---------------------------------------------------------------------------
def build_hr_eda_series(start_time: datetime, seed: int):
    idx = pd.date_range(start_time, periods=480, freq="3min")
    rng = np.random.RandomState(seed)
    t = np.arange(480)
    circ = 68 + 6 * np.sin(2 * np.pi * t / 480 - 1.2)
    hr = circ + rng.randn(480) * 3.0

    # Exercise 9-11h (2h intensive) + cooldown
    hr[180:220] += 75
    hr[220:260] += 25
    # Stress 14-15h
    hr[280:300] += 18
    # Evening walk 18:30-19:30
    hr[360:390] += 25

    eda = np.full(480, 0.12)
    eda[280:300] = 0.7
    eda[220:240] = 0.45
    eda[360:390] = 0.2

    def add_micro(center, hr_amp, eda_amp, width):
        s, e = max(0, center - width), min(len(hr), center + width)
        hr[s:e] += hr_amp
        eda[s:e] += max(0.0, eda_amp)

    for _ in range(8):
        c = rng.randint(12, len(hr) - 12)
        add_micro(c, rng.uniform(8, 22), rng.uniform(0.0, 0.25), rng.randint(2, 6))
    for _ in range(4):
        c = rng.randint(12, len(hr) - 12)
        add_micro(c, -rng.uniform(5, 12), 0.0, rng.randint(3, 7))

    hr += rng.randn(len(hr)) * 1.5
    eda += rng.randn(len(eda)) * 0.05
    eda = np.clip(eda, 0.05, 0.9)

    return pd.Series(hr, index=idx), pd.Series(eda, index=idx)


def context_cfg() -> ContextConfig:
    """ContextStream config for the virtual patient's physiology."""
    return ContextConfig(
        dt_minutes=3,
        alpha=1.25,
        beta=0.3,
        mmax=2.5,
        mmin=0.7,
        hr_rest=HR_REST_PATIENT,
        hr_max=180.0,
        ema_half_life_min=8.0,
        stress_hr_supp_pow=2.0,
        stress_hr_off_threshold=0.75,
        use_asymmetric_kinetics=True,
        tau_onset_min=8.0,
        tau_offset_min=40.0,
    )


def jitter_meals(base, seed, max_min=15, max_g=15):
    rng = np.random.RandomState(seed)
    return [
        (t + timedelta(minutes=int(rng.randint(-max_min, max_min + 1))),
         max(0, g + int(rng.randint(-max_g, max_g + 1))))
        for t, g in base
    ]


def make_profile() -> dict:
    """OpenAPS profile with SMB flags — from generate_batch_plots.py."""
    return {
        "current_basal": 1.0,
        "sens": 50.0,
        "min_bg": 90.0,
        "max_bg": 120.0,
        "max_iob": 4.0,
        "bolus_increment": 0.05,
        "maxSMBBasalMinutes": 30,
        "SMBInterval": 3,
        "enableSMB_always": False,
        "enableSMB_with_COB": True,
        "enableSMB_high_bg": True,
        "enableSMB_after_carbs": True,
        "enableSMB_with_temptarget": False,
        "enableSMB_high_bg_target": 140,
        "temp_basal_duration_min": 30,
        "max_safe_basal": 3.0,
    }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def run_arm(arm_label: str, controller_mode: str, hr: pd.Series, eda: pd.Series,
            meals: list, start_time: datetime):
    print(f"[{arm_label}] mode={controller_mode} ...")

    patient = T1DPatient.withName(COHORT, seed=1)
    sensor = CGMSensor.withName("Dexcom", seed=1)
    pump = InsulinPump.withName("Insulet")

    scenario = CustomScenario(start_time=start_time, scenario=meals)

    # Patient-side physiology always context-aware (we want exercise to
    # actually affect the virtual patient, regardless of controller choice)
    ctx = ContextStream(hr, eda, context_cfg(), preprocess=False)
    env = ContextAwareT1DSimEnv(
        patient, sensor, pump, scenario,
        context_stream=ctx, modulate_vm0=True, modulate_vmx=True, modulate_p2u=True,
    )

    profile = make_profile()
    if controller_mode == "baseline":
        ctrl = ContextAwareOpenAPSController(profile, model_path=None, hr_rest=HR_REST_PATCH)
    elif controller_mode == "patch":
        ctrl = ContextAwareOpenAPSController(profile, model_path=PATCH_PATH,
                                              placebo_mode=False, hr_rest=HR_REST_PATCH)
    elif controller_mode == "placebo":
        ctrl = ContextAwareOpenAPSController(profile, model_path=PATCH_PATH,
                                              placebo_mode=True, hr_rest=HR_REST_PATCH)
    else:
        raise ValueError(controller_mode)

    reset = env.reset()
    obs = reset.observation if hasattr(reset, "observation") else reset
    last_meal = 0.0
    rows = []

    for _ in range(480):
        st = env.env.sample_time if hasattr(env, "env") else env.sample_time
        info = {"sample_time": st, "meal": last_meal}

        # Pass context to patch controller
        current_time = env.env.time if hasattr(env, "env") else env.time
        ts = pd.Timestamp(current_time)
        pos = hr.index.searchsorted(ts, side="right")
        idx = hr.index[0] if pos == 0 else hr.index[pos - 1]
        info["context_hr"] = float(hr.loc[idx])
        # ContextStream's EDA is in [0, 1] (normalized stress proxy).
        # Convert to 0-100 stress_level scale for the patch's EDA_norm feature.
        info["context_eda"] = float(eda.loc[idx]) * 100.0
        info["context_eda_is_stress_level"] = True
        info["context_hr_is_relative"] = False

        action = ctrl.policy(obs, reward=0.0, done=False, **info)
        step = env.step(action)
        obs = step.observation
        last_meal = step.info.get("meal", 0.0)
        m_t_val = getattr(ctrl, "last_m_t", 1.0) if controller_mode != "baseline" else 1.0

        rows.append({
            "time": env.env.time if hasattr(env, "env") else env.time,
            "glucose": float(obs.CGM if hasattr(obs, "CGM") else obs),
            "basal": step.info.get("action_basal", action.basal),
            "bolus": step.info.get("action_bolus", action.bolus),
            "m_t": m_t_val,
            "m_t_physio": step.info.get("context_m", 1.0),   # ContextStream's m(t) for patient
            "hr": info["context_hr"],
            "eda": eda.loc[idx],
        })

    df = pd.DataFrame(rows).set_index("time")
    df["insulin"] = df["basal"] + df["bolus"]
    # IOB (exp decay, DIA=6h)
    lam = np.exp(-3.0 / (360.0 / 3.0))
    iob = []
    r = 0.0
    for x in df["insulin"].to_numpy(dtype=float):
        r = r * lam + x * 3.0
        iob.append(r)
    df["IOB"] = iob
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(df):
    g = df["glucose"]
    tir = ((g >= 70) & (g <= 180)).mean() * 100
    tbr_70 = (g < 70).mean() * 100
    tbr_54 = (g < 54).mean() * 100
    tar_180 = (g > 180).mean() * 100
    tar_250 = (g > 250).mean() * 100
    ins = df["insulin"].sum() * 3.0
    return dict(mean=g.mean(), std=g.std(), min=g.min(), max=g.max(),
                tir=tir, tbr_70=tbr_70, tbr_54=tbr_54, tar_180=tar_180, tar_250=tar_250,
                total_insulin=ins)


# ---------------------------------------------------------------------------
# Plot (rich format, adapted from generate_batch_plots.py)
# ---------------------------------------------------------------------------
def plot_three_arm(df_A, df_B, df_C, meals, out_path: Path, title_extra=""):
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(7, 1, height_ratios=[2.5, 1.1, 1.0, 1.2, 1.2, 1.0, 1.0], hspace=0.32)

    t = df_A.index

    # Panel 1: Glucose
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, df_A["glucose"], "gray", linestyle="--", label="Arm A (blind)", linewidth=2.0, alpha=0.8)
    ax1.plot(t, df_B["glucose"], "#2E8B57", label="Arm B (2307 patch)", linewidth=2.6, alpha=0.95)
    ax1.plot(t, df_C["glucose"], "black", linestyle=":", label="Arm C (placebo)", linewidth=1.6, alpha=0.6)
    ax1.axhspan(70, 180, alpha=0.12, color="green", label="Target 70-180")
    ax1.axhline(70, color="red", linestyle="--", alpha=0.4)
    ax1.axhline(180, color="red", linestyle="--", alpha=0.4)
    for mt, mg in meals:
        ax1.axvline(mt, color="purple", linestyle=":", alpha=0.5, linewidth=1.3)
        ax1.text(mt, 288, f"{mg}g", ha="center", fontsize=8, color="purple", fontweight="bold")
    # Highlight exercise / stress blocks
    ax1.axvspan(t[180], t[260], alpha=0.10, color="orange", label="Exercise+cooldown")
    ax1.axvspan(t[280], t[300], alpha=0.10, color="red", label="Stress")
    ax1.axvspan(t[360], t[390], alpha=0.08, color="yellow", label="Evening walk")
    ax1.set_ylabel("Glucose (mg/dL)", fontweight="bold")
    ax1.set_ylim(40, 300)
    ax1.legend(loc="upper right", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"2307 patch closed-loop trial — {COHORT} seed={SEED}{title_extra}",
                  fontsize=13, fontweight="bold")

    # Panel 2: Patch m(t) vs placebo m(t)
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(t, df_B["m_t"], "#2E8B57", label="Patch m(t) (XGBoost 2307)", linewidth=2.4)
    ax2.plot(t, df_C["m_t"], "black", linestyle=":", label="Placebo m(t) (random)", alpha=0.6)
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.fill_between(t, 1.0, df_B["m_t"], where=(df_B["m_t"] > 1.0), alpha=0.25, color="orange")
    ax2.fill_between(t, 1.0, df_B["m_t"], where=(df_B["m_t"] < 1.0), alpha=0.25, color="red")
    ax2.set_ylabel("Patch m(t)\n(ISF multiplier)", fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.5, 2.6)

    # Panel 3: Physiology m(t) from ContextStream (same for all 3 arms)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(t, df_A["m_t_physio"], "purple", linewidth=2.0, label="ContextStream m(t) (patient physio)")
    ax3.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax3.set_ylabel("Physio m(t)\n(same all arms)", fontweight="bold")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: HR + EDA
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4_eda = ax4.twinx()
    ax4.plot(t, df_B["hr"], "blue", linewidth=1.3, alpha=0.75, label="Heart Rate")
    ax4_eda.plot(t, df_B["eda"], "red", linewidth=1.3, alpha=0.6, label="EDA (stress)")
    ax4.axhline(HR_REST_PATIENT, color="gray", linestyle="--", alpha=0.4)
    ax4.set_ylabel("HR (bpm)", fontweight="bold", color="blue")
    ax4_eda.set_ylabel("EDA (norm.)", fontweight="bold", color="red")
    ax4.tick_params(axis="y", labelcolor="blue")
    ax4_eda.tick_params(axis="y", labelcolor="red")
    ax4.legend(loc="upper left", fontsize=8)
    ax4_eda.legend(loc="upper right", fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Panel 5: IOB
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    ax5.plot(t, df_A["IOB"], "gray", linestyle="--", linewidth=1.7, alpha=0.7, label="A IOB")
    ax5.plot(t, df_B["IOB"], "#2E8B57", linewidth=2.0, alpha=0.9, label="B IOB")
    ax5.plot(t, df_C["IOB"], "black", linestyle=":", linewidth=1.3, alpha=0.6, label="C IOB")
    ax5.set_ylabel("IOB (U)", fontweight="bold")
    ax5.legend(loc="upper right", fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Panel 6: Basal per timestep (U/h)
    ax6 = fig.add_subplot(gs[5], sharex=ax1)
    ax6.plot(t, df_A["basal"] * 60, "gray", linestyle="--", linewidth=1.3, alpha=0.7, label="A basal")
    ax6.plot(t, df_B["basal"] * 60, "#2E8B57", linewidth=1.6, alpha=0.9, label="B basal")
    ax6.plot(t, df_C["basal"] * 60, "black", linestyle=":", linewidth=1.0, alpha=0.5, label="C basal")
    ax6.set_ylabel("Basal (U/h)", fontweight="bold")
    ax6.legend(loc="upper right", fontsize=8)
    ax6.grid(True, alpha=0.3)

    # Panel 7: SMB/bolus as dots
    ax7 = fig.add_subplot(gs[6], sharex=ax1)
    for df, color, marker, label in [
        (df_A, "gray", "o", "A SMB"),
        (df_B, "#2E8B57", "s", "B SMB"),
        (df_C, "black", "^", "C SMB"),
    ]:
        boluses = df[df["bolus"] > 0.001]
        if len(boluses):
            ax7.scatter(boluses.index, boluses["bolus"] * 3, color=color, marker=marker,
                        s=35, alpha=0.7, label=label, edgecolors="black", linewidths=0.5)
    ax7.set_ylabel("Bolus (U)", fontweight="bold")
    ax7.set_xlabel("Time")
    ax7.legend(loc="upper right", fontsize=8)
    ax7.grid(True, alpha=0.3)

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"plot saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("2307 PATCH CLOSED-LOOP TRIAL v2 (realistic scenario + 3 arms)")
    print(f"  cohort={COHORT}  seed={SEED}  HR_REST_patch={HR_REST_PATCH}")
    print(f"  patch={PATCH_PATH}")
    print("=" * 72)

    start_time = datetime(2025, 1, 1, 0, 0)
    hr, eda = build_hr_eda_series(start_time, SEED)

    base_meals = [
        (datetime(2025, 1, 1, 7, 30), 50),
        (datetime(2025, 1, 1, 12, 0), 70),
        (datetime(2025, 1, 1, 18, 0), 60),
        (datetime(2025, 1, 1, 10, 45), 15),
        (datetime(2025, 1, 1, 21, 15), 12),
    ]
    meals = jitter_meals(base_meals, SEED)

    df_A = run_arm("Arm A (blind)", "baseline", hr, eda, meals, start_time)
    df_B = run_arm("Arm B (patch)", "patch", hr, eda, meals, start_time)
    df_C = run_arm("Arm C (placebo)", "placebo", hr, eda, meals, start_time)

    # Metrics table
    print("\n" + "=" * 72)
    print("METRICS (full 24h)")
    print("=" * 72)
    mA, mB, mC = metrics(df_A), metrics(df_B), metrics(df_C)
    hdr = f"{'':>20} {'Arm A (blind)':>14} {'Arm B (patch)':>14} {'Arm C (plac)':>14}"
    print(hdr)
    print("-" * len(hdr))
    for k in ["mean", "std", "min", "max", "tir", "tbr_70", "tbr_54", "tar_180", "tar_250", "total_insulin"]:
        print(f"{k:>20} {mA[k]:>14.2f} {mB[k]:>14.2f} {mC[k]:>14.2f}")

    # Per-period glucose
    def period(df, s, e):
        m = (df.index.hour >= s) & (df.index.hour < e)
        return df.loc[m, "glucose"].mean()

    print("\nPER-PERIOD GLUCOSE MEAN (mg/dL):")
    print(f"{'period':>20} {'A':>14} {'B':>14} {'C':>14} {'B-A':>10}")
    for name, s, e in [("exercise 9-11", 9, 11), ("cooldown 11-13", 11, 13),
                       ("stress 14-15", 14, 15), ("evening walk 18-20", 18, 20),
                       ("night 0-6", 0, 6)]:
        gA, gB, gC = period(df_A, s, e), period(df_B, s, e), period(df_C, s, e)
        print(f"{name:>20} {gA:>14.1f} {gB:>14.1f} {gC:>14.1f} {gB-gA:>+10.1f}")

    # Per-period insulin
    def period_ins(df, s, e):
        m = (df.index.hour >= s) & (df.index.hour < e)
        return df.loc[m, "insulin"].sum() * 3.0  # U/min × min = U over window

    print("\nPER-PERIOD TOTAL INSULIN (U):")
    print(f"{'period':>20} {'A':>14} {'B':>14} {'C':>14} {'B-A':>10}")
    for name, s, e in [("exercise 9-11", 9, 11), ("cooldown 11-13", 11, 13),
                       ("stress 14-15", 14, 15), ("evening walk 18-20", 18, 20)]:
        iA, iB, iC = period_ins(df_A, s, e), period_ins(df_B, s, e), period_ins(df_C, s, e)
        print(f"{name:>20} {iA:>14.2f} {iB:>14.2f} {iC:>14.2f} {iB-iA:>+10.2f}")

    # Save CSVs + plot
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True, parents=True)
    df_A.to_csv(out_dir / "closed_loop_2307_v2_arm_A_blind.csv")
    df_B.to_csv(out_dir / "closed_loop_2307_v2_arm_B_patch.csv")
    df_C.to_csv(out_dir / "closed_loop_2307_v2_arm_C_placebo.csv")
    plot_three_arm(df_A, df_B, df_C, meals, out_dir / "closed_loop_2307_v2.png",
                   title_extra=" (rich scenario)")

    print(f"\nCSVs + plot saved to {out_dir}")


if __name__ == "__main__":
    main()
