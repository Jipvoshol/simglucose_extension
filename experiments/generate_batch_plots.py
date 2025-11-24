"""
Final Batch Plots: Baseline vs Context-Aware (with delayed response)

Generates two versions per cohort/seed:
1. Standard: Glucose, m(t), HR/EDA, IOB, Basal, Bolus
2. Extended: + Delta Glucose + Delta Insulin panels for analysis

Author: Jip Voshol
Date: October 29, 2025
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

from simglucose_ctx.openaps_controller import OpenAPSController


def build_hr_eda_series(start_time: datetime, seed: int):
    """HR/EDA timeseries with circadian rhythm + exercise/stress events."""
    idx = pd.date_range(start_time, periods=480, freq="3min")
    rng = np.random.RandomState(seed)
    t = np.arange(480)
    circ = 68 + 6 * np.sin(2 * np.pi * t / 480 - 1.2)
    hr = circ + rng.randn(480) * 3.0

    # Exercise 9-11h (2 hours intensive) + cooldown
    hr[180:220] += 75  # intensive block
    hr[220:260] += 25  # cooldown

    # Stress 14-15h
    hr[280:300] += 18

    # Evening walk 18:30-19:30
    hr[360:390] += 25

    eda = np.full(480, 0.12)
    eda[280:300] = 0.7
    eda[220:240] = 0.45
    eda[360:390] = 0.2

    def add_micro_event(center: int, hr_amp: float, eda_amp: float, width: int):
        start = max(0, center - width)
        end = min(len(hr), center + width)
        hr[start:end] += hr_amp
        eda[start:end] += max(0.0, eda_amp)

    # Micro-events: sprint to train, stairs, etc.
    for _ in range(8):
        center = rng.randint(12, len(hr) - 12)
        hr_amp = rng.uniform(8, 22)
        eda_amp = rng.uniform(0.0, 0.25)
        width = rng.randint(2, 6)
        add_micro_event(center, hr_amp, eda_amp, width)

    # Smaller counterparts (rest moments) with lower HR
    for _ in range(4):
        center = rng.randint(12, len(hr) - 12)
        hr_amp = -rng.uniform(5, 12)
        width = rng.randint(3, 7)
        add_micro_event(center, hr_amp, 0.0, width)

    # Extra jitter
    hr += rng.randn(len(hr)) * 1.5
    eda += rng.randn(len(eda)) * 0.05

    eda = np.clip(eda, 0.05, 0.9)

    return pd.Series(hr, index=idx), pd.Series(eda, index=idx)


def context_cfg() -> ContextConfig:
    """Config for context-aware.

    CRITICAL: These values are calibrated for the K=3.5 exponent in sim_hook.py!
    alpha=1.25 ensures theoretical m_max = 2.25, which gives 17x Vm0 boost.
    """
    return ContextConfig(
        dt_minutes=3,
        alpha=1.25,  # MUST be 1.25 for K=3.5 calibration!
        beta=0.3,  # Stress gain
        mmax=2.5,  # Cap high enough for theoretical max
        mmin=0.7,
        hr_rest=68.0,
        hr_max=180.0,
        ema_half_life_min=8.0,
        stress_hr_supp_pow=2.0,
        stress_hr_off_threshold=0.75,
        use_asymmetric_kinetics=True,
        tau_onset_min=8.0,
        tau_offset_min=40.0,
    )


def jitter_meals(
    base_meals: List[Tuple[datetime, int]], seed: int, max_min: int = 15, max_g: int = 15
):
    rng = np.random.RandomState(seed)
    res = []
    for t, g in base_meals:
        dt = int(rng.randint(-max_min, max_min + 1))
        dg = int(rng.randint(-max_g, max_g + 1))
        res.append((t + timedelta(minutes=dt), max(0, g + dg)))
    return res


def run_simulation(
    cohort: str, seed: int, use_context: bool
) -> Tuple[pd.DataFrame, List[Tuple[datetime, int]]]:
    """Run simulation and return DataFrame."""
    start = datetime(2025, 1, 1, 0, 0)
    base_meals = [
        (datetime(2025, 1, 1, 7, 30), 50),
        (datetime(2025, 1, 1, 12, 0), 70),
        (datetime(2025, 1, 1, 18, 0), 60),
        (datetime(2025, 1, 1, 10, 45), 15),
        (datetime(2025, 1, 1, 21, 15), 12),
    ]
    meals = jitter_meals(base_meals, seed)
    scenario = CustomScenario(start_time=start, scenario=meals)

    hr, eda = build_hr_eda_series(start, seed)

    if use_context:
        cfg = context_cfg()
        ctx = ContextStream(hr, eda, cfg, preprocess=False)
    else:
        ctx = None

    patient = T1DPatient.withName(cohort, seed=1)
    sensor = CGMSensor.withName("Dexcom", seed=1)
    pump = InsulinPump.withName("Insulet")

    if ctx is not None:
        env = ContextAwareT1DSimEnv(
            patient,
            sensor,
            pump,
            scenario,
            context_stream=ctx,
            modulate_vm0=True,
            modulate_vmx=True,
            modulate_p2u=True,
        )
    else:
        env = ContextAwareT1DSimEnv(patient, sensor, pump, scenario, context_stream=None)

    profile = {
        "current_basal": 1.0,
        "sens": 50.0,
        "min_bg": 90.0,
        "max_bg": 120.0,
        "max_iob": 4.0,
        "bolus_increment": 0.05,
        "maxSMBBasalMinutes": 30,
        "SMBInterval": 3,
        # Oref0-style: temp basal + SMB triggers (COB/high BG/after carbs)
        "enableSMB_always": False,
        "enableSMB_with_COB": True,
        "enableSMB_high_bg": True,
        "enableSMB_after_carbs": True,
        "enableSMB_with_temptarget": False,
        "enableSMB_high_bg_target": 140,
        "temp_basal_duration_min": 30,
        "max_safe_basal": 3.0,
    }
    ctrl = OpenAPSController(profile)

    reset = env.reset()
    obs = reset.observation if hasattr(reset, "observation") else reset
    rows = []
    last_meal = 0.0

    for _ in range(480):
        st = env.env.sample_time if hasattr(env, "env") else env.sample_time
        act = ctrl.policy(obs, reward=0.0, done=False, sample_time=st, meal=last_meal)
        step = env.step(act)
        obs = step.observation
        info = step.info
        last_meal = info.get("meal", 0.0)
        rows.append(
            {
                "time": env.env.time if hasattr(env, "env") else env.time,
                "glucose": float(obs.CGM if hasattr(obs, "CGM") else obs),
                "basal": info.get("action_basal", act.basal),
                "bolus": info.get("action_bolus", act.bolus),
                "insulin": info.get("action_basal", act.basal)
                + info.get("action_bolus", act.bolus),
                "m_t": info.get("context_m", 1.0),
                "hr": info.get("context_hr", 70.0),
                "eda": info.get("context_eda", 0.0),
            }
        )

    df = pd.DataFrame(rows).set_index("time")

    # IOB calculation
    def compute_iob(
        ins_rate_series: pd.Series, dt_min: float = 3.0, dia_min: float = 360.0
    ) -> pd.Series:
        iob = []
        remain = 0.0
        lam = np.exp(-dt_min / (dia_min / 3.0))
        for r in ins_rate_series.to_numpy(dtype=float):
            units = r * dt_min
            remain = remain * lam + units
            iob.append(remain)
        return pd.Series(iob, index=ins_rate_series.index)

    df["IOB"] = compute_iob(df["insulin"])
    return df, meals


def plot_standard(
    cohort: str,
    seed: int,
    df_b: pd.DataFrame,
    df_c: pd.DataFrame,
    meals: List[Tuple[datetime, int]],
    out: Path,
):
    """Standard plot: Glucose, m(t), HR/EDA, IOB, Basal, Bolus."""

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(6, 1, height_ratios=[2.5, 1.0, 1.2, 1.2, 1.0, 1.0], hspace=0.3)

    # Panel 1: Glucose
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(
        df_b.index,
        df_b["glucose"],
        "gray",
        linestyle="--",
        label="Baseline (no context)",
        linewidth=2.5,
        alpha=0.7,
    )
    ax1.plot(
        df_c.index,
        df_c["glucose"],
        "purple",
        label="Context-aware (delayed response)",
        linewidth=2.5,
        alpha=0.9,
    )

    ax1.axhspan(70, 180, alpha=0.12, color="green", label="Target range (70-180)")
    ax1.axhline(70, color="red", linestyle="--", alpha=0.4, linewidth=1)
    ax1.axhline(180, color="red", linestyle="--", alpha=0.4, linewidth=1)

    for t, g in meals:
        ax1.axvline(t, color="purple", linestyle=":", alpha=0.5, linewidth=1.5)
        ax1.text(
            t,
            ax1.get_ylim()[1] * 0.97,
            f"{g}g",
            ha="center",
            fontsize=8,
            color="purple",
            fontweight="bold",
        )

    ax1.set_ylabel("Glucose (mg/dL)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(40, 300)
    ax1.set_title(
        f"OpenAPS: Baseline vs Context-Aware — {cohort} seed={seed}", fontsize=14, fontweight="bold"
    )

    # Panel 2: m(t)
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(df_c.index, df_c["m_t"], "green", linewidth=2.5, label="ISF Multiplier m(t)")
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax2.fill_between(
        df_c.index,
        1.0,
        df_c["m_t"],
        where=(df_c["m_t"] > 1.0),
        alpha=0.3,
        color="orange",
        label="↑ Sensitivity (exercise)",
    )
    ax2.fill_between(
        df_c.index,
        1.0,
        df_c["m_t"],
        where=(df_c["m_t"] < 1.0),
        alpha=0.3,
        color="red",
        label="↓ Sensitivity (stress)",
    )
    ax2.set_ylabel("ISF Multiplier m(t)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.6, 2.4)

    # Panel 3: HR + EDA
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3_eda = ax3.twinx()
    ax3.plot(df_c.index, df_c["hr"], "blue", linewidth=1.5, alpha=0.7, label="Heart Rate")
    ax3_eda.plot(df_c.index, df_c["eda"], "red", linewidth=1.5, alpha=0.6, label="EDA (stress)")
    ax3.axhline(70, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax3.set_ylabel("HR (bpm)", fontsize=11, fontweight="bold", color="blue")
    ax3_eda.set_ylabel("EDA (norm.)", fontsize=11, fontweight="bold", color="red")
    ax3.tick_params(axis="y", labelcolor="blue")
    ax3_eda.tick_params(axis="y", labelcolor="red")
    ax3.legend(loc="upper left", fontsize=9)
    ax3_eda.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(50, 160)
    ax3_eda.set_ylim(0, 1.0)

    # Panel 4: IOB
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(
        df_b.index,
        df_b["IOB"],
        "gray",
        linestyle="--",
        label="Baseline IOB",
        linewidth=2,
        alpha=0.7,
    )
    ax4.plot(df_c.index, df_c["IOB"], "purple", label="Context IOB", linewidth=2, alpha=0.8)
    ax4.set_ylabel("IOB (Units)", fontsize=11, fontweight="bold")
    ax4.legend(loc="upper right", fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Panel 5: Basal
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    ax5.plot(
        df_b.index,
        df_b["basal"] * 60,
        "gray",
        linestyle="--",
        label="Baseline basal",
        linewidth=1.5,
        alpha=0.7,
    )
    ax5.plot(
        df_c.index, df_c["basal"] * 60, "purple", label="Context basal", linewidth=1.5, alpha=0.8
    )
    ax5.set_ylabel("Basal (U/hr)", fontsize=11, fontweight="bold")
    ax5.legend(loc="upper right", fontsize=9)
    ax5.grid(True, alpha=0.3)

    # Panel 6: Bolus
    ax6 = fig.add_subplot(gs[5], sharex=ax1)
    b_bolus = df_b[df_b["bolus"] > 0.001]
    c_bolus = df_c[df_c["bolus"] > 0.001]
    if len(b_bolus) > 0:
        ax6.scatter(
            b_bolus.index,
            b_bolus["bolus"] * 3,
            color="gray",
            marker="o",
            s=50,
            label="Baseline SMB",
            alpha=0.7,
            edgecolors="black",
        )
    if len(c_bolus) > 0:
        ax6.scatter(
            c_bolus.index,
            c_bolus["bolus"] * 3,
            color="purple",
            marker="s",
            s=50,
            label="Context SMB",
            alpha=0.8,
            edgecolors="darkviolet",
        )
    ax6.set_ylabel("Bolus (U)", fontsize=11, fontweight="bold")
    ax6.set_xlabel("Time", fontsize=12, fontweight="bold")
    ax6.legend(loc="upper right", fontsize=9)
    ax6.grid(True, alpha=0.3)

    import matplotlib.dates as mdates

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_extended(
    cohort: str,
    seed: int,
    df_b: pd.DataFrame,
    df_c: pd.DataFrame,
    meals: List[Tuple[datetime, int]],
    out: Path,
):
    """Extended plot: + Delta Glucose + Delta Insulin panels."""

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(8, 1, height_ratios=[2.5, 1.2, 1.0, 1.2, 1.2, 1.0, 1.0, 1.0], hspace=0.3)

    # Panel 1: Glucose (same as standard)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(
        df_b.index,
        df_b["glucose"],
        "gray",
        linestyle="--",
        label="Baseline",
        linewidth=2.5,
        alpha=0.7,
    )
    ax1.plot(df_c.index, df_c["glucose"], "purple", label="Context-aware", linewidth=2.5, alpha=0.9)
    ax1.axhspan(70, 180, alpha=0.12, color="green")
    ax1.axhline(70, color="red", linestyle="--", alpha=0.4)
    ax1.axhline(180, color="red", linestyle="--", alpha=0.4)
    for t, g in meals:
        ax1.axvline(t, color="purple", linestyle=":", alpha=0.5, linewidth=1.5)
        ax1.text(t, 290, f"{g}g", ha="center", fontsize=8, color="purple", fontweight="bold")
    ax1.set_ylabel("Glucose (mg/dL)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(40, 300)
    ax1.set_title(
        f"OpenAPS: Baseline vs Context — {cohort} seed={seed} (Extended Analysis)",
        fontsize=14,
        fontweight="bold",
    )

    # Panel 2: DELTA GLUCOSE (physiological effect)
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    delta_glucose = df_c["glucose"].values - df_b["glucose"].values
    ax2.fill_between(
        df_c.index,
        0,
        delta_glucose,
        where=(delta_glucose < 0),
        alpha=0.5,
        color="orange",
        label="Context LOWER (good for exercise)",
    )
    ax2.fill_between(
        df_c.index,
        0,
        delta_glucose,
        where=(delta_glucose > 0),
        alpha=0.5,
        color="red",
        label="Context HIGHER",
    )
    ax2.axhline(0, color="black", linestyle="-", linewidth=1.5, alpha=0.7)
    ax2.set_ylabel("ΔGlucose\n(Context - Baseline)\n(mg/dL)", fontsize=10, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Physiological Effect: How context alters glucose", fontsize=10, style="italic")

    # Panel 3: m(t)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(df_c.index, df_c["m_t"], "green", linewidth=2.5)
    ax3.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax3.fill_between(
        df_c.index, 1.0, df_c["m_t"], where=(df_c["m_t"] > 1.0), alpha=0.3, color="orange"
    )
    ax3.fill_between(
        df_c.index, 1.0, df_c["m_t"], where=(df_c["m_t"] < 1.0), alpha=0.3, color="red"
    )
    ax3.set_ylabel("m(t)", fontsize=11, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.6, 2.4)

    # Panel 4: HR + EDA
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4_eda = ax4.twinx()
    ax4.plot(df_c.index, df_c["hr"], "blue", linewidth=1.5, alpha=0.7)
    ax4_eda.plot(df_c.index, df_c["eda"], "red", linewidth=1.5, alpha=0.6)
    ax4.set_ylabel("HR (bpm)", fontsize=11, fontweight="bold", color="blue")
    ax4_eda.set_ylabel("EDA", fontsize=11, fontweight="bold", color="red")
    ax4.tick_params(axis="y", labelcolor="blue")
    ax4_eda.tick_params(axis="y", labelcolor="red")
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(50, 160)
    ax4_eda.set_ylim(0, 1.0)

    # Panel 5: IOB
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    ax5.plot(df_b.index, df_b["IOB"], "gray", linestyle="--", linewidth=2, alpha=0.7)
    ax5.plot(df_c.index, df_c["IOB"], "purple", linewidth=2, alpha=0.8)
    ax5.set_ylabel("IOB (Units)", fontsize=11, fontweight="bold")
    ax5.grid(True, alpha=0.3)

    # Panel 6: DELTA INSULIN (controller response)
    ax6 = fig.add_subplot(gs[5], sharex=ax1)
    delta_insulin = (df_c["insulin"].values - df_b["insulin"].values) * 60  # Convert to U/hr
    ax6.fill_between(
        df_c.index,
        0,
        delta_insulin,
        where=(delta_insulin < 0),
        alpha=0.5,
        color="blue",
        label="Less insulin (context)",
    )
    ax6.fill_between(
        df_c.index,
        0,
        delta_insulin,
        where=(delta_insulin > 0),
        alpha=0.5,
        color="red",
        label="More insulin (context)",
    )
    ax6.axhline(0, color="black", linestyle="-", linewidth=1.5, alpha=0.7)
    ax6.set_ylabel("ΔInsulin\n(Context - Baseline)\n(U/hr)", fontsize=10, fontweight="bold")
    ax6.legend(loc="upper right", fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.set_title(
        "Controller Response: How context alters insulin delivery", fontsize=10, style="italic"
    )

    # Panel 7: Basal
    ax7 = fig.add_subplot(gs[6], sharex=ax1)
    ax7.plot(df_b.index, df_b["basal"] * 60, "gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax7.plot(df_c.index, df_c["basal"] * 60, "purple", linewidth=1.5, alpha=0.8)
    ax7.set_ylabel("Basal (U/hr)", fontsize=11, fontweight="bold")
    ax7.grid(True, alpha=0.3)

    # Panel 8: Bolus
    ax8 = fig.add_subplot(gs[7], sharex=ax1)
    b_bolus = df_b[df_b["bolus"] > 0.001]
    c_bolus = df_c[df_c["bolus"] > 0.001]
    if len(b_bolus) > 0:
        ax8.scatter(
            b_bolus.index,
            b_bolus["bolus"] * 3,
            color="gray",
            marker="o",
            s=50,
            alpha=0.7,
            edgecolors="black",
        )
    if len(c_bolus) > 0:
        ax8.scatter(
            c_bolus.index,
            c_bolus["bolus"] * 3,
            color="purple",
            marker="s",
            s=50,
            alpha=0.8,
            edgecolors="darkviolet",
        )
    ax8.set_ylabel("Bolus (U)", fontsize=11, fontweight="bold")
    ax8.set_xlabel("Time", fontsize=12, fontweight="bold")
    ax8.grid(True, alpha=0.3)

    import matplotlib.dates as mdates

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    """Generate batch plots: baseline vs context-aware."""
    cohorts = ["adolescent#001", "adult#001", "child#001"]
    seeds = [3, 5, 7]
    out_dir = Path(__file__).parent / "batch_plots"

    print("=" * 80)
    print("FINAL BATCH PLOTS: BASELINE vs CONTEXT-AWARE")
    print("=" * 80)
    print(f"\nCohorts: {len(cohorts)}")
    print(f"Seeds: {len(seeds)}")
    print(f"Plots per run: 2 (standard + extended)")
    print(f"Total plots: {len(cohorts) * len(seeds) * 2}")
    print(f"\nOutput: {out_dir}/")
    print("=" * 80)

    for cohort in cohorts:
        for seed in seeds:
            print(f"\n {cohort} seed={seed}")

            print("  [1/2] Baseline...")
            df_b, meals = run_simulation(cohort, seed, use_context=False)

            print("  [2/2] Context-aware...")
            df_c, _ = run_simulation(cohort, seed, use_context=True)

            # Stats
            tir_b = ((df_b["glucose"] >= 70) & (df_b["glucose"] <= 180)).mean() * 100
            tir_c = ((df_c["glucose"] >= 70) & (df_c["glucose"] <= 180)).mean() * 100
            mean_b = df_b["glucose"].mean()
            mean_c = df_c["glucose"].mean()

            print(f"   Baseline:  TIR={tir_b:5.1f}% | Mean BG={mean_b:6.1f}")
            print(f"     Context:   TIR={tir_c:5.1f}% | Mean BG={mean_c:6.1f}")
            print(f"     Diff TIR: {tir_c - tir_b:+.1f}%")

            # Generate plots
            cohort_clean = cohort.replace("#", "")

            print(f"   Generating plots...")
            out_std = out_dir / f"comparison_{cohort_clean}_seed{seed}_standard.png"
            plot_standard(cohort, seed, df_b, df_c, meals, out_std)
            print(f"     - {out_std.name}")

            out_ext = out_dir / f"comparison_{cohort_clean}_seed{seed}_extended.png"
            plot_extended(cohort, seed, df_b, df_c, meals, out_ext)
            print(f"     - {out_ext.name}")

    print("\n" + "=" * 80)
    print("BATCH PLOTS COMPLETE!")
    print("=" * 80)
    print(f"\nAll plots saved to: {out_dir}/")
    print("\nTwo versions per cohort/seed:")
    print("   - Standard: Glucose, m(t), HR/EDA, IOB, Basal, Bolus")
    print("   - Extended: + Delta Glucose + Delta Insulin (deeper analysis)")
    print("\nKey Interpretation Delta Panels:")
    print("   - Delta Glucose: Physiological effect (context lower during exercise = good)")
    print("   - Delta Insulin: Controller response (less insulin during exercise = correct)")
    print("=" * 80)


if __name__ == "__main__":
    main()
