"""Spoor-C LUT controller demo for thesis Chapter 7.

This is a translation/stability demonstration, not clinical validation.

Manchester participant IDs provide learned SRL LUTs (`m*(HR)`). SimGlucose
virtual cohorts are only stress-test vehicles for the OpenAPS-style controller;
there is no identity mapping from Manchester patients to virtual patients.

Default full grid:
  - Manchester LUTs: 2307, 2304, 2305
  - SimGlucose cohorts: adolescent#003, adolescent#005, adult#003
  - seeds: 1, 3, 7, 13, 42
  - arms: baseline, patch, placebo; plus patch_t2 for 2304 only

Smoke:
  .venv/bin/python simglucose_extension/experiments/thesis_spoorc_lut_controller_demo.py --smoke
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PATCH_DIR = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PATCH_DIR) not in sys.path:
    sys.path.insert(0, str(PATCH_DIR))

from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario import CustomScenario

from simglucose_ctx.env_wrapper import ContextAwareT1DSimEnv
from simglucose_ctx.context import ContextConfig, ContextStream
from simglucose_ctx.context_aware_openaps import ContextAwareOpenAPSController


DEFAULT_HR_REST = 68.0
COHORTS = ["adolescent#003", "adolescent#005", "adult#003"]
SEEDS = [1, 3, 7, 13, 42]
MANCHESTER_PIDS = ["2307", "2304", "2305"]
TIER2_PIDS = {"2304"}

STEPS_PER_DAY = 480          # 480 x 3 min = 24 h
N_DAYS = 7
TOTAL_STEPS = STEPS_PER_DAY * N_DAYS
STEP_MIN = 3.0
EXERCISE_DAYS = {0, 2, 4}
SENS_MULTIPLIER = 0.75

LUT_DIR = ROOT / "manchester" / "spoor_c_patches" / "lut"
LABEL_DIR = ROOT / "manchester" / "spoor_c_labels"
TIER2_BETA = {"2304": -0.0100}


@dataclass(frozen=True)
class ProtocolScenario:
    name: str
    peak_delta: float
    exercise_start_min: int = 9 * 60
    ramp_min: int = 15
    plateau_min: int = 60
    cooldown_min: int = 30
    breakfast_carb_scale: float = 1.0
    skip_breakfast_on_exercise_days: bool = False

    @property
    def onset_step(self) -> int:
        return int(self.exercise_start_min / STEP_MIN)

    @property
    def ramp_steps(self) -> int:
        return int(self.ramp_min / STEP_MIN)

    @property
    def plateau_steps(self) -> int:
        return int(self.plateau_min / STEP_MIN)

    @property
    def cooldown_steps(self) -> int:
        return int(self.cooldown_min / STEP_MIN)

    @property
    def exercise_end_min(self) -> int:
        return self.exercise_start_min + 90

    @property
    def cooldown_start_min(self) -> int:
        return self.exercise_start_min + self.ramp_min + self.plateau_min

    @property
    def cooldown_end_min(self) -> int:
        return self.cooldown_start_min + self.cooldown_min


SCENARIOS = {
    "post_breakfast_moderate": ProtocolScenario(
        name="post_breakfast_moderate", peak_delta=50.0
    ),
    "post_breakfast_vigorous": ProtocolScenario(
        name="post_breakfast_vigorous", peak_delta=70.0
    ),
    "post_breakfast_vigorous_halfbreakfast": ProtocolScenario(
        name="post_breakfast_vigorous_halfbreakfast",
        peak_delta=70.0,
        breakfast_carb_scale=0.5,
    ),
    "late_post_breakfast_vigorous": ProtocolScenario(
        name="late_post_breakfast_vigorous",
        peak_delta=70.0,
        exercise_start_min=11 * 60,
    ),
    "fasted_morning_vigorous": ProtocolScenario(
        name="fasted_morning_vigorous",
        peak_delta=70.0,
        skip_breakfast_on_exercise_days=True,
    ),
}


@dataclass(frozen=True)
class PhysiologyMode:
    name: str
    vm0_exponent: float
    vmx_exponent: float
    modulate_p2u: bool
    vmx_duration_gain_per_hour: float = 0.0
    vmx_duration_gain_cap: float = 1.0
    vmx_decay_half_life_min: float | None = None
    vmx_factor_cap: float | None = None


PHYSIOLOGY_MODES = {
    "current_vm0_dominant": PhysiologyMode(
        name="current_vm0_dominant",
        vm0_exponent=3.5,
        vmx_exponent=0.2,
        modulate_p2u=True,
    ),
    "balanced_static": PhysiologyMode(
        name="balanced_static",
        vm0_exponent=1.8,
        vmx_exponent=1.35,
        modulate_p2u=False,
    ),
    "balanced_duration_decay": PhysiologyMode(
        name="balanced_duration_decay",
        vm0_exponent=1.8,
        vmx_exponent=1.35,
        modulate_p2u=False,
        vmx_duration_gain_per_hour=0.20,
        vmx_duration_gain_cap=1.30,
        vmx_decay_half_life_min=90.0,
        vmx_factor_cap=3.5,
    ),
}


@lru_cache(maxsize=None)
def patient_hr_rest(pid: str) -> float:
    path = LUT_DIR / f"lut_{pid}.npz"
    if path.exists():
        data = np.load(path, allow_pickle=True)
        if "rhr" in data.files:
            return float(data["rhr"])
    return DEFAULT_HR_REST


@lru_cache(maxsize=None)
def robust_hr_bounds(pid: str) -> tuple[float, float]:
    path = LUT_DIR / f"lut_{pid}.npz"
    if not path.exists():
        return (np.nan, np.nan)
    data = np.load(path, allow_pickle=True)
    hr_grid = np.asarray(data["hr_grid"], dtype=float)
    curve = np.asarray(data["m_curve"], dtype=float)
    robust = np.asarray(data["robust_mask"], dtype=bool) & np.isfinite(curve)
    if not robust.any():
        return (np.nan, np.nan)
    robust_hr = hr_grid[robust]
    return float(robust_hr.min()), float(robust_hr.max())


def get_scenario(name: str) -> ProtocolScenario:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario {name!r}. Choose from {sorted(SCENARIOS)}")
    return SCENARIOS[name]


def _exercise_delta_profile(scenario: ProtocolScenario) -> np.ndarray:
    ramp = np.linspace(0.0, scenario.peak_delta, scenario.ramp_steps, endpoint=False)
    plateau = np.full(scenario.plateau_steps, scenario.peak_delta)
    cooldown = np.linspace(scenario.peak_delta, 0.0, scenario.cooldown_steps, endpoint=False)
    return np.concatenate([ramp, plateau, cooldown])


def _eda_profile(scenario: ProtocolScenario) -> np.ndarray:
    ramp = np.linspace(0.12, 0.35, scenario.ramp_steps, endpoint=False)
    plateau = np.full(scenario.plateau_steps, 0.35)
    cooldown = np.linspace(0.22, 0.12, scenario.cooldown_steps, endpoint=False)
    return np.concatenate([ramp, plateau, cooldown])


def build_7day_hr_eda(start, seed, scenario: ProtocolScenario, hr_rest: float):
    idx = pd.date_range(start, periods=TOTAL_STEPS, freq="3min")
    rng = np.random.RandomState(seed)

    t_of_day = np.arange(TOTAL_STEPS) % STEPS_PER_DAY
    circ = hr_rest + 5 * np.sin(2 * np.pi * t_of_day / STEPS_PER_DAY - 1.2)
    hr = circ + rng.randn(TOTAL_STEPS) * 2.5
    eda = np.full(TOTAL_STEPS, 0.12)
    delta_profile = _exercise_delta_profile(scenario)
    eda_profile = _eda_profile(scenario)
    block_len = len(delta_profile)
    block_start = scenario.onset_step
    block_end = block_start + block_len

    for d in range(N_DAYS):
        off = d * STEPS_PER_DAY
        if d in EXERCISE_DAYS:
            hr[off + block_start:off + block_end] += delta_profile
            eda[off + block_start:off + block_end] = eda_profile
        elif rng.rand() < 0.5:
            hr[off + 350:off + 380] += 15
            eda[off + 350:off + 380] = 0.18

        for _ in range(int(rng.randint(4, 7))):
            c = int(rng.randint(12, STEPS_PER_DAY - 12))
            if d in EXERCISE_DAYS and block_start - 10 <= c <= block_end + 10:
                continue
            w = int(rng.randint(2, 5))
            hr[off + max(0, c - w):off + c + w] += rng.uniform(5, 15)

    hr += rng.randn(len(hr)) * 1.5
    eda += rng.randn(len(eda)) * 0.03
    eda = np.clip(eda, 0.05, 0.9)
    return pd.Series(hr, index=idx), pd.Series(eda, index=idx)


def build_7day_meals(start, seed, scenario: ProtocolScenario | None = None):
    scenario = scenario or SCENARIOS["post_breakfast_vigorous"]
    rng = np.random.RandomState(seed + 100)
    meals = []
    for d in range(N_DAYS):
        day_start = start + timedelta(days=d)
        if not (scenario.skip_breakfast_on_exercise_days and d in EXERCISE_DAYS):
            breakfast_carbs = max(
                0,
                int(round((60 + int(rng.randint(-5, 6))) * scenario.breakfast_carb_scale)),
            )
            if breakfast_carbs > 0:
                meals.append((day_start + timedelta(hours=7, minutes=30)
                              + timedelta(minutes=int(rng.randint(-10, 11))),
                              breakfast_carbs))
        meals.append((day_start + timedelta(hours=12, minutes=30)
                      + timedelta(minutes=int(rng.randint(-15, 16))),
                      55 + int(rng.randint(-5, 6))))
        meals.append((day_start + timedelta(hours=18, minutes=30)
                      + timedelta(minutes=int(rng.randint(-15, 16))),
                      55 + int(rng.randint(-5, 6))))
        if d not in EXERCISE_DAYS and rng.rand() < 0.5:
            meals.append((day_start + timedelta(hours=10, minutes=30)
                          + timedelta(minutes=int(rng.randint(-20, 21))),
                          15 + int(rng.randint(-3, 4))))
    meals.sort(key=lambda x: x[0])
    return meals


def get_physiology_mode(name: str) -> PhysiologyMode:
    try:
        return PHYSIOLOGY_MODES[name]
    except KeyError as exc:
        valid = ", ".join(sorted(PHYSIOLOGY_MODES))
        raise ValueError(f"Unknown physiology mode {name!r}. Valid: {valid}") from exc


def ctx_cfg(hr_rest: float, physiology: PhysiologyMode | None = None):
    physiology = physiology or PHYSIOLOGY_MODES["current_vm0_dominant"]
    return ContextConfig(
        dt_minutes=3,
        alpha=1.25,
        beta=0.3,
        mmax=2.5,
        mmin=0.7,
        hr_rest=hr_rest,
        hr_max=180.0,
        ema_half_life_min=8.0,
        stress_hr_supp_pow=2.0,
        stress_hr_off_threshold=0.75,
        vm0_exponent_exercise=physiology.vm0_exponent,
        vmx_exponent_exercise=physiology.vmx_exponent,
        vmx_duration_gain_per_hour=physiology.vmx_duration_gain_per_hour,
        vmx_duration_gain_cap=physiology.vmx_duration_gain_cap,
        vmx_decay_half_life_min=physiology.vmx_decay_half_life_min,
        vmx_factor_cap=physiology.vmx_factor_cap,
    )


def profile(sens_mult=SENS_MULTIPLIER):
    sens_base = 50.0
    return {
        "current_basal": 1.0,
        "sens": sens_base * sens_mult,
        "min_bg": 90.0, "max_bg": 120.0,
        "max_iob": 4.0, "bolus_increment": 0.05,
        "maxSMBBasalMinutes": 30, "SMBInterval": 3,
        "enableSMB_always": False, "enableSMB_with_COB": True,
        "enableSMB_high_bg": True, "enableSMB_after_carbs": True,
        "enableSMB_with_temptarget": False, "enableSMB_high_bg_target": 140,
        "temp_basal_duration_min": 30, "max_safe_basal": 3.0,
    }


def build_tier2_lookup(pid: str):
    if pid not in TIER2_PIDS:
        return None, None, None
    path = LABEL_DIR / f"{pid}.csv"
    if not path.exists():
        return None, None, None

    df = pd.read_csv(path, parse_dates=["time"])
    df = df[df["valid"] == 1].dropna(subset=["hr_abs", "stress_level"]).sort_values("time")
    if len(df) < 50:
        return None, None, None

    split = int(0.8 * len(df))
    train = df.iloc[:split]
    hr_ref = train["hr_abs"].to_numpy(dtype=float)
    stress_ref = train["stress_level"].to_numpy(dtype=float)
    hr_grid = np.arange(50.0, 170.0)
    expected = np.full(len(hr_grid), np.nan)
    for i, h in enumerate(hr_grid):
        mask = (hr_ref >= h - 5.0) & (hr_ref <= h + 5.0)
        if mask.sum() >= 10:
            expected[i] = float(np.median(stress_ref[mask]))

    valid = np.isfinite(expected)
    if valid.sum() < 2:
        return None, None, None
    expected = np.interp(hr_grid, hr_grid[valid], expected[valid])
    return TIER2_BETA[pid], hr_grid, expected


def make_controller(pid, arm, sens_mult, hr_rest=None):
    if hr_rest is None:
        hr_rest = patient_hr_rest(pid)
    p = profile(sens_mult=sens_mult)
    if arm == "baseline":
        return ContextAwareOpenAPSController(p, model_path=None, hr_rest=hr_rest)

    beta, stress_grid, expected_stress = build_tier2_lookup(pid) if arm == "patch_t2" else (None, None, None)
    lut_policy = "full_shape" if arm == "patch_full_shape" else "safe_upward"
    lut_anticipation = "slope_projected" if arm in {"patch_slope_projected", "patch_anticipation"} else "off"
    exercise_guard = "smb_basal_cap" if arm == "patch_exercise_guard" else "off"
    return ContextAwareOpenAPSController(
        p,
        model_path=str(LUT_DIR / f"lut_{pid}.npz"),
        hr_rest=hr_rest,
        patch_version="spoorC_lut",
        lut_policy=lut_policy,
        lut_anticipation=lut_anticipation,
        exercise_guard=exercise_guard,
        tier2_beta=beta,
        tier2_stress_hr_grid=stress_grid,
        tier2_expected_stress=expected_stress,
    )


def precompute_placebo_m(pid, hr, eda, seed, sens_mult, hr_rest=None):
    ctrl = make_controller(pid, "patch", sens_mult, hr_rest=hr_rest)
    vals = []
    for h, e in zip(hr.to_numpy(dtype=float), eda.to_numpy(dtype=float)):
        vals.append(ctrl._predict_m_t({
            "context_hr": float(h),
            "context_eda": float(e) * 100.0,
            "context_eda_is_stress_level": True,
            "context_hr_is_relative": False,
        }, ctrl._now))
    vals = np.asarray(vals, dtype=float)
    rng = np.random.RandomState(seed)
    return vals[rng.permutation(len(vals))]


def run_arm(
    pid, cohort, seed, scenario_name, arm, hr, eda, meals, start,
    sens_mult, hr_rest, physiology_name="current_vm0_dominant",
):
    physiology = get_physiology_mode(physiology_name)
    patient = T1DPatient.withName(cohort, seed=1)
    sensor = CGMSensor.withName("Dexcom", seed=1)
    pump = InsulinPump.withName("Insulet")
    scenario = CustomScenario(start_time=start, scenario=meals)

    ctx = ContextStream(hr, eda, ctx_cfg(hr_rest, physiology), preprocess=False)
    env = ContextAwareT1DSimEnv(
        patient, sensor, pump, scenario,
        context_stream=ctx,
        modulate_vm0=True,
        modulate_vmx=True,
        modulate_p2u=physiology.modulate_p2u,
        max_log_size=3600,
    )
    ctrl = make_controller(pid, "patch" if arm == "placebo" else arm, sens_mult, hr_rest=hr_rest)
    placebo_m = (
        precompute_placebo_m(pid, hr, eda, seed + int(pid), sens_mult, hr_rest=hr_rest)
        if arm == "placebo" else None
    )
    robust_min, robust_max = robust_hr_bounds(pid)

    reset = env.reset()
    obs = reset.observation if hasattr(reset, "observation") else reset
    last_meal = 0.0
    rows = []
    for step_i in range(TOTAL_STEPS):
        st = env.env.sample_time if hasattr(env, "env") else env.sample_time
        current_time = env.env.time if hasattr(env, "env") else env.time
        ts = pd.Timestamp(current_time)
        pos = hr.index.searchsorted(ts, side="right")
        idx = hr.index[0] if pos == 0 else hr.index[pos - 1]
        info = {
            "sample_time": st,
            "meal": last_meal,
            "context_hr": float(hr.loc[idx]),
            "context_eda": float(eda.loc[idx]) * 100.0,
            "context_eda_is_stress_level": True,
            "context_hr_is_relative": False,
        }
        if placebo_m is not None:
            info["context_m_override"] = float(placebo_m[min(step_i, len(placebo_m) - 1)])

        action = ctrl.policy(obs, reward=0.0, done=False, **info)
        step = env.step(action)
        obs = step.observation
        last_meal = step.info.get("meal", 0.0)
        m_t_val = getattr(ctrl, "last_m_t", 1.0) if arm != "baseline" else 1.0
        hr_val = info["context_hr"]
        hr_above_robust = bool(np.isfinite(robust_max) and hr_val > robust_max)
        rows.append({
            "time": current_time,
            "manchester_pid": pid,
            "scenario": scenario_name,
            "physiology_mode": physiology.name,
            "cohort": cohort,
            "seed": seed,
            "arm": arm,
            "glucose": float(obs.CGM if hasattr(obs, "CGM") else obs),
            "basal": step.info.get("action_basal", action.basal),
            "bolus": step.info.get("action_bolus", action.bolus),
            "context_m": step.info.get("context_m", 1.0),
            "Vm0_factor": step.info.get("context_vm0_factor", 1.0),
            "Vmx_factor": step.info.get("context_vmx_factor", 1.0),
            "p2u_factor": step.info.get("context_p2u_factor", 1.0),
            "duration_gain": step.info.get("context_vmx_duration_gain", 1.0),
            "exercise_minutes": step.info.get("context_exercise_minutes", 0.0),
            "m_t": m_t_val,
            "m_t_lut": getattr(ctrl, "last_m_t_lut", m_t_val),
            "m_t_projected": getattr(ctrl, "last_m_t_projected", m_t_val),
            "m_t_final": getattr(ctrl, "last_m_t_final", m_t_val),
            "hr_slope_bpm_per_min": getattr(ctrl, "last_hr_slope", 0.0),
            "hr_projected": getattr(ctrl, "last_hr_projected", np.nan),
            "lut_anticipation_active": bool(getattr(ctrl, "last_lut_anticipation_active", False)),
            "lut_anticipation_mode": getattr(ctrl, "last_lut_anticipation_mode", "off"),
            "exercise_guard_active": bool(getattr(ctrl, "last_exercise_guard_active", False)),
            "exercise_guard_mode": getattr(ctrl, "last_exercise_guard_mode", "off"),
            "patched_isf": getattr(ctrl, "last_patched_isf", profile(sens_mult)["sens"]),
            "iob": getattr(ctrl, "last_iob", np.nan),
            "basaliob": getattr(ctrl, "last_basaliob", np.nan),
            "bolusiob": getattr(ctrl, "last_bolusiob", np.nan),
            "netbasalinsulin": getattr(ctrl, "last_netbasalinsulin", np.nan),
            "bolusinsulin": getattr(ctrl, "last_bolusinsulin", np.nan),
            "activity": getattr(ctrl, "last_activity", np.nan),
            "cob": getattr(ctrl, "last_cob", np.nan),
            "basal_rate_u_per_hr": getattr(ctrl, "last_basal_rate_u_per_hr", np.nan),
            "basal_duration_min": getattr(ctrl, "last_basal_duration_min", np.nan),
            "basal_reason": getattr(ctrl, "last_basal_reason", None),
            "bolus_units": getattr(ctrl, "last_bolus_units", np.nan),
            "bolus_reason": getattr(ctrl, "last_bolus_reason", None),
            "smb_projection_bg": getattr(ctrl, "last_smb_projection_bg", np.nan),
            "smb_insulin_req": getattr(ctrl, "last_smb_insulin_req", np.nan),
            "smb_enabled": bool(getattr(ctrl, "last_smb_enabled", False)),
            "smb_veto_reason": getattr(ctrl, "last_smb_veto_reason", None),
            "min_guard_bg": getattr(ctrl, "last_min_guard_bg", np.nan),
            "min_pred_bg": getattr(ctrl, "last_min_pred_bg", np.nan),
            "eventual_bg": getattr(ctrl, "last_eventual_bg", np.nan),
            "naive_eventual_bg": getattr(ctrl, "last_naive_eventual_bg", np.nan),
            "deviation": getattr(ctrl, "last_deviation", np.nan),
            "hr": hr_val,
            "eda": float(eda.loc[idx]),
            "hr_rest": hr_rest,
            "robust_hr_min": robust_min,
            "robust_hr_max": robust_max,
            "hr_above_robust": hr_above_robust,
            "edge_hold": bool(arm != "baseline" and hr_above_robust and m_t_val > 1.0),
        })
    df = pd.DataFrame(rows).set_index("time")
    df["insulin"] = df["basal"] + df["bolus"]
    return df


def lbgi_hbgi(glucose):
    g = np.clip(np.asarray(glucose, dtype=float), 20.0, 600.0)
    f = 1.509 * (np.log(g) ** 1.084 - 5.381)
    rl = np.where(f < 0, 10 * f ** 2, 0.0)
    rh = np.where(f > 0, 10 * f ** 2, 0.0)
    return float(rl.mean()), float(rh.mean())


def _auc_under(series, threshold):
    return float(np.maximum(threshold - series.to_numpy(dtype=float), 0.0).sum() * STEP_MIN)


def compute_metrics(df, scenario: ProtocolScenario):
    g = df["glucose"]
    secs = np.asarray((df.index - df.index[0]).total_seconds(), dtype=np.float64)
    day_num = (secs // 86400).astype(int)
    minute_of_day = ((secs % 86400) // 60).astype(int)
    exer_day_mask = pd.Series(np.isin(day_num, list(EXERCISE_DAYS)), index=df.index)
    rest_day_mask = ~exer_day_mask
    minute_series = pd.Series(minute_of_day, index=df.index)
    pre_mask = (
        exer_day_mask
        & (minute_series >= 7 * 60 + 30)
        & (minute_series < scenario.exercise_start_min)
    )
    exercise_mask = (
        exer_day_mask
        & (minute_series >= scenario.exercise_start_min)
        & (minute_series < scenario.exercise_end_min)
    )
    early_recovery_mask = (
        exer_day_mask
        & (minute_series >= scenario.exercise_end_min)
        & (minute_series < 12 * 60)
    )
    late_recovery_mask = (
        exer_day_mask
        & (minute_series >= 12 * 60)
        & (minute_series < 16 * 60 + 30)
    )
    lbgi, hbgi = lbgi_hbgi(g.values)

    def _stats(frame, prefix):
        if len(frame) == 0:
            return {
                f"{prefix}_tir": np.nan, f"{prefix}_tbr_70": np.nan,
                f"{prefix}_tbr_54": np.nan, f"{prefix}_min": np.nan,
                f"{prefix}_mean": np.nan, f"{prefix}_auc70": np.nan,
                f"{prefix}_auc75": np.nan,
                f"{prefix}_tar_180": np.nan, f"{prefix}_total_insulin": np.nan,
                f"{prefix}_mean_m_t": np.nan, f"{prefix}_max_m_t": np.nan,
                f"{prefix}_frac_m_gt_1_2": np.nan, f"{prefix}_frac_m_gt_2_0": np.nan,
                f"{prefix}_frac_m_ge_1_2": np.nan,
                f"{prefix}_frac_hr_above_robust": np.nan, f"{prefix}_frac_edge_hold": np.nan,
                f"{prefix}_mean_context_m": np.nan,
                f"{prefix}_frac_anticipation_active": np.nan,
                f"{prefix}_frac_exercise_guard_active": np.nan,
                f"{prefix}_mean_hr_slope": np.nan,
                f"{prefix}_mean_iob": np.nan,
                f"{prefix}_mean_bolusiob": np.nan,
                f"{prefix}_mean_basaliob": np.nan,
                f"{prefix}_mean_cob": np.nan,
                f"{prefix}_mean_smb_insulin_req": np.nan,
                f"{prefix}_total_smb_units": np.nan,
                f"{prefix}_frac_smb_enabled": np.nan,
                f"{prefix}_mean_smb_projection_bg": np.nan,
                f"{prefix}_mean_Vm0_factor": np.nan, f"{prefix}_max_Vm0_factor": np.nan,
                f"{prefix}_mean_Vmx_factor": np.nan, f"{prefix}_max_Vmx_factor": np.nan,
                f"{prefix}_mean_p2u_factor": np.nan,
                f"{prefix}_mean_duration_gain": np.nan, f"{prefix}_max_duration_gain": np.nan,
                f"{prefix}_max_exercise_minutes": np.nan,
            }
        series = frame["glucose"]
        def _mean_col(col, default=np.nan):
            return float(frame[col].mean()) if col in frame else default
        def _max_col(col, default=np.nan):
            return float(frame[col].max()) if col in frame else default
        return {
            f"{prefix}_tir": float(((series >= 70) & (series <= 180)).mean() * 100),
            f"{prefix}_tbr_70": float((series < 70).mean() * 100),
            f"{prefix}_tbr_54": float((series < 54).mean() * 100),
            f"{prefix}_tar_180": float((series > 180).mean() * 100),
            f"{prefix}_min": float(series.min()),
            f"{prefix}_mean": float(series.mean()),
            f"{prefix}_auc70": _auc_under(series, 70.0),
            f"{prefix}_auc75": _auc_under(series, 75.0),
            f"{prefix}_total_insulin": float(frame["insulin"].sum() * STEP_MIN),
            f"{prefix}_mean_m_t": float(frame["m_t"].mean()),
            f"{prefix}_max_m_t": float(frame["m_t"].max()),
            f"{prefix}_frac_m_gt_1_2": float((frame["m_t"] > 1.2).mean()),
            f"{prefix}_frac_m_ge_1_2": float((frame["m_t"] >= 1.2).mean()),
            f"{prefix}_frac_m_gt_2_0": float((frame["m_t"] > 2.0).mean()),
            f"{prefix}_frac_hr_above_robust": float(frame["hr_above_robust"].mean()),
            f"{prefix}_frac_edge_hold": float(frame["edge_hold"].mean()),
            f"{prefix}_frac_anticipation_active": _mean_col("lut_anticipation_active"),
            f"{prefix}_frac_exercise_guard_active": _mean_col("exercise_guard_active"),
            f"{prefix}_mean_hr_slope": _mean_col("hr_slope_bpm_per_min"),
            f"{prefix}_mean_iob": _mean_col("iob"),
            f"{prefix}_mean_bolusiob": _mean_col("bolusiob"),
            f"{prefix}_mean_basaliob": _mean_col("basaliob"),
            f"{prefix}_mean_cob": _mean_col("cob"),
            f"{prefix}_mean_smb_insulin_req": _mean_col("smb_insulin_req"),
            f"{prefix}_total_smb_units": float(frame["bolus_units"].sum()) if "bolus_units" in frame else np.nan,
            f"{prefix}_frac_smb_enabled": _mean_col("smb_enabled"),
            f"{prefix}_mean_smb_projection_bg": _mean_col("smb_projection_bg"),
            f"{prefix}_mean_context_m": _mean_col("context_m"),
            f"{prefix}_mean_Vm0_factor": _mean_col("Vm0_factor"),
            f"{prefix}_max_Vm0_factor": _max_col("Vm0_factor"),
            f"{prefix}_mean_Vmx_factor": _mean_col("Vmx_factor"),
            f"{prefix}_max_Vmx_factor": _max_col("Vmx_factor"),
            f"{prefix}_mean_p2u_factor": _mean_col("p2u_factor"),
            f"{prefix}_mean_duration_gain": _mean_col("duration_gain"),
            f"{prefix}_max_duration_gain": _max_col("duration_gain"),
            f"{prefix}_max_exercise_minutes": _max_col("exercise_minutes"),
        }

    m = dict(
        tir=float(((g >= 70) & (g <= 180)).mean() * 100),
        tbr_70=float((g < 70).mean() * 100),
        tbr_54=float((g < 54).mean() * 100),
        tar_180=float((g > 180).mean() * 100),
        tar_250=float((g > 250).mean() * 100),
        min_glu=float(g.min()),
        max_glu=float(g.max()),
        mean_glu=float(g.mean()),
        auc70=_auc_under(g, 70.0),
        auc75=_auc_under(g, 75.0),
        lbgi=lbgi,
        hbgi=hbgi,
        total_insulin=float(df["insulin"].sum() * STEP_MIN),
        mean_m_t=float(df["m_t"].mean()),
        max_m_t=float(df["m_t"].max()),
        frac_m_gt_1_2=float((df["m_t"] > 1.2).mean()),
        frac_m_ge_1_2=float((df["m_t"] >= 1.2).mean()),
        frac_m_gt_2_0=float((df["m_t"] > 2.0).mean()),
        frac_hr_above_robust=float(df["hr_above_robust"].mean()),
        frac_edge_hold=float(df["edge_hold"].mean()),
        frac_anticipation_active=float(df["lut_anticipation_active"].mean()) if "lut_anticipation_active" in df else np.nan,
        frac_exercise_guard_active=float(df["exercise_guard_active"].mean()) if "exercise_guard_active" in df else np.nan,
        mean_hr_slope=float(df["hr_slope_bpm_per_min"].mean()) if "hr_slope_bpm_per_min" in df else np.nan,
        mean_iob=float(df["iob"].mean()) if "iob" in df else np.nan,
        mean_bolusiob=float(df["bolusiob"].mean()) if "bolusiob" in df else np.nan,
        mean_basaliob=float(df["basaliob"].mean()) if "basaliob" in df else np.nan,
        mean_cob=float(df["cob"].mean()) if "cob" in df else np.nan,
        mean_smb_insulin_req=float(df["smb_insulin_req"].mean()) if "smb_insulin_req" in df else np.nan,
        total_smb_units=float(df["bolus_units"].sum()) if "bolus_units" in df else np.nan,
        frac_smb_enabled=float(df["smb_enabled"].mean()) if "smb_enabled" in df else np.nan,
        mean_smb_projection_bg=float(df["smb_projection_bg"].mean()) if "smb_projection_bg" in df else np.nan,
        mean_context_m=float(df["context_m"].mean()) if "context_m" in df else np.nan,
        max_context_m=float(df["context_m"].max()) if "context_m" in df else np.nan,
        mean_Vm0_factor=float(df["Vm0_factor"].mean()) if "Vm0_factor" in df else np.nan,
        max_Vm0_factor=float(df["Vm0_factor"].max()) if "Vm0_factor" in df else np.nan,
        mean_Vmx_factor=float(df["Vmx_factor"].mean()) if "Vmx_factor" in df else np.nan,
        max_Vmx_factor=float(df["Vmx_factor"].max()) if "Vmx_factor" in df else np.nan,
        mean_p2u_factor=float(df["p2u_factor"].mean()) if "p2u_factor" in df else np.nan,
        mean_duration_gain=float(df["duration_gain"].mean()) if "duration_gain" in df else np.nan,
        max_duration_gain=float(df["duration_gain"].max()) if "duration_gain" in df else np.nan,
        max_exercise_minutes=float(df["exercise_minutes"].max()) if "exercise_minutes" in df else np.nan,
    )
    m.update(_stats(df[pre_mask], "pre_exercise"))
    m.update(_stats(df[exercise_mask], "exercise"))
    m.update(_stats(df[early_recovery_mask], "early_recovery"))
    m.update(_stats(df[late_recovery_mask], "late_recovery"))
    m.update(_stats(df[exer_day_mask], "exerday"))
    m.update(_stats(df[rest_day_mask], "restday"))
    return m


def arms_for_pid(pid, include_full_shape=False):
    arms = ["baseline", "patch", "placebo"]
    if pid in TIER2_PIDS:
        arms.append("patch_t2")
    if include_full_shape and pid in {"2304", "2305"}:
        arms.append("patch_full_shape")
    return arms


def trajectory_path(out_dir, scenario_name, physiology_name, pid, cohort, seed, arm):
    cohort_slug = cohort.replace("#", "")
    return out_dir / f"traj_{physiology_name}_{scenario_name}_pid{pid}_{cohort_slug}_s{seed}_{arm}.csv"


def plot_exercise_profiles(out_dir, metrics):
    for physiology_name in sorted(metrics["physiology_mode"].unique()):
        for scenario_name in sorted(metrics["scenario"].unique()):
            scenario = get_scenario(scenario_name)
            exercise_onset = pd.Timedelta(minutes=scenario.exercise_start_min)
            window_start = -90
            window_end = 360
            for pid in sorted(metrics["manchester_pid"].unique()):
                rows = []
                pattern = f"traj_{physiology_name}_{scenario_name}_pid{pid}_*_*.csv"
                for path in sorted(out_dir.glob(pattern)):
                    df = pd.read_csv(path, parse_dates=["time"]).set_index("time")
                    arm = str(df["arm"].iloc[0])
                    for d in EXERCISE_DAYS:
                        day_start = df.index[0].normalize() + pd.Timedelta(days=int(d))
                        onset = day_start + exercise_onset
                        start = onset + pd.Timedelta(minutes=window_start)
                        end = onset + pd.Timedelta(minutes=window_end)
                        seg = df[(df.index >= start) & (df.index < end)].copy()
                        if len(seg) == 0:
                            continue
                        seg["rel_min"] = (seg.index - onset).total_seconds() / 60.0
                        rows.append(seg[[
                            "rel_min", "glucose", "m_t", "hr", "insulin",
                            "Vm0_factor", "Vmx_factor", "arm"
                        ]])
                if not rows:
                    continue
                prof = pd.concat(rows, ignore_index=True)
                med = prof.groupby(["arm", "rel_min"], as_index=False)[[
                    "glucose", "m_t", "hr", "insulin", "Vm0_factor", "Vmx_factor"
                ]].median()
                pivot_g = med.pivot(index="rel_min", columns="arm", values="glucose")
                pivot_m = med.pivot(index="rel_min", columns="arm", values="m_t")
                pivot_hr = med.pivot(index="rel_min", columns="arm", values="hr")
                pivot_i = med.pivot(index="rel_min", columns="arm", values="insulin")
                pivot_vm0 = med.pivot(index="rel_min", columns="arm", values="Vm0_factor")
                pivot_vmx = med.pivot(index="rel_min", columns="arm", values="Vmx_factor")

                fig, axes = plt.subplots(5, 1, figsize=(10.0, 11.5), sharex=True)
                if "baseline" in pivot_g:
                    for arm, color in [
                        ("patch", "crimson"),
                        ("patch_slope_projected", "mediumblue"),
                        ("patch_exercise_guard", "seagreen"),
                        ("placebo", "gray"),
                        ("patch_t2", "darkorange"),
                        ("patch_full_shape", "royalblue"),
                    ]:
                        if arm in pivot_g:
                            axes[0].plot(
                                pivot_g.index,
                                pivot_g[arm] - pivot_g["baseline"],
                                label=f"{arm} - baseline",
                                color=color,
                            )
                axes[0].axhline(0, color="black", lw=0.8, ls="--")
                axes[0].set_ylabel("Median glucose diff (mg/dL)")
                axes[0].grid(alpha=0.25)
                axes[0].legend(fontsize=8)

                for arm, color in [
                    ("patch", "crimson"),
                    ("patch_slope_projected", "mediumblue"),
                    ("patch_exercise_guard", "seagreen"),
                    ("placebo", "gray"),
                    ("patch_t2", "darkorange"),
                    ("patch_full_shape", "royalblue"),
                ]:
                    if arm in pivot_m:
                        axes[1].plot(pivot_m.index, pivot_m[arm], label=arm, color=color)
                axes[1].axhline(1.0, color="black", lw=0.8, ls="--")
                axes[1].set_ylabel("Median m_t")
                axes[1].grid(alpha=0.25)
                axes[1].legend(fontsize=8)

                if "baseline" in pivot_i:
                    for arm, color in [
                        ("patch", "crimson"),
                        ("patch_slope_projected", "mediumblue"),
                        ("patch_exercise_guard", "seagreen"),
                        ("placebo", "gray"),
                        ("patch_t2", "darkorange"),
                    ]:
                        if arm in pivot_i:
                            axes[2].plot(
                                pivot_i.index,
                                pivot_i[arm] - pivot_i["baseline"],
                                label=f"{arm} - baseline",
                                color=color,
                            )
                axes[2].axhline(0, color="black", lw=0.8, ls="--")
                axes[2].set_ylabel("Median insulin diff")
                axes[2].grid(alpha=0.25)
                axes[2].legend(fontsize=8)

                plant_source = "baseline" if "baseline" in pivot_vm0 else pivot_vm0.columns[0]
                axes[3].plot(pivot_vm0.index, pivot_vm0[plant_source], label="Vm0 factor", color="firebrick")
                axes[3].plot(pivot_vmx.index, pivot_vmx[plant_source], label="Vmx factor", color="purple")
                axes[3].axhline(1.0, color="black", lw=0.8, ls="--")
                axes[3].set_ylabel("Plant factors")
                axes[3].grid(alpha=0.25)
                axes[3].legend(fontsize=8)

                if "baseline" in pivot_hr:
                    axes[4].plot(pivot_hr.index, pivot_hr["baseline"], label="HR", color="black")
                else:
                    axes[4].plot(pivot_hr.index, pivot_hr.iloc[:, 0], label="HR", color="black")
                axes[4].set_xlabel("Minutes relative to exercise onset")
                axes[4].set_ylabel("Median HR (bpm)")
                axes[4].grid(alpha=0.25)
                axes[4].legend(fontsize=8)

                for ax in axes:
                    ax.axvspan(0, 90, color="orange", alpha=0.10, label=None)
                    ax.axvspan(scenario.cooldown_start_min - scenario.exercise_start_min,
                               scenario.cooldown_end_min - scenario.exercise_start_min,
                               color="teal", alpha=0.10, label=None)
                    ax.axvspan(180, window_end, color="purple", alpha=0.06, label=None)

                fig.suptitle(f"Spoor-C LUT {physiology_name} / {scenario_name} — Manchester PID {pid}")
                fig.tight_layout()
                fig.savefig(out_dir / f"exercise_profile_{physiology_name}_{scenario_name}_pid{pid}.png", dpi=130)
                plt.close(fig)


def plot_summary_bars(out_dir, pairwise):
    if len(pairwise) == 0:
        return
    plot_df = pairwise[pairwise["arm"].isin(["patch", "patch_slope_projected", "patch_exercise_guard", "placebo", "patch_t2"])].copy()
    if len(plot_df) == 0:
        return
    summary = (
        plot_df.groupby(["physiology_mode", "scenario", "manchester_pid", "arm"], as_index=False)
        [["exercise_auc75_diff", "early_recovery_auc75_diff", "late_recovery_auc75_diff"]]
        .mean(numeric_only=True)
    )
    for physiology_name in sorted(summary["physiology_mode"].unique()):
        for scenario_name in sorted(summary["scenario"].unique()):
            sub = summary[
                (summary["physiology_mode"] == physiology_name)
                & (summary["scenario"] == scenario_name)
            ]
            if len(sub) == 0:
                continue
            labels = []
            values = []
            colors = []
            for _, row in sub.iterrows():
                labels.append(f"{row['manchester_pid']} {row['arm']}")
                values.append(row["exercise_auc75_diff"] + row["early_recovery_auc75_diff"])
                colors.append(
                    {
                        "patch": "crimson",
                        "patch_slope_projected": "mediumblue",
                        "patch_exercise_guard": "seagreen",
                        "placebo": "gray",
                        "patch_t2": "darkorange",
                    }.get(
                        row["arm"], "royalblue"
                    )
                )
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.bar(labels, values, color=colors)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_ylabel("Mean AUC<75 diff, exercise + early recovery")
            ax.set_title(f"Spoor-C LUT summary — {physiology_name} / {scenario_name}")
            ax.tick_params(axis="x", rotation=35)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_dir / f"summary_auc75_{physiology_name}_{scenario_name}.png", dpi=130)
            plt.close(fig)


def _first_crossing_min(frame, threshold):
    hit = frame[frame["m_t"] > threshold]
    if len(hit) == 0:
        return np.nan
    return float(hit["rel_min"].min())


def run_timing_assay(out_dir, pids, seeds, scenarios, sens_mult):
    """Predictor-only assay: no SimGlucose plant, just LUT activation timing."""
    start = datetime(2025, 1, 6, 0, 0)
    rows = []
    arms = ["patch", "patch_slope_projected"]

    for scenario in scenarios:
        exercise_onset = pd.Timedelta(minutes=scenario.exercise_start_min)
        for pid in pids:
            hr_rest = patient_hr_rest(pid)
            robust_min, robust_max = robust_hr_bounds(pid)
            for seed in seeds:
                hr, eda = build_7day_hr_eda(start, seed, scenario, hr_rest)
                for arm in arms:
                    ctrl = make_controller(pid, arm, sens_mult, hr_rest=hr_rest)
                    for ts, h_val, e_val in zip(hr.index, hr.to_numpy(dtype=float), eda.to_numpy(dtype=float)):
                        info = {
                            "context_hr": float(h_val),
                            "context_eda": float(e_val) * 100.0,
                            "context_eda_is_stress_level": True,
                            "context_hr_is_relative": False,
                        }
                        m_t = ctrl._predict_m_t(info, ts.to_pydatetime())
                        for d in EXERCISE_DAYS:
                            day_start = hr.index[0].normalize() + pd.Timedelta(days=int(d))
                            onset = day_start + exercise_onset
                            rel_min = (ts - onset).total_seconds() / 60.0
                            if -90.0 <= rel_min < 180.0:
                                rows.append({
                                    "time": ts,
                                    "scenario": scenario.name,
                                    "manchester_pid": pid,
                                    "seed": seed,
                                    "exercise_day": int(d),
                                    "arm": arm,
                                    "rel_min": rel_min,
                                    "hr": float(h_val),
                                    "eda": float(e_val),
                                    "hr_rest": hr_rest,
                                    "robust_hr_min": robust_min,
                                    "robust_hr_max": robust_max,
                                    "m_t": float(m_t),
                                    "m_t_lut": getattr(ctrl, "last_m_t_lut", float(m_t)),
                                    "m_t_projected": getattr(ctrl, "last_m_t_projected", float(m_t)),
                                    "hr_slope_bpm_per_min": getattr(ctrl, "last_hr_slope", 0.0),
                                    "hr_projected": getattr(ctrl, "last_hr_projected", np.nan),
                                    "lut_anticipation_active": bool(
                                        getattr(ctrl, "last_lut_anticipation_active", False)
                                    ),
                                })

    timing = pd.DataFrame(rows)
    timing.to_csv(out_dir / "activation_timing.csv", index=False)

    summary_rows = []
    if len(timing) > 0:
        group_cols = ["scenario", "manchester_pid", "seed", "exercise_day", "arm"]
        for key, grp in timing.groupby(group_cols):
            pre = grp[(grp["rel_min"] >= -90) & (grp["rel_min"] < 0)]
            first_30 = grp[(grp["rel_min"] >= 0) & (grp["rel_min"] < 30)]
            exercise = grp[(grp["rel_min"] >= 0) & (grp["rel_min"] < 90)]
            summary_rows.append({
                "scenario": key[0],
                "manchester_pid": key[1],
                "seed": key[2],
                "exercise_day": key[3],
                "arm": key[4],
                "first_m_gt_1_2_min": _first_crossing_min(exercise, 1.2),
                "first_m_gt_2_0_min": _first_crossing_min(exercise, 2.0),
                "mean_m_0_30": float(first_30["m_t"].mean()),
                "mean_m_exercise": float(exercise["m_t"].mean()),
                "mean_m_pre": float(pre["m_t"].mean()),
                "frac_pre_m_gt_1_2": float((pre["m_t"] > 1.2).mean()),
                "frac_anticipation_active_exercise": float(
                    exercise["lut_anticipation_active"].mean()
                ),
                "mean_hr_slope_exercise": float(exercise["hr_slope_bpm_per_min"].mean()),
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "activation_timing_summary.csv", index=False)
    if len(summary) > 0:
        pivot = summary.pivot_table(
            index=["scenario", "manchester_pid", "seed", "exercise_day"],
            columns="arm",
            values=["first_m_gt_1_2_min", "first_m_gt_2_0_min", "mean_m_0_30"],
            aggfunc="first",
        )
        pivot.columns = [f"{metric}_{arm}" for metric, arm in pivot.columns]
        pivot = pivot.reset_index()
        for metric in ("first_m_gt_1_2_min", "first_m_gt_2_0_min"):
            a = f"{metric}_patch_slope_projected"
            b = f"{metric}_patch"
            if a in pivot and b in pivot:
                pivot[f"{metric}_advance_min"] = pivot[b] - pivot[a]
        pivot.to_csv(out_dir / "activation_timing_pairwise.csv", index=False)

    if len(timing) > 0:
        for scenario_name in sorted(timing["scenario"].unique()):
            for pid in sorted(timing["manchester_pid"].unique()):
                sub = timing[
                    (timing["scenario"] == scenario_name)
                    & (timing["manchester_pid"] == pid)
                ]
                if len(sub) == 0:
                    continue
                med = sub.groupby(["arm", "rel_min"], as_index=False)[
                    ["m_t", "m_t_lut", "m_t_projected", "hr", "hr_slope_bpm_per_min"]
                ].median(numeric_only=True)
                fig, axes = plt.subplots(3, 1, figsize=(9.5, 7.5), sharex=True)
                colors = {"patch": "crimson", "patch_slope_projected": "mediumblue"}
                for arm in arms:
                    arm_med = med[med["arm"] == arm]
                    axes[0].plot(arm_med["rel_min"], arm_med["m_t"], label=arm, color=colors[arm])
                    axes[1].plot(arm_med["rel_min"], arm_med["hr"], label=arm, color=colors[arm])
                    axes[2].plot(
                        arm_med["rel_min"],
                        arm_med["hr_slope_bpm_per_min"],
                        label=arm,
                        color=colors[arm],
                    )
                axes[0].axhline(1.0, color="black", lw=0.8, ls="--")
                axes[0].axhline(1.2, color="gray", lw=0.8, ls=":")
                axes[0].axhline(2.0, color="gray", lw=0.8, ls=":")
                axes[0].set_ylabel("m_t")
                axes[1].set_ylabel("HR (bpm)")
                axes[2].set_ylabel("HR slope (bpm/min)")
                axes[2].set_xlabel("Minutes relative to exercise onset")
                for ax in axes:
                    ax.axvspan(0, 90, color="orange", alpha=0.10)
                    ax.grid(alpha=0.25)
                    ax.legend(fontsize=8)
                fig.suptitle(f"Spoor-C LUT activation timing - {scenario_name} PID {pid}")
                fig.tight_layout()
                fig.savefig(out_dir / f"activation_timing_{scenario_name}_pid{pid}.png", dpi=130)
                plt.close(fig)

    return timing, summary


def write_pairwise(metrics, out_dir):
    pairs = []
    index_cols = ["physiology_mode", "scenario", "manchester_pid", "cohort", "seed"]
    for key, grp in metrics.groupby(index_cols):
        base_rows = grp[grp["arm"] == "baseline"]
        if len(base_rows) == 0:
            continue
        base = base_rows.iloc[0]
        for arm in ("patch", "patch_slope_projected", "patch_exercise_guard", "placebo", "patch_t2", "patch_full_shape"):
            arm_rows = grp[grp["arm"] == arm]
            if len(arm_rows) == 0:
                continue
            row = arm_rows.iloc[0]
            pair = {
                "physiology_mode": key[0],
                "scenario": key[1],
                "manchester_pid": key[2],
                "cohort": key[3],
                "seed": key[4],
                "arm": arm,
            }
            diff_cols = [
                "tir", "tbr_70", "tbr_54", "tar_180", "auc70", "auc75",
                "lbgi", "total_insulin", "min_glu",
                "exercise_tbr_70", "exercise_auc75", "exercise_min",
                "early_recovery_tbr_70", "early_recovery_auc75", "early_recovery_min",
                "late_recovery_tbr_70", "late_recovery_auc75", "late_recovery_min",
                "exercise_mean_m_t", "exercise_max_m_t",
                "exercise_frac_m_gt_1_2", "exercise_frac_m_gt_2_0",
                "exercise_frac_m_ge_1_2",
                "exercise_frac_hr_above_robust", "exercise_frac_edge_hold",
                "exercise_frac_anticipation_active", "exercise_mean_hr_slope",
                "exercise_frac_exercise_guard_active",
                "exercise_mean_iob", "exercise_mean_bolusiob",
                "exercise_mean_basaliob", "exercise_mean_cob",
                "exercise_mean_smb_insulin_req", "exercise_total_smb_units",
                "exercise_frac_smb_enabled", "exercise_mean_smb_projection_bg",
                "exercise_mean_context_m", "exercise_mean_Vm0_factor",
                "exercise_max_Vm0_factor", "exercise_mean_Vmx_factor",
                "exercise_max_Vmx_factor", "exercise_mean_duration_gain",
                "exercise_max_duration_gain", "exercise_max_exercise_minutes",
                "exercise_total_insulin",
            ]
            for col in diff_cols:
                if col in row and col in base:
                    pair[f"{col}_diff"] = row[col] - base[col]
            pairs.append(pair)
    pairwise = pd.DataFrame(pairs)
    pairwise.to_csv(out_dir / "pairwise.csv", index=False)
    if len(pairwise) > 0:
        summary = pairwise.groupby(
            ["physiology_mode", "scenario", "manchester_pid", "arm"],
            as_index=False,
        ).mean(numeric_only=True)
        summary.to_csv(out_dir / "pairwise_summary.csv", index=False)
    return pairwise


def write_placebo_pairwise(metrics, out_dir):
    pairs = []
    index_cols = ["physiology_mode", "scenario", "manchester_pid", "cohort", "seed"]
    diff_cols = [
        "tir", "tbr_70", "tbr_54", "tar_180", "auc70", "auc75",
        "lbgi", "total_insulin", "min_glu",
        "exercise_tbr_70", "exercise_auc75", "exercise_min",
        "early_recovery_tbr_70", "early_recovery_auc75", "early_recovery_min",
        "late_recovery_tbr_70", "late_recovery_auc75", "late_recovery_min",
        "exercise_mean_m_t", "exercise_max_m_t",
        "exercise_frac_m_gt_1_2", "exercise_frac_m_ge_1_2", "exercise_frac_m_gt_2_0",
        "exercise_frac_hr_above_robust", "exercise_frac_edge_hold",
        "exercise_frac_anticipation_active", "exercise_frac_exercise_guard_active",
        "exercise_mean_iob", "exercise_mean_bolusiob",
        "exercise_mean_basaliob", "exercise_mean_cob",
        "exercise_mean_smb_insulin_req", "exercise_total_smb_units",
        "exercise_frac_smb_enabled", "exercise_mean_smb_projection_bg",
        "exercise_mean_context_m", "exercise_mean_Vm0_factor",
        "exercise_max_Vm0_factor", "exercise_mean_Vmx_factor",
        "exercise_max_Vmx_factor", "exercise_mean_duration_gain",
        "exercise_max_duration_gain", "exercise_max_exercise_minutes",
        "exercise_total_insulin",
    ]
    for key, grp in metrics.groupby(index_cols):
        placebo_rows = grp[grp["arm"] == "placebo"]
        if len(placebo_rows) == 0:
            continue
        placebo = placebo_rows.iloc[0]
        for arm in ("patch", "patch_slope_projected", "patch_exercise_guard", "patch_t2", "patch_full_shape"):
            arm_rows = grp[grp["arm"] == arm]
            if len(arm_rows) == 0:
                continue
            row = arm_rows.iloc[0]
            pair = {
                "physiology_mode": key[0],
                "scenario": key[1],
                "manchester_pid": key[2],
                "cohort": key[3],
                "seed": key[4],
                "arm": arm,
                "comparison": f"{arm}_minus_placebo",
            }
            for col in diff_cols:
                if col in row and col in placebo:
                    pair[f"{col}_diff"] = row[col] - placebo[col]
            pairs.append(pair)
    placebo_pairwise = pd.DataFrame(pairs)
    placebo_pairwise.to_csv(out_dir / "placebo_pairwise.csv", index=False)
    if len(placebo_pairwise) > 0:
        summary = placebo_pairwise.groupby(
            ["physiology_mode", "scenario", "manchester_pid", "arm", "comparison"],
            as_index=False,
        ).mean(numeric_only=True)
        summary.to_csv(out_dir / "placebo_pairwise_summary.csv", index=False)
    return placebo_pairwise


def write_guard_redundancy_gate(metrics, out_dir):
    rows = []
    index_cols = ["physiology_mode", "scenario", "manchester_pid", "cohort", "seed"]
    for key, grp in metrics.groupby(index_cols):
        by_arm = {str(row["arm"]): row for _, row in grp.iterrows()}
        if not {"patch", "placebo", "patch_exercise_guard"}.issubset(by_arm):
            continue
        patch = by_arm["patch"]
        placebo = by_arm["placebo"]
        guard = by_arm["patch_exercise_guard"]
        patch_minus_placebo = float(patch["exercise_auc75"] - placebo["exercise_auc75"])
        guard_minus_placebo = float(guard["exercise_auc75"] - placebo["exercise_auc75"])
        guard_minus_patch = float(guard["exercise_auc75"] - patch["exercise_auc75"])
        patch_benefit = -patch_minus_placebo
        guard_extra_benefit = patch["exercise_auc75"] - guard["exercise_auc75"]
        if patch_benefit > 1e-9:
            guard_extra_benefit_pct = max(0.0, guard_extra_benefit) / patch_benefit * 100.0
        else:
            guard_extra_benefit_pct = np.nan
        rows.append({
            "physiology_mode": key[0],
            "scenario": key[1],
            "manchester_pid": key[2],
            "cohort": key[3],
            "seed": key[4],
            "patch_minus_placebo_exercise_auc75": patch_minus_placebo,
            "guard_minus_placebo_exercise_auc75": guard_minus_placebo,
            "guard_minus_patch_exercise_auc75": guard_minus_patch,
            "guard_extra_benefit_pct_vs_patch_benefit": guard_extra_benefit_pct,
            "patch_beats_placebo": bool(patch_minus_placebo < 0),
            "guard_better_than_patch": bool(guard_minus_patch < 0),
            "patch_exercise_frac_m_ge_1_2": float(patch.get("exercise_frac_m_ge_1_2", np.nan)),
            "patch_exercise_total_smb_units": float(patch.get("exercise_total_smb_units", np.nan)),
            "placebo_exercise_total_smb_units": float(placebo.get("exercise_total_smb_units", np.nan)),
            "guard_exercise_total_smb_units": float(guard.get("exercise_total_smb_units", np.nan)),
            "patch_exercise_mean_smb_insulin_req": float(patch.get("exercise_mean_smb_insulin_req", np.nan)),
            "placebo_exercise_mean_smb_insulin_req": float(placebo.get("exercise_mean_smb_insulin_req", np.nan)),
            "guard_exercise_mean_smb_insulin_req": float(guard.get("exercise_mean_smb_insulin_req", np.nan)),
            "patch_exercise_total_insulin": float(patch.get("exercise_total_insulin", np.nan)),
            "placebo_exercise_total_insulin": float(placebo.get("exercise_total_insulin", np.nan)),
            "guard_exercise_total_insulin": float(guard.get("exercise_total_insulin", np.nan)),
        })
    gate = pd.DataFrame(rows)
    gate.to_csv(out_dir / "guard_redundancy_gate.csv", index=False)
    if len(gate) > 0:
        summary = (
            gate.groupby(["physiology_mode", "scenario", "manchester_pid"], as_index=False)
            .mean(numeric_only=True)
        )
        summary.to_csv(out_dir / "guard_redundancy_gate_summary.csv", index=False)
    return gate


def run_one_task(task):
    t_arm = time.time()
    scenario = get_scenario(task["scenario"])
    pid = task["pid"]
    cohort = task["cohort"]
    seed = task["seed"]
    arm = task["arm"]
    physiology_name = task["physiology_mode"]
    out_dir = Path(task["out_dir"])
    sens_mult = task["sens_mult"]
    start = task["start"]
    hr_rest = patient_hr_rest(pid)

    hr, eda = build_7day_hr_eda(start, seed, scenario, hr_rest)
    meals = build_7day_meals(start, seed, scenario)
    df = run_arm(
        pid, cohort, seed, scenario.name, arm, hr, eda, meals,
        start, sens_mult, hr_rest, physiology_name,
    )
    metrics = compute_metrics(df, scenario)
    df.to_csv(trajectory_path(out_dir, scenario.name, physiology_name, pid, cohort, seed, arm))
    physiology = get_physiology_mode(physiology_name)
    row = {
        "physiology_mode": physiology_name,
        "scenario": scenario.name,
        "manchester_pid": pid,
        "cohort": cohort,
        "seed": seed,
        "arm": arm,
        "sens_mult": sens_mult,
        "hr_rest": hr_rest,
        "scenario_peak_delta": scenario.peak_delta,
        "plant_vm0_exponent": physiology.vm0_exponent,
        "plant_vmx_exponent": physiology.vmx_exponent,
        "plant_modulate_p2u": physiology.modulate_p2u,
        "plant_vmx_duration_gain_per_hour": physiology.vmx_duration_gain_per_hour,
        "plant_vmx_duration_gain_cap": physiology.vmx_duration_gain_cap,
        "plant_vmx_decay_half_life_min": physiology.vmx_decay_half_life_min,
        "plant_vmx_factor_cap": physiology.vmx_factor_cap,
        **metrics,
    }
    return row, time.time() - t_arm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing-assay", action="store_true",
                        help="Predictor-only LUT activation assay; no SimGlucose trajectories")
    parser.add_argument("--smoke", action="store_true",
                        help="1 Manchester PID x 1 cohort x 1 seed x relevant arms")
    parser.add_argument("--pid", default=None,
                        help="Manchester PID subset, comma-separated. Default: all; smoke: 2307")
    parser.add_argument("--cohort", default=None,
                        help="SimGlucose cohort subset. Default: all; smoke: adolescent#003")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed subset. Default: all; smoke: 1")
    parser.add_argument("--seeds", default=None,
                        help="Seed subset, comma-separated. Overrides --seed.")
    parser.add_argument("--sens-mult", type=float, default=SENS_MULTIPLIER)
    parser.add_argument("--scenario", default=None,
                        help="Scenario subset, comma-separated. Default: both post-breakfast scenarios")
    parser.add_argument("--physiology-mode", default="current_vm0_dominant",
                        help="Physiology mode subset, comma-separated. Use 'all' for all modes.")
    parser.add_argument("--include-full-shape", action="store_true",
                        help="Appendix sensitivity arm for 2304/2305 only")
    parser.add_argument("--arms", default=None,
                        help="Explicit arm subset, comma-separated. Overrides default arms_for_pid.")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Parallel worker processes for independent trajectories")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.pid:
        pids = [p.strip() for p in args.pid.split(",") if p.strip()]
    else:
        pids = ["2307"] if args.smoke else MANCHESTER_PIDS
    cohorts = [args.cohort] if args.cohort else ([COHORTS[0]] if args.smoke else COHORTS)
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [args.seed] if args.seed is not None else ([SEEDS[0]] if args.smoke else SEEDS)
    if args.scenario:
        scenarios = [get_scenario(s.strip()) for s in args.scenario.split(",") if s.strip()]
    else:
        scenarios = [SCENARIOS["post_breakfast_moderate"], SCENARIOS["post_breakfast_vigorous"]]
    if args.physiology_mode == "all":
        physiology_modes = list(PHYSIOLOGY_MODES.values())
    else:
        physiology_modes = [
            get_physiology_mode(m.strip()) for m in args.physiology_mode.split(",") if m.strip()
        ]

    start = datetime(2025, 1, 6, 0, 0)
    out_dir = Path(__file__).parent / "results" / (
        args.out or ("spoorc_lut_protocol_v2_pilot" if args.smoke else "spoorc_lut_protocol_v2")
    )
    out_dir.mkdir(exist_ok=True, parents=True)

    if args.timing_assay:
        timing, summary = run_timing_assay(out_dir, pids, seeds, scenarios, args.sens_mult)
        print(f"\nSaved activation timing: {out_dir / 'activation_timing.csv'}")
        print(f"Saved activation summary: {out_dir / 'activation_timing_summary.csv'}")
        if len(summary) > 0:
            print("\nMedian activation summary:")
            print(
                summary.groupby(["scenario", "manchester_pid", "arm"])
                [["first_m_gt_1_2_min", "first_m_gt_2_0_min", "mean_m_0_30", "frac_pre_m_gt_1_2"]]
                .median(numeric_only=True)
                .round(3)
            )
        print(f"Timing assay rows: {len(timing)}")
        return

    tasks = []
    for physiology in physiology_modes:
        for scenario in scenarios:
            for pid in pids:
                for cohort in cohorts:
                    for seed in seeds:
                        arms = (
                            [a.strip() for a in args.arms.split(",") if a.strip()]
                            if args.arms else
                            arms_for_pid(pid, include_full_shape=args.include_full_shape)
                        )
                        for arm in arms:
                            tasks.append({
                                "physiology_mode": physiology.name,
                                "scenario": scenario.name,
                                "pid": pid,
                                "cohort": cohort,
                                "seed": seed,
                                "arm": arm,
                                "sens_mult": args.sens_mult,
                                "start": start,
                                "out_dir": str(out_dir),
                            })

    rows = []
    total = len(tasks)
    done = 0
    t0 = time.time()

    def _print_progress(row, elapsed):
        print(
            f"[{done:3d}/{total}] {row['physiology_mode']:<25} {row['scenario']:<24} "
            f"pid={row['manchester_pid']} "
            f"{row['cohort']:>16} s={row['seed']:>2} {row['arm']:>16} "
            f"TIR={row['tir']:>5.1f} TBR={row['tbr_70']:>5.1f} "
            f"AUC75={row['auc75']:>8.1f} ex_m={row['exercise_mean_m_t']:>4.2f} "
            f"Vmx={row['exercise_mean_Vmx_factor']:>4.2f} {elapsed:>5.1f}s"
        )

    if args.jobs <= 1:
        for task in tasks:
            row, elapsed = run_one_task(task)
            rows.append(row)
            done += 1
            _print_progress(row, elapsed)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_one_task, task) for task in tasks]
            for fut in as_completed(futures):
                row, elapsed = fut.result()
                rows.append(row)
                done += 1
                _print_progress(row, elapsed)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    pairwise = write_pairwise(metrics_df, out_dir)
    placebo_pairwise = write_placebo_pairwise(metrics_df, out_dir)
    guard_gate = write_guard_redundancy_gate(metrics_df, out_dir)
    plot_exercise_profiles(out_dir, metrics_df)
    plot_summary_bars(out_dir, pairwise)

    print(f"\nSaved metrics: {out_dir / 'metrics.csv'}")
    print(f"Saved pairwise: {out_dir / 'pairwise.csv'}")
    print(f"Saved placebo pairwise: {out_dir / 'placebo_pairwise.csv'}")
    print(f"Saved guard redundancy gate: {out_dir / 'guard_redundancy_gate.csv'}")
    print(f"Total time: {(time.time() - t0) / 60:.1f} min for {done} runs")
    if len(pairwise) > 0:
        print("\nPairwise mean differences (arm - baseline):")
        print(
            pairwise.groupby(["physiology_mode", "scenario", "manchester_pid", "arm"])
            .mean(numeric_only=True)
            .round(3)
        )
    if len(placebo_pairwise) > 0:
        print("\nPairwise mean differences (arm - placebo):")
        print(
            placebo_pairwise.groupby(["physiology_mode", "scenario", "manchester_pid", "arm"])
            .mean(numeric_only=True)
            .round(3)
        )
    if len(guard_gate) > 0:
        print("\nGuard redundancy gate summary:")
        print(
            guard_gate.groupby(["physiology_mode", "scenario", "manchester_pid"])
            .mean(numeric_only=True)
            .round(3)
        )


if __name__ == "__main__":
    main()
