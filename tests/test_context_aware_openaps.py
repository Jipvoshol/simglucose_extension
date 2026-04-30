import pickle
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from simglucose_ctx.context_aware_openaps import ContextAwareOpenAPSController
from simglucose_ctx.context import ContextConfig
from simglucose_ctx.sim_hook import apply_vmx_multiplier, apply_vm0_multiplier

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from thesis_spoorc_lut_controller_demo import (  # noqa: E402
    STEP_MIN,
    TOTAL_STEPS,
    build_7day_hr_eda,
    build_7day_meals,
    compute_metrics,
    ctx_cfg,
    get_physiology_mode,
    get_scenario,
    make_controller,
    patient_hr_rest,
    precompute_placebo_m,
    write_placebo_pairwise,
)


class DummyPipeline:
    def predict(self, X):
        return np.asarray([1.23])


class DummyParams:
    def __init__(self):
        self.Vmx = 10.0
        self.Vm0 = 5.0
        self.p2u = 0.1


class DummyPatient:
    def __init__(self):
        self._params = DummyParams()


def _profile():
    return {
        "current_basal": 1.0,
        "sens": 50.0,
        "min_bg": 90.0,
        "max_bg": 120.0,
        "max_iob": 3.0,
        "bolus_increment": 0.1,
        "temp_basal_duration_min": 30,
        "max_safe_basal": 4.0,
    }


def _write_lut(path, curve=None, robust=None):
    np.savez(
        path,
        hr_grid=np.asarray([50.0, 60.0, 70.0]),
        m_curve=np.asarray(curve if curve is not None else [0.8, 1.4, 2.0]),
        robust_mask=np.asarray(robust if robust is not None else [True, True, True]),
        deploy_clip_lo=0.6,
        deploy_clip_hi=2.5,
    )


def test_spoorab_pickle_loading_still_predicts(tmp_path):
    model_path = tmp_path / "patch.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "pipeline": DummyPipeline(),
                "features": ["HR_norm", "EDA_norm"],
                "feature_semantics": {"HR_norm": "relative", "EDA_norm": "stress"},
            },
            f,
        )

    ctrl = ContextAwareOpenAPSController(_profile(), model_path=str(model_path))
    m_t = ctrl._predict_m_t(
        {"context_hr": 120.0, "context_eda": 30.0, "context_hr_is_relative": False},
        ctrl._now,
    )

    assert m_t == 1.23


def test_spoorc_lut_safe_upward_bounds_and_interpolates(tmp_path):
    lut_path = tmp_path / "lut_2307.npz"
    _write_lut(lut_path)

    ctrl = ContextAwareOpenAPSController(
        _profile(),
        model_path=str(lut_path),
        patch_version="spoorC_lut",
        lut_policy="safe_upward",
    )

    assert ctrl._predict_m_t({"context_hr": 55.0}, ctrl._now) == 1.1
    assert ctrl._predict_m_t({"context_hr": 50.0}, ctrl._now) == 1.0


def test_spoorc_lut_full_shape_allows_below_one(tmp_path):
    lut_path = tmp_path / "lut_2307.npz"
    _write_lut(lut_path)

    ctrl = ContextAwareOpenAPSController(
        _profile(),
        model_path=str(lut_path),
        patch_version="spoorC_lut",
        lut_policy="full_shape",
    )

    assert ctrl._predict_m_t({"context_hr": 50.0}, ctrl._now) == 0.8


def test_spoorc_lut_failsafe_missing_out_of_grid_or_nonrobust(tmp_path):
    lut_path = tmp_path / "lut_2307.npz"
    _write_lut(lut_path, robust=[True, False, True])

    ctrl = ContextAwareOpenAPSController(
        _profile(),
        model_path=str(lut_path),
        patch_version="spoorC_lut",
    )

    assert ctrl._predict_m_t({}, ctrl._now) == 1.0
    assert ctrl._predict_m_t({"context_hr": 49.0}, ctrl._now) == 1.0
    assert ctrl._predict_m_t({"context_hr": 55.0}, ctrl._now) == 1.0
    assert ctrl._predict_m_t({"context_hr": 75.0}, ctrl._now) == 2.0


def test_spoorc_lut_high_hr_uses_max_robust_m(tmp_path):
    lut_path = tmp_path / "lut_2307.npz"
    _write_lut(lut_path, curve=[0.9, 1.2, 2.3], robust=[True, True, False])

    ctrl = ContextAwareOpenAPSController(
        _profile(),
        model_path=str(lut_path),
        patch_version="spoorC_lut",
    )

    assert ctrl._predict_m_t({"context_hr": 70.0}, ctrl._now) == 1.2
    assert ctrl._predict_m_t({"context_hr": 180.0}, ctrl._now) == 1.2


def test_spoorc_lut_slope_anticipation_projects_rising_hr(tmp_path):
    lut_path = tmp_path / "lut_2307.npz"
    _write_lut(lut_path, curve=[1.0, 1.0, 2.0])
    t0 = pd.Timestamp(datetime(2025, 1, 1, 9, 0))

    ctrl = ContextAwareOpenAPSController(
        _profile(),
        model_path=str(lut_path),
        patch_version="spoorC_lut",
        lut_anticipation="slope_projected",
        lut_lookahead_min=9.0,
        lut_slope_min_bpm_per_min=3.0,
    )

    assert ctrl._predict_m_t({"context_hr": 50.0}, t0.to_pydatetime()) == 1.0
    m_t = ctrl._predict_m_t({"context_hr": 59.5}, (t0 + pd.Timedelta(minutes=3)).to_pydatetime())

    assert m_t == 2.0
    assert ctrl.last_m_t_lut == 1.0
    assert ctrl.last_m_t_projected == 2.0
    assert ctrl.last_lut_anticipation_active is True
    assert ctrl.last_hr_slope > 3.0


def test_spoorc_lut_slope_anticipation_ignores_flat_hr_and_reset_clears_history(tmp_path):
    lut_path = tmp_path / "lut_2307.npz"
    _write_lut(lut_path, curve=[1.0, 1.0, 2.0])
    t0 = pd.Timestamp(datetime(2025, 1, 1, 9, 0))

    ctrl = ContextAwareOpenAPSController(
        _profile(),
        model_path=str(lut_path),
        patch_version="spoorC_lut",
        lut_anticipation="slope_projected",
        lut_lookahead_min=9.0,
        lut_slope_min_bpm_per_min=3.0,
    )

    assert ctrl._predict_m_t({"context_hr": 50.0}, t0.to_pydatetime()) == 1.0
    assert ctrl._predict_m_t({"context_hr": 52.0}, (t0 + pd.Timedelta(minutes=3)).to_pydatetime()) == 1.0
    assert ctrl.last_lut_anticipation_active is False

    ctrl.reset()
    assert ctrl._predict_m_t({"context_hr": 59.5}, (t0 + pd.Timedelta(minutes=6)).to_pydatetime()) == 1.0
    assert ctrl.last_hr_slope == 0.0


def test_tier2_uses_stress_residual_when_metadata_present(tmp_path):
    lut_path = tmp_path / "lut_2304.npz"
    _write_lut(lut_path, curve=[1.4, 1.4, 1.4])

    ctrl = ContextAwareOpenAPSController(
        _profile(),
        model_path=str(lut_path),
        patch_version="spoorC_lut",
        lut_policy="full_shape",
        tier2_beta=-0.01,
        tier2_stress_hr_grid=[50.0, 60.0, 70.0],
        tier2_expected_stress=[30.0, 30.0, 30.0],
    )

    m_t = ctrl._predict_m_t({"context_hr": 60.0, "context_eda": 50.0}, ctrl._now)

    assert abs(m_t - 1.2) < 1e-9


def test_tier2_falls_back_to_tier1_without_metadata(tmp_path):
    lut_path = tmp_path / "lut_2304.npz"
    _write_lut(lut_path, curve=[1.4, 1.4, 1.4])

    ctrl = ContextAwareOpenAPSController(
        _profile(),
        model_path=str(lut_path),
        patch_version="spoorC_lut",
        lut_policy="full_shape",
        tier2_beta=-0.01,
    )

    assert ctrl._predict_m_t({"context_hr": 60.0, "context_eda": 50.0}, ctrl._now) == 1.4


def test_override_placebo_path_and_policy_restore_baseline_isf():
    ctrl = ContextAwareOpenAPSController(_profile())
    action = ctrl.policy(120.0, reward=0.0, done=False, context_m_override=1.6)

    assert action is not None
    assert ctrl.last_m_t == 1.6
    assert ctrl.last_patched_isf == 80.0
    assert ctrl.profile["sens"] == 50.0


def test_exercise_guard_temporarily_suppresses_smb_and_restores_profile():
    p = _profile()
    p.update({
        "enableSMB_with_COB": True,
        "enableSMB_after_carbs": True,
        "enableSMB_high_bg": True,
        "max_safe_basal": 4.0,
    })
    ctrl = ContextAwareOpenAPSController(
        p,
        patch_version="spoorC_lut",
        exercise_guard="smb_basal_cap",
        exercise_guard_threshold=1.2,
    )
    action = ctrl.policy(180.0, reward=0.0, done=False, context_m_override=1.6, meal=1.0)

    assert action is not None
    assert ctrl.last_exercise_guard_active is True
    assert ctrl.last_m_t == 1.6
    assert ctrl.profile["sens"] == 50.0
    assert ctrl.profile["enableSMB_with_COB"] is True
    assert ctrl.profile["enableSMB_after_carbs"] is True
    assert ctrl.profile["enableSMB_high_bg"] is True
    assert ctrl.profile["max_safe_basal"] == 4.0


def test_spoorc_protocol_uses_patient_rhr_and_ramped_hr():
    scenario = get_scenario("post_breakfast_moderate")
    hr_rest = patient_hr_rest("2307")
    hr, eda = build_7day_hr_eda(datetime(2025, 1, 6), 1, scenario, hr_rest)

    assert hr_rest == 59.0
    assert scenario.exercise_start_min == 9 * 60
    assert scenario.exercise_end_min == 10 * 60 + 30

    onset = scenario.onset_step
    ramp_end = onset + scenario.ramp_steps
    plateau_end = ramp_end + scenario.plateau_steps
    before = float(hr.iloc[onset - 1])
    early_ramp = float(hr.iloc[onset + 1])
    plateau = float(hr.iloc[ramp_end + 1])
    cooldown = float(hr.iloc[plateau_end + scenario.cooldown_steps - 1])

    assert plateau > before + 35.0
    assert plateau > early_ramp
    assert cooldown < plateau
    assert float(eda.iloc[ramp_end + 1]) > float(eda.iloc[onset - 1])


def test_spoorc_scenario_meal_variants_reduce_breakfast_iob_pressure():
    start = datetime(2025, 1, 6)
    base = get_scenario("post_breakfast_vigorous")
    half = get_scenario("post_breakfast_vigorous_halfbreakfast")
    fasted = get_scenario("fasted_morning_vigorous")
    late = get_scenario("late_post_breakfast_vigorous")

    base_meals = build_7day_meals(start, 1, base)
    half_meals = build_7day_meals(start, 1, half)
    fasted_meals = build_7day_meals(start, 1, fasted)

    day0 = start
    base_breakfast = [c for t, c in base_meals if day0 + pd.Timedelta(hours=7) <= t <= day0 + pd.Timedelta(hours=8)]
    half_breakfast = [c for t, c in half_meals if day0 + pd.Timedelta(hours=7) <= t <= day0 + pd.Timedelta(hours=8)]
    fasted_breakfast = [c for t, c in fasted_meals if day0 + pd.Timedelta(hours=7) <= t <= day0 + pd.Timedelta(hours=8)]

    assert half.breakfast_carb_scale == 0.5
    assert half_breakfast[0] < base_breakfast[0]
    assert fasted_breakfast == []
    assert late.exercise_start_min == 11 * 60


def test_spoorc_metrics_include_recovery_windows_and_activation():
    idx = pd.date_range(datetime(2025, 1, 6), periods=TOTAL_STEPS, freq=f"{int(STEP_MIN)}min")
    df = pd.DataFrame(index=idx)
    df["glucose"] = 100.0
    df["basal"] = 1.0
    df["bolus"] = 0.0
    df["insulin"] = 1.0
    df["m_t"] = 1.0
    df["hr"] = 70.0
    df["hr_above_robust"] = False
    df["edge_hold"] = False

    scenario = get_scenario("post_breakfast_vigorous")
    for d in [0, 2, 4]:
        day = idx[0].normalize() + pd.Timedelta(days=d)
        exercise = (df.index >= day + pd.Timedelta(hours=9)) & (df.index < day + pd.Timedelta(hours=10, minutes=30))
        early_recovery = (df.index >= day + pd.Timedelta(hours=10, minutes=30)) & (df.index < day + pd.Timedelta(hours=12))
        late_recovery = (df.index >= day + pd.Timedelta(hours=12)) & (df.index < day + pd.Timedelta(hours=16, minutes=30))
        df.loc[exercise, "glucose"] = 65.0
        df.loc[exercise, "m_t"] = 2.2
        df.loc[exercise, "hr_above_robust"] = True
        df.loc[exercise, "edge_hold"] = True
        df.loc[early_recovery, "glucose"] = 72.0
        df.loc[late_recovery, "glucose"] = 85.0

    metrics = compute_metrics(df, scenario)

    assert metrics["pre_exercise_tbr_70"] == 0.0
    assert metrics["exercise_tbr_70"] == 100.0
    assert metrics["exercise_frac_m_gt_2_0"] == 1.0
    assert metrics["exercise_frac_m_ge_1_2"] == 1.0
    assert metrics["exercise_frac_edge_hold"] == 1.0
    assert metrics["early_recovery_auc75"] > 0.0
    assert metrics["late_recovery_tbr_70"] == 0.0


def test_placebo_pairwise_reports_arm_minus_placebo(tmp_path):
    metrics = pd.DataFrame([
        {
            "physiology_mode": "balanced_duration_decay",
            "scenario": "post_breakfast_vigorous",
            "manchester_pid": "2307",
            "cohort": "adolescent#003",
            "seed": 1,
            "arm": "placebo",
            "exercise_auc75": 100.0,
            "exercise_frac_m_ge_1_2": 0.1,
            "exercise_total_insulin": 4.0,
        },
        {
            "physiology_mode": "balanced_duration_decay",
            "scenario": "post_breakfast_vigorous",
            "manchester_pid": "2307",
            "cohort": "adolescent#003",
            "seed": 1,
            "arm": "patch_exercise_guard",
            "exercise_auc75": 70.0,
            "exercise_frac_m_ge_1_2": 0.8,
            "exercise_total_insulin": 2.0,
        },
    ])

    placebo_pairwise = write_placebo_pairwise(metrics, tmp_path)
    row = placebo_pairwise.iloc[0]

    assert row["comparison"] == "patch_exercise_guard_minus_placebo"
    assert row["exercise_auc75_diff"] == -30.0
    assert abs(row["exercise_frac_m_ge_1_2_diff"] - 0.7) < 1e-9
    assert (tmp_path / "placebo_pairwise.csv").exists()


def test_physiology_modes_keep_default_backward_compatible():
    current = get_physiology_mode("current_vm0_dominant")
    cfg = ctx_cfg(60.0, current)

    assert cfg.vm0_exponent_exercise == 3.5
    assert cfg.vmx_exponent_exercise == 0.2
    assert cfg.vmx_duration_gain_per_hour == 0.0
    assert cfg.vmx_duration_gain_cap == 1.0
    assert cfg.vmx_decay_half_life_min is None
    assert current.modulate_p2u is True


def test_balanced_physiology_modes_configure_vmx_without_p2u():
    static = get_physiology_mode("balanced_static")
    duration = get_physiology_mode("balanced_duration_decay")
    cfg_static = ctx_cfg(60.0, static)
    cfg_duration = ctx_cfg(60.0, duration)

    assert cfg_static.vm0_exponent_exercise == 1.8
    assert cfg_static.vmx_exponent_exercise == 1.35
    assert static.modulate_p2u is False
    assert cfg_duration.vmx_duration_gain_per_hour == 0.20
    assert cfg_duration.vmx_duration_gain_cap == 1.30
    assert cfg_duration.vmx_decay_half_life_min == 90.0
    assert cfg_duration.vmx_factor_cap == 3.5
    assert duration.modulate_p2u is False


def test_vmx_duration_gain_rises_caps_and_decays():
    patient = DummyPatient()
    cfg = ContextConfig(
        vmx_exponent_exercise=1.35,
        vmx_duration_gain_per_hour=0.20,
        vmx_duration_gain_cap=1.30,
        vmx_decay_half_life_min=90.0,
        vmx_factor_cap=3.5,
    )
    ctx = SimpleNamespace(cfg=cfg)
    t0 = pd.Timestamp(datetime(2025, 1, 1, 9, 0))

    f0 = apply_vmx_multiplier(patient, t0, ctx, 10.0, m_value=2.0) / 10.0
    f60 = apply_vmx_multiplier(patient, t0 + pd.Timedelta(minutes=60), ctx, 10.0, m_value=2.0) / 10.0
    f180 = apply_vmx_multiplier(patient, t0 + pd.Timedelta(minutes=180), ctx, 10.0, m_value=2.0) / 10.0
    capped_gain = patient._ctx_vmx_state["duration_gain"]
    f270 = apply_vmx_multiplier(patient, t0 + pd.Timedelta(minutes=270), ctx, 10.0, m_value=1.0) / 10.0

    assert f60 > f0
    assert f180 <= 3.5
    assert capped_gain == 1.30
    assert 1.0 < f270 < f180


def test_vmx_stress_does_not_apply_positive_duration_decay():
    patient = DummyPatient()
    cfg = ContextConfig(
        vmx_exponent_exercise=1.35,
        vmx_duration_gain_per_hour=0.20,
        vmx_duration_gain_cap=1.30,
        vmx_decay_half_life_min=90.0,
    )
    ctx = SimpleNamespace(cfg=cfg)
    t0 = pd.Timestamp(datetime(2025, 1, 1, 9, 0))

    apply_vmx_multiplier(patient, t0, ctx, 10.0, m_value=2.0)
    stressed = apply_vmx_multiplier(patient, t0 + pd.Timedelta(minutes=30), ctx, 10.0, m_value=0.8)

    assert abs(stressed / 10.0 - 0.8) < 1e-9
    assert patient._ctx_vmx_state["exercise_minutes"] == 0.0
    assert patient._ctx_vmx_state["tail_factor"] == 1.0


def test_vm0_balanced_exponent_is_lower_than_current():
    patient_current = DummyPatient()
    patient_balanced = DummyPatient()
    t0 = pd.Timestamp(datetime(2025, 1, 1, 9, 0))
    current_ctx = SimpleNamespace(cfg=ContextConfig(vm0_exponent_exercise=3.5))
    balanced_ctx = SimpleNamespace(cfg=ContextConfig(vm0_exponent_exercise=1.8))

    current = apply_vm0_multiplier(patient_current, t0, current_ctx, 5.0, m_value=2.0) / 5.0
    balanced = apply_vm0_multiplier(patient_balanced, t0, balanced_ctx, 5.0, m_value=2.0) / 5.0

    assert current > balanced
    assert abs(balanced - (2.0 ** 1.8)) < 1e-9


def test_placebo_precompute_is_deterministic_and_keeps_marginal_distribution():
    scenario = get_scenario("post_breakfast_moderate")
    hr_rest = patient_hr_rest("2307")
    hr, eda = build_7day_hr_eda(datetime(2025, 1, 6), 1, scenario, hr_rest)
    hr = hr.iloc[:120]
    eda = eda.iloc[:120]

    m1 = precompute_placebo_m("2307", hr, eda, 2308, 0.75, hr_rest=hr_rest)
    m2 = precompute_placebo_m("2307", hr, eda, 2308, 0.75, hr_rest=hr_rest)
    m3 = precompute_placebo_m("2307", hr, eda, 2309, 0.75, hr_rest=hr_rest)

    assert np.array_equal(m1, m2)
    assert np.allclose(np.sort(m1), np.sort(m3))
    assert not np.array_equal(m1, m3)


def test_make_controller_can_build_slope_projected_arm():
    ctrl = make_controller("2307", "patch_slope_projected", 0.75, hr_rest=59.0)

    assert ctrl.patch_version == "spoorC_lut"
    assert ctrl.lut_anticipation == "slope_projected"


def test_make_controller_can_build_exercise_guard_arm():
    ctrl = make_controller("2307", "patch_exercise_guard", 0.75, hr_rest=59.0)

    assert ctrl.patch_version == "spoorC_lut"
    assert ctrl.exercise_guard == "smb_basal_cap"
