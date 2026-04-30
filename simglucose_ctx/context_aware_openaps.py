import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from copy import deepcopy

from simglucose.controller.base import Action
from .openaps_controller import OpenAPSController


class ContextAwareOpenAPSController(OpenAPSController):
    """
    Wraps the OpenAPSController to proactively modulate the ISF profile
    based on a pre-trained XGBoost model (m(t) predictor).

    FEATURE ALIGNMENT NOTES:
    The XGBoost model was trained on Manchester T1D-UOM data where:
      - HR_norm = heart_rate - resting_heart_rate (relative, in bpm)
      - EDA_norm = stress_level (Garmin derived, 0-100 scale)
    
    The controller must convert raw biometric inputs to match these semantics.
    Feeding absolute HR (e.g. 130 bpm) as HR_norm would cause massive
    extrapolation errors since the model was trained on HR_norm ∈ [-30, +50].
    """

    def __init__(self, profile: dict, model_path: str = None,
                 placebo_mode: bool = False, hr_rest: float = 65.0,
                 patch_version: str = None, lut_policy: str = None,
                 tier2_beta: float = None,
                 tier2_stress_hr_grid=None, tier2_expected_stress=None,
                 lut_anticipation: str = None,
                 lut_lookahead_min: float = None,
                 lut_slope_min_bpm_per_min: float = None,
                 lut_history_min: float = None,
                 exercise_guard: str = None,
                 exercise_guard_threshold: float = None):
        super().__init__(deepcopy(profile))

        self.placebo_mode = placebo_mode
        self.patch_version = patch_version or os.getenv("SRL_PATCH_VERSION", "spoorAB_xgb")
        self.lut_policy = lut_policy or os.getenv("SRL_LUT_POLICY", "safe_upward")
        self.lut_anticipation = (
            lut_anticipation
            if lut_anticipation is not None
            else os.getenv("SRL_LUT_ANTICIPATION", "off")
        )
        self.lut_lookahead_min = float(
            lut_lookahead_min
            if lut_lookahead_min is not None
            else os.getenv("SRL_LUT_LOOKAHEAD_MIN", 9.0)
        )
        self.lut_slope_min_bpm_per_min = float(
            lut_slope_min_bpm_per_min
            if lut_slope_min_bpm_per_min is not None
            else os.getenv("SRL_LUT_SLOPE_MIN_BPM_PER_MIN", 3.0)
        )
        self.lut_history_min = float(
            lut_history_min
            if lut_history_min is not None
            else os.getenv("SRL_LUT_HISTORY_MIN", 12.0)
        )
        self.exercise_guard = (
            exercise_guard
            if exercise_guard is not None
            else os.getenv("SRL_EXERCISE_GUARD", "off")
        )
        self.exercise_guard_threshold = float(
            exercise_guard_threshold
            if exercise_guard_threshold is not None
            else os.getenv("SRL_EXERCISE_GUARD_THRESHOLD", 1.2)
        )
        self.model_pipeline = None
        self.features = None
        self.feature_semantics = None
        self.lut_data = None
        self.tier2_beta = tier2_beta
        self.tier2_stress_hr_grid = (
            np.asarray(tier2_stress_hr_grid, dtype=float)
            if tier2_stress_hr_grid is not None else None
        )
        self.tier2_expected_stress = (
            np.asarray(tier2_expected_stress, dtype=float)
            if tier2_expected_stress is not None else None
        )

        # Resting heart rate for normalizing absolute HR → HR_norm
        self.hr_rest = hr_rest

        # Buffer for computing continuous features (HR 30m, EDA 60m)
        self._history = []
        self._lut_hr_history = []

        # Store initial Baseline ISF from profile so we can always revert
        self.isf_baseline = float(self.profile.get('sens', 50.0))
        self._reset_lut_prediction_diagnostics()

        # Load the patient's predictor/patch artifact. Default remains the
        # historical Spoor-A/B XGBoost pickle for backwards compatibility.
        if model_path is not None:
            if self.patch_version == "spoorC_lut":
                self._load_spoorc_lut(model_path)
            else:
                self._load_spoorab_xgb(model_path)

    def reset(self):
        super().reset()
        self._history.clear()
        self._lut_hr_history.clear()
        self._reset_lut_prediction_diagnostics()
        return None

    def _reset_lut_prediction_diagnostics(self) -> None:
        self.last_m_t_lut = 1.0
        self.last_m_t_projected = 1.0
        self.last_m_t_final = 1.0
        self.last_hr_slope = 0.0
        self.last_hr_projected = np.nan
        self.last_lut_anticipation_active = False
        self.last_lut_anticipation_mode = self.lut_anticipation
        self.last_exercise_guard_active = False
        self.last_exercise_guard_mode = self.exercise_guard

    def _load_spoorab_xgb(self, model_path: str) -> None:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model_pipeline = data['pipeline']
            self.features = data['features']
            self.feature_semantics = data.get('feature_semantics', None)

            if self.feature_semantics:
                print(f"  [ISF-Patch] Loaded model with semantics: "
                      f"HR={self.feature_semantics.get('HR_norm','?')}, "
                      f"EDA={self.feature_semantics.get('EDA_norm','?')}")

    def _load_spoorc_lut(self, model_path: str) -> None:
        data = np.load(model_path, allow_pickle=True)
        self.lut_data = {
            "hr_grid": np.asarray(data["hr_grid"], dtype=float),
            "m_curve": np.asarray(data["m_curve"], dtype=float),
            "robust_mask": np.asarray(data["robust_mask"], dtype=bool),
            "clip_lo": float(data["deploy_clip_lo"]) if "deploy_clip_lo" in data.files else 0.6,
            "clip_hi": float(data["deploy_clip_hi"]) if "deploy_clip_hi" in data.files else 2.5,
        }

        if "tier2_beta" in data and self.tier2_beta is None:
            self.tier2_beta = float(data["tier2_beta"])
        if "tier2_stress_hr_grid" in data and self.tier2_stress_hr_grid is None:
            self.tier2_stress_hr_grid = np.asarray(data["tier2_stress_hr_grid"], dtype=float)
        if "tier2_expected_stress" in data and self.tier2_expected_stress is None:
            self.tier2_expected_stress = np.asarray(data["tier2_expected_stress"], dtype=float)

        print(f"  [ISF-Patch] Loaded Spoor-C LUT ({self.lut_policy}) from {model_path}")

    def _compute_rolling_features(self, current_time: datetime,
                                  hr_norm: float, stress: float) -> dict:
        """
        Maintains an internal buffer of the last 60 minutes to calculate
        the delayed AMPK-pathway effects (30m HR avg, 60m stress avg).
        
        Parameters use the NORMALIZED feature space:
          hr_norm: HR relative to resting (bpm), NOT absolute HR
          stress: stress_level (0-100 scale), NOT raw EDA µS
        """
        self._history.append({'time': current_time, 'hr_norm': hr_norm, 'stress': stress})

        # Keep only the last 65 minutes
        cutoff = current_time - pd.Timedelta(minutes=65)
        self._history = [x for x in self._history if x['time'] >= cutoff]

        df = pd.DataFrame(self._history)

        # 30-min rolling mean of HR_norm
        df_30 = df[df['time'] >= current_time - pd.Timedelta(minutes=30)]
        hr_roll_30m = df_30['hr_norm'].mean() if len(df_30) > 0 else hr_norm

        # 60-min rolling mean of stress
        df_60 = df[df['time'] >= current_time - pd.Timedelta(minutes=60)]
        eda_roll_60m = df_60['stress'].mean() if len(df_60) > 0 else stress

        return {
            'hr_roll_30m': hr_roll_30m,
            'eda_roll_60m': eda_roll_60m
        }

    def _predict_m_t(self, info: dict, current_time: datetime) -> float:
        """
        Predict the contextual shift multiplier m(t).
        
        Returns 1.0 (no modulation) if no model is loaded or inputs are missing.

        CRITICAL FEATURE ALIGNMENT:
        The model expects:
          HR_norm  = heart_rate - resting_heart_rate (relative bpm, range ~[-30, +50])
          EDA_norm = stress_level (0-100 scale)
        
        The simulation/controller must provide:
          context_hr  = absolute heart rate in bpm  → we subtract self.hr_rest
          context_eda = stress level (0-100)        → passed through directly
        
        OR if context_hr is already relative (flagged by 'context_hr_is_relative'):
          context_hr = relative HR → used directly
        """
        if current_time is None:
            current_time = datetime.utcnow()

        override = info.get("context_m_override")
        if override is not None:
            m_override = self._clip_patch_m(float(override))
            self.last_m_t_lut = m_override
            self.last_m_t_projected = m_override
            self.last_m_t_final = m_override
            self.last_hr_slope = 0.0
            self.last_hr_projected = np.nan
            self.last_lut_anticipation_active = False
            self.last_lut_anticipation_mode = "override"
            return m_override

        if self.patch_version == "spoorC_lut":
            return self._predict_lut_m_t(info, current_time)

        hr_raw = info.get("context_hr")
        eda_raw = info.get("context_eda")

        if hr_raw is None or eda_raw is None or self.model_pipeline is None:
            return 1.0

        # FEATURE ALIGNMENT FIX:
        # Convert absolute HR to relative HR_norm (subtract resting HR)
        if info.get("context_hr_is_relative", False):
            hr_norm = float(hr_raw)
        else:
            hr_norm = float(hr_raw) - self.hr_rest

        # EDA_norm in the training data = stress_level (0-100 scale)
        # If deployment provides raw EDA in µS (typically 0.1-5.0),
        # we need to scale it. If it's already stress_level, pass through.
        eda_val = float(eda_raw)
        if info.get("context_eda_is_stress_level", True):
            stress = eda_val
        else:
            # Rough mapping: raw EDA µS → pseudo stress level
            # This is a placeholder; proper calibration would be needed
            stress = np.clip(eda_val * 20.0, 0, 100)

        # Compute rolling features using the normalized values
        roll = self._compute_rolling_features(current_time, hr_norm, stress)

        # Interaction / Non-linear features (identical to training pipeline)
        hr_pos = max(0.0, hr_norm)
        hr_x_eda = hr_norm * stress

        # Circadian Phase
        minutes_since_midnight = current_time.hour * 60 + current_time.minute
        hour_rad = (minutes_since_midnight / 1440) * 2 * np.pi
        hour_sin = np.sin(hour_rad)
        hour_cos = np.cos(hour_rad)

        # Build feature vector exactly matching training pipeline
        feature_dict = {
            'HR_norm': hr_norm,
            'EDA_norm': stress,
            'HR_pos': hr_pos,
            'HR_x_EDA': hr_x_eda,
            'HR_roll_30m': roll['hr_roll_30m'],
            'EDA_roll_60m': roll['eda_roll_60m'],
            'hour_sin': hour_sin,
            'hour_cos': hour_cos
        }

        X = np.array([[feature_dict[f] for f in self.features]])

        raw_m = float(self.model_pipeline.predict(X)[0])
        safe_m = np.clip(raw_m, 0.6, 2.5)  # Hard physiological limits

        if self.placebo_mode:
            return np.random.uniform(0.7, 1.8)

        return safe_m

    def _clip_patch_m(self, m_value: float) -> float:
        if not np.isfinite(m_value):
            return 1.0
        if self.patch_version == "spoorC_lut" and self.lut_policy == "safe_upward":
            lo = 1.0
            hi = 2.5
        elif self.patch_version == "spoorC_lut":
            lo = 0.6
            hi = 2.5
        else:
            lo = 0.6
            hi = 2.5
        return float(np.clip(m_value, lo, hi))

    def _stress_level_from_info(self, info: dict):
        eda_raw = info.get("context_eda")
        if eda_raw is None:
            return None
        eda_val = float(eda_raw)
        if info.get("context_eda_is_stress_level", True):
            return eda_val
        return float(np.clip(eda_val * 20.0, 0, 100))

    def _absolute_hr_from_info(self, info: dict):
        hr_raw = info.get("context_hr")
        if hr_raw is None:
            return None
        hr = float(hr_raw)
        if not np.isfinite(hr):
            return None
        if info.get("context_hr_is_relative", False):
            hr = hr + self.hr_rest
        return hr

    def _robust_lut_support(self):
        if self.lut_data is None:
            return None, None, None
        hr_grid = self.lut_data["hr_grid"]
        curve = self.lut_data["m_curve"]
        robust = self.lut_data["robust_mask"]
        robust_finite = robust & np.isfinite(curve)
        if not robust_finite.any():
            return None, None, None
        return hr_grid, curve, robust_finite

    def _predict_lut_m_t_for_hr(self, hr: float, info: dict) -> float:
        if self.lut_data is None:
            return 1.0

        if hr is None or not np.isfinite(hr):
            return 1.0

        hr_grid = self.lut_data["hr_grid"]
        curve = self.lut_data["m_curve"]
        robust = self.lut_data["robust_mask"]
        robust_finite = robust & np.isfinite(curve)

        if not robust_finite.any():
            return 1.0
        robust_hr = hr_grid[robust_finite]
        robust_curve = curve[robust_finite]

        if hr < robust_hr[0]:
            return 1.0
        if hr > robust_hr[-1]:
            m_t = float(np.nanmax(robust_curve))
        else:
            if hr < hr_grid[0] or hr > hr_grid[-1]:
                return 1.0

            idx = int(np.searchsorted(hr_grid, hr) - 1)
            idx = max(0, min(idx, len(hr_grid) - 2))
            if not robust[idx] or not robust[idx + 1]:
                return 1.0
            if not np.isfinite(curve[idx]) or not np.isfinite(curve[idx + 1]):
                return 1.0

            frac = (hr - hr_grid[idx]) / max(1e-9, hr_grid[idx + 1] - hr_grid[idx])
            m_t = float(curve[idx] * (1.0 - frac) + curve[idx + 1] * frac)

        stress_residual = info.get("context_stress_residual")
        if stress_residual is None:
            stress = self._stress_level_from_info(info)
            if (
                stress is not None
                and self.tier2_beta is not None
                and self.tier2_stress_hr_grid is not None
                and self.tier2_expected_stress is not None
            ):
                expected = np.interp(
                    hr,
                    self.tier2_stress_hr_grid,
                    self.tier2_expected_stress,
                    left=np.nan,
                    right=np.nan,
                )
                if np.isfinite(expected):
                    stress_residual = stress - expected

        if stress_residual is not None and self.tier2_beta is not None:
            m_t += float(self.tier2_beta) * float(stress_residual)

        return self._clip_patch_m(m_t)

    def _update_lut_hr_history(self, current_time: datetime, hr: float) -> float:
        timestamp = pd.Timestamp(current_time)
        self._lut_hr_history.append({"time": timestamp, "hr": float(hr)})
        cutoff = timestamp - pd.Timedelta(minutes=max(self.lut_history_min, 0.0))
        self._lut_hr_history = [x for x in self._lut_hr_history if x["time"] >= cutoff]

        if len(self._lut_hr_history) < 2:
            return 0.0
        oldest = self._lut_hr_history[0]
        elapsed_min = (timestamp - oldest["time"]).total_seconds() / 60.0
        if elapsed_min <= 0:
            return 0.0
        return float((hr - oldest["hr"]) / elapsed_min)

    def _predict_lut_m_t(self, info: dict, current_time: datetime) -> float:
        self._reset_lut_prediction_diagnostics()
        if self.lut_data is None:
            return 1.0

        hr = self._absolute_hr_from_info(info)
        if hr is None:
            return 1.0

        m_lut = self._predict_lut_m_t_for_hr(hr, info)
        m_projected = m_lut
        hr_projected = hr
        slope = 0.0
        active = False

        if str(self.lut_anticipation).lower() == "slope_projected":
            slope = self._update_lut_hr_history(current_time, hr)
            if slope > self.lut_slope_min_bpm_per_min and self.lut_lookahead_min > 0:
                _, curve, robust_finite = self._robust_lut_support()
                if curve is not None and robust_finite is not None:
                    robust_hr = self.lut_data["hr_grid"][robust_finite]
                    max_project_hr = float(robust_hr[-1])
                    hr_projected = min(hr + self.lut_lookahead_min * slope, max_project_hr)
                else:
                    hr_projected = hr + self.lut_lookahead_min * slope
                m_projected = self._predict_lut_m_t_for_hr(hr_projected, info)
                active = m_projected > m_lut

        m_final = max(m_lut, m_projected)
        self.last_m_t_lut = m_lut
        self.last_m_t_projected = m_projected
        self.last_m_t_final = m_final
        self.last_hr_slope = slope
        self.last_hr_projected = hr_projected
        self.last_lut_anticipation_active = active
        self.last_lut_anticipation_mode = self.lut_anticipation
        return m_final

    def policy(self, observation, reward, done, **info):
        # 1. Track controller time
        if self._now is None:
            self._now = datetime.utcnow()

        # 2. Predict m(t) from normalized context
        m_t = self._predict_m_t(info, self._now)

        # 3. Patch the OpenAPS profile ISF before determining basal/bolus
        # m_t = 2.0 (exercise) → ISF doubles → OpenAPS reduces insulin delivery
        original_profile = dict(self.profile)
        patched_isf = self.isf_baseline * m_t
        self.profile['sens'] = patched_isf
        guard_active = self._apply_exercise_guard(m_t)

        # 4. Run standard OpenAPS logic under patched perception
        action = super().policy(observation, reward, done, **info)

        # 5. Revert profile to baseline (prevent modifier stacking)
        self.profile = original_profile

        # 6. Store diagnostics on the controller for external logging.
        # Cannot attach to `action` because simglucose.controller.base.Action
        # is a namedtuple (immutable). Scripts should read these via
        # `controller.last_m_t` and `controller.last_patched_isf` after policy().
        self.last_m_t = m_t
        self.last_patched_isf = patched_isf
        self.last_patch_version = self.patch_version
        self.last_lut_policy = self.lut_policy
        self.last_exercise_guard_active = guard_active
        self.last_exercise_guard_mode = self.exercise_guard

        return action

    def _apply_exercise_guard(self, m_t: float) -> bool:
        mode = str(self.exercise_guard).lower()
        if mode in {"off", "none", ""}:
            return False
        if self.patch_version != "spoorC_lut" or m_t < self.exercise_guard_threshold:
            return False

        if mode in {"smb_basal_cap", "smb_off", "exercise_temp_target"}:
            self.profile["enableSMB_always"] = False
            self.profile["enableSMB_with_COB"] = False
            self.profile["enableSMB_after_carbs"] = False
            self.profile["enableSMB_high_bg"] = False

        if mode in {"smb_basal_cap", "basal_cap", "exercise_temp_target"}:
            current_basal = float(self.profile.get("current_basal", 0.0))
            max_safe_basal = float(self.profile.get("max_safe_basal", current_basal))
            self.profile["max_safe_basal"] = min(max_safe_basal, current_basal)

        return True
