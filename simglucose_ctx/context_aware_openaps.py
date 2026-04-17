import pandas as pd
import numpy as np
import pickle
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
                 placebo_mode: bool = False, hr_rest: float = 65.0):
        super().__init__(deepcopy(profile))

        self.placebo_mode = placebo_mode
        self.model_pipeline = None
        self.features = None
        self.feature_semantics = None

        # Resting heart rate for normalizing absolute HR → HR_norm
        self.hr_rest = hr_rest

        # Buffer for computing continuous features (HR 30m, EDA 60m)
        self._history = []

        # Store initial Baseline ISF from profile so we can always revert
        self.isf_baseline = float(self.profile.get('sens', 50.0))

        # Load the patient's XGBoost predictor
        if model_path is not None:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.model_pipeline = data['pipeline']
                self.features = data['features']
                self.feature_semantics = data.get('feature_semantics', None)
                
                if self.feature_semantics:
                    print(f"  [ISF-Patch] Loaded model with semantics: "
                          f"HR={self.feature_semantics.get('HR_norm','?')}, "
                          f"EDA={self.feature_semantics.get('EDA_norm','?')}")

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

    def policy(self, observation, reward, done, **info):
        # 1. Track controller time
        if self._now is None:
            self._now = datetime.utcnow()

        # 2. Predict m(t) from normalized context
        m_t = self._predict_m_t(info, self._now)

        # 3. Patch the OpenAPS profile ISF before determining basal/bolus
        # m_t = 2.0 (exercise) → ISF doubles → OpenAPS reduces insulin delivery
        patched_isf = self.isf_baseline * m_t
        self.profile['sens'] = patched_isf

        # 4. Run standard OpenAPS logic under patched perception
        action = super().policy(observation, reward, done, **info)

        # 5. Revert profile to baseline (prevent modifier stacking)
        self.profile['sens'] = self.isf_baseline

        # 6. Store diagnostics on the controller for external logging.
        # Cannot attach to `action` because simglucose.controller.base.Action
        # is a namedtuple (immutable). Scripts should read these via
        # `controller.last_m_t` and `controller.last_patched_isf` after policy().
        self.last_m_t = m_t
        self.last_patched_isf = patched_isf

        return action
