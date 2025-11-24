"""Context-aware ISF multiplier utilities for SimGlucose.

Drop-in helpers for computing a time-varying insulin sensitivity factor (ISF)
multiplier m(t) based on heart rate (HR) and electrodermal activity (EDA).

Author: Jip Voshol
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ContextConfig:
    dt_minutes: int = 5
    hr_rest: float = 60.0
    hr_max: float = 180.0
    alpha: float = 1.25  # exercise gain
    beta: float = 0.3  # stress gain
    mmin: float = 0.6
    mmax: float = 2.5
    max_delta_per_step: float = 0.1
    ema_half_life_min: float = 15.0
    hysteresis_steps: int = 2
    night_cap: Optional[float] = None  # e.g., 0.2 for ±20% at night
    night_hours: tuple = (23, 6)  # (start, end) local time range
    # Suppress stress effect when HR is high (exercise-like)
    # effective_beta = beta * (1 - I)^stress_hr_supp_pow, where I in [0,1]
    stress_hr_supp_pow: float = 2.0
    # Optional hard gating: if HR intensity >= threshold, ignore stress
    stress_hr_off_threshold: Optional[float] = 0.65

    # Vm0/Vmx scaling exponents (hybrid strategy)
    vm0_exponent_exercise: float = 3.5  # Exponential boost for AMPK (10-20x literature)
    vmx_exponent_exercise: float = 0.2  # Minimal scaling to avoid math artifacts

    # Advanced features (sigmoid-based + asymmetric kinetics)
    use_sigmoid: bool = False  # Use sigmoid instead of linear for HRR
    sigmoid_threshold: float = 0.5  # θ in sigmoid (HRR center point)
    sigmoid_width: float = 0.15  # b in sigmoid (transition sharpness)
    use_asymmetric_kinetics: bool = False  # Different rise/fall times
    tau_onset_min: float = 10.0  # Rise time (exercise onset)
    tau_offset_min: float = 30.0  # Fall time (post-exercise)
    use_log_additive: bool = False  # Log-space combination (more stable)

    # Validation (v1.0 feature)
    strict_validation: bool = False  # Enable strict errors (will be default in v2.0)

    def __post_init__(self):
        """Validate configuration parameters.

        In v1.0, validation warnings are logged but don't raise errors by default.
        Set strict_validation=True to enable errors (will be default in v2.0).
        """
        errors = []

        # Critical validations
        if self.hr_max <= self.hr_rest:
            errors.append(f"hr_max ({self.hr_max}) must be > hr_rest ({self.hr_rest})")

        if not (0 < self.mmin < self.mmax):
            errors.append(f"Invalid m bounds: 0 < mmin ({self.mmin}) < mmax ({self.mmax}) required")

        if self.alpha <= 0:
            errors.append(f"alpha ({self.alpha}) must be > 0 (exercise gain)")

        if self.beta <= 0:
            errors.append(f"beta ({self.beta}) must be > 0 (stress gain)")

        if self.vm0_exponent_exercise <= 0:
            errors.append(f"vm0_exponent_exercise ({self.vm0_exponent_exercise}) must be > 0")

        if self.vmx_exponent_exercise < 0:
            errors.append(f"vmx_exponent_exercise ({self.vmx_exponent_exercise}) must be >= 0")

        if self.night_cap is not None and not (0 < self.night_cap < 1):
            errors.append(f"night_cap ({self.night_cap}) must be in range (0, 1)")

        if self.max_delta_per_step <= 0:
            errors.append(f"max_delta_per_step ({self.max_delta_per_step}) must be > 0")

        if self.ema_half_life_min <= 0:
            errors.append(f"ema_half_life_min ({self.ema_half_life_min}) must be > 0")

        if self.dt_minutes <= 0:
            errors.append(f"dt_minutes ({self.dt_minutes}) must be > 0")

        # Report or raise
        if errors:
            msg = "Invalid ContextConfig parameters:\n  - " + "\n  - ".join(errors)
            if self.strict_validation:
                raise ValueError(msg)
            else:
                logger.warning(
                    msg + "\n(Set strict_validation=True to raise errors. "
                    "This will be the default in v2.0)"
                )


class ContextStream:
    """Serve m(t) for a given timestamp.

    Parameters
    ----------
    hr : pd.Series
        Heart rate, indexed by tz-aware or naive timestamps.
    eda : pd.Series
        Stress proxy (EDA), same index domain as hr.
    cfg : ContextConfig
        Configuration hyperparameters.
    preprocess : bool
        If True, resample to cfg.dt_minutes and apply smoothing.
    eda_min_max : Optional[tuple]
        Optional (min,max) for EDA scaling; if None, computed from training window.
    """

    def __init__(
        self,
        hr: pd.Series,
        eda: pd.Series,
        cfg: ContextConfig = ContextConfig(),
        preprocess: bool = True,
        eda_min_max: Optional[tuple] = None,
    ):
        self.cfg = cfg
        self.prev_m = 1.0

        if preprocess:
            rule = f"{cfg.dt_minutes}min"
            hr = hr.sort_index().astype(float).resample(rule).mean().ffill()
            eda = eda.sort_index().astype(float).resample(rule).mean().ffill()

            # Robust clipping to 5th-95th percentile to reduce spikes/outliers
            def _clip(s: pd.Series) -> pd.Series:
                lo, hi = s.quantile(0.05), s.quantile(0.95)
                return s.clip(lower=lo, upper=hi)

            hr = _clip(hr)
            eda = _clip(eda)

            # Exponential moving average smoothing
            alpha = 1 - np.exp(-np.log(2) / (cfg.ema_half_life_min / cfg.dt_minutes))
            hr = hr.ewm(alpha=alpha, adjust=False).mean()
            eda = eda.ewm(alpha=alpha, adjust=False).mean()

            # Data quality validation (v1.0 feature)
            hr_nan_pct = hr.isna().mean()
            eda_nan_pct = eda.isna().mean()

            if hr_nan_pct > 0.1:
                logger.warning(
                    f"HR data has {hr_nan_pct:.1%} missing values after resampling. "
                    f"Consider using a longer measurement window or interpolating gaps."
                )

            if eda_nan_pct > 0.1:
                logger.warning(
                    f"EDA data has {eda_nan_pct:.1%} missing values after resampling. "
                    f"Consider using a longer measurement window or interpolating gaps."
                )

            # Fill remaining NaNs with sensible defaults
            # Forward fill → backward fill → default value
            hr = hr.ffill().bfill().fillna(cfg.hr_rest)
            eda = eda.ffill().bfill().fillna(0.0)

        self.hr = hr
        self.eda = eda

        # EDA normalization to [0,1]
        if eda_min_max is None:
            eda_min, eda_max = float(self.eda.min()), float(self.eda.max())
            if eda_max <= eda_min:
                eda_max = eda_min + 1.0
        else:
            eda_min, eda_max = eda_min_max
        self._eda_min = eda_min
        self._eda_max = eda_max

    def _intensity_hr(self, hr: float) -> float:
        cfg = self.cfg
        if cfg.use_sigmoid:
            x = (hr - cfg.hr_rest) / max(1e-6, (cfg.hr_max - cfg.hr_rest))
            return float(1 / (1 + np.exp(-(x - cfg.sigmoid_threshold) / cfg.sigmoid_width)))
        else:
            denom = max(1e-6, (cfg.hr_max - cfg.hr_rest))
            x = (hr - cfg.hr_rest) / denom
            return float(np.clip(x, 0.0, 1.0))

    def _stress_idx(self, eda: float) -> float:
        # Min-max normalized to [0, 1]
        x = (eda - self._eda_min) / max(1e-6, (self._eda_max - self._eda_min))
        return float(np.clip(x, 0.0, 1.0))

    def _rate_limit(self, m_raw: float) -> float:
        cfg = self.cfg
        delta = np.clip(m_raw - self.prev_m, -cfg.max_delta_per_step, cfg.max_delta_per_step)
        return float(self.prev_m + delta)

    def m(self, now_ts: pd.Timestamp, steps_above_thresh: Optional[dict] = None) -> float:
        """Return m(t) for timestamp now_ts.

        Optionally pass a dict to keep track of hysteresis counters across calls:
        steps_above_thresh = {"exercise": int, "stress": int}
        """
        cfg = self.cfg
        # Align to previous available grid index if exact timestamp not present
        idx_index = self.hr.index
        ts = pd.Timestamp(now_ts)

        # Check if timestamp is significantly outside data range (30min grace period)
        GRACE_PERIOD = pd.Timedelta(minutes=30)

        if ts < idx_index[0] - GRACE_PERIOD:
            logger.warning(
                f"Timestamp {ts} is >30min before first HR data point ({idx_index[0]}). "
                f"Using first available data point."
            )
        elif ts > idx_index[-1] + GRACE_PERIOD:
            logger.warning(
                f"Timestamp {ts} is >30min after last HR data point ({idx_index[-1]}). "
                f"Extrapolating from last available data point."
            )

        # Handle out-of-bounds (clamp to valid range)
        if ts < idx_index[0]:
            idx = idx_index[0]
        elif ts > idx_index[-1]:
            idx = idx_index[-1]
        else:
            # Original forward-fill logic (correct!)
            # searchsorted with side='right' finds first index > ts
            # pos - 1 gives last timestamp <= ts
            pos = idx_index.searchsorted(ts, side="right")
            if pos == 0:
                # Safety: prevent negative index (though ts < idx_index[0] caught above)
                idx = idx_index[0]
            else:
                idx = idx_index[pos - 1]

        hr = float(self.hr.loc[idx])
        eda = float(self.eda.loc[idx])

        I = self._intensity_hr(hr)
        S = self._stress_idx(eda)

        # Hysteresis bookkeeping
        if steps_above_thresh is not None:
            if I >= 0.4:
                steps_above_thresh["exercise"] = steps_above_thresh.get("exercise", 0) + 1
            else:
                steps_above_thresh["exercise"] = 0
            if S >= 0.5:
                steps_above_thresh["stress"] = steps_above_thresh.get("stress", 0) + 1
            else:
                steps_above_thresh["stress"] = 0
            if steps_above_thresh["exercise"] < cfg.hysteresis_steps:
                I = 0.0
            if steps_above_thresh["stress"] < cfg.hysteresis_steps:
                S = 0.0

        m_ex = 1.0 + cfg.alpha * I

        # Stress effect disambiguation:
        # Only suppress stress when HR is clearly in exercise territory (I >= 0.5).
        # This prevents false suppression during genuine stress with mild HR elevation
        # (e.g., HR 110 during exam stress should still show full stress effect).
        if I >= 0.5:
            # Clear exercise zone (HR > ~110-120): Apply gradual suppression
            beta_eff = cfg.beta * ((1.0 - I) ** cfg.stress_hr_supp_pow)
        else:
            # Ambiguous or low zone (HR < 110-120): Full stress effect
            beta_eff = cfg.beta

        if cfg.stress_hr_off_threshold is not None and I >= cfg.stress_hr_off_threshold:
            # When HR is clearly in high exercise territory, ignore stress entirely
            beta_eff = 0.0

        m_st = 1.0 - beta_eff * S

        # Combination strategy: Use dominance logic instead of multiplication
        # At low HR (I < 0.5): Stress dominates → use min(m_ex, m_st)
        # At high HR (I >= 0.5): Exercise dominates → use m_ex (stress already suppressed)
        if I < 0.5 and m_st < 1.0:
            # Low HR + stress present: Stress effect dominates
            # Example: HR 110 + high stress → m_st = 0.7, m_ex = 1.4 → use 0.7
            m_raw = float(np.clip(min(m_ex, m_st), cfg.mmin, cfg.mmax))
        else:
            # High HR or no stress: Multiplicative (original behavior)
            m_raw = float(np.clip(m_ex * m_st, cfg.mmin, cfg.mmax))

        # Optional night cap
        if cfg.night_cap is not None:
            hour = pd.Timestamp(idx).hour
            s, e = cfg.night_hours
            in_night = (hour >= s) or (hour < e) if s > e else (s <= hour < e)
            if in_night:
                # bring m_raw closer to 1.0 by cfg.night_cap
                cap = cfg.night_cap
                m_raw = 1.0 + np.clip(m_raw - 1.0, -cap, cap)

        m_smooth = self._rate_limit(m_raw)
        self.prev_m = m_smooth
        return m_smooth
