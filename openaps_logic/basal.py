"""
OpenAPS oref0 determine_basal logic - Simplified Python port

This module implements core OpenAPS determine_basal logic in Python for
integration with SimGlucose. It is a SIMPLIFIED port for research purposes.

**Relation to OpenAPS oref0:**
- Based on: OpenAPS oref0 v0.7.0 (https://github.com/openaps/oref0)
- Original: JavaScript (oref0/lib/determine-basal/determine-basal.js)
- This port: Python (simplified for thesis scope)

**Validation status:**
- TODO: Validate against oref0 on standard test cases (see tests/test_openaps_vs_oref0.py)

**Scope limitations vs production oref0:**
- Included: Core temp basal logic (BGI, eventual_bg, safety limits)
- Included: IOB tracking with standard decay curves
- Excluded: Autosens (automatic sensitivity adjustment)
- Excluded: UAM (unannounced meal detection)
- Excluded: Dynamic ISF adjustment
- Excluded: Advanced meal absorption models

Author: Jip Voshol
License: MIT (consistent with OpenAPS oref0)
Reference: https://github.com/openaps/oref0
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


@dataclass
class BasalAdvice:
    rate_u_per_hr: float
    duration_min: int
    reason: Optional[str] = None
    # Diagnostics from prediction loop (populated when forecast is used)
    min_iob_pred_bg: Optional[float] = None
    min_iob_guard_bg: Optional[float] = None
    min_cob_pred_bg: Optional[float] = None
    min_cob_guard_bg: Optional[float] = None
    min_pred_bg: Optional[float] = None
    min_guard_bg: Optional[float] = None
    eventual_bg: Optional[float] = None
    naive_eventual_bg: Optional[float] = None
    deviation: Optional[float] = None


def determine_basal(
    glucose_mgdl: float,
    iob_u: float,
    activity_u_per_min: float,
    profile: Dict,
    min_delta: Optional[float] = None,
    short_avgdelta: Optional[float] = None,
    long_avgdelta: Optional[float] = None,
    iob_forecast_ticks: Optional[List[Any]] = None,
    cob_g: float = 0.0,
) -> BasalAdvice:
    """
    Compute basal recommendation. Mirrors oref0's determine-basal logic.

    When `iob_forecast_ticks` is provided (a list of IOBTotal snapshots over
    the next ~4 hours, one per 5-min tick), the function runs oref0's
    prediction loop: it builds IOBpredBG (and optionally COBpredBG) arrays
    tick-by-tick, then reduces these to minGuardBG (used for low-suspend)
    and minPredBG (used for high-temp insulin requirement). This replaces
    the old shortcut `naive_eventual = bg - iob*sens` as the decision driver
    and removes the structural over-suspension that formula causes.

    When `iob_forecast_ticks` is None, falls back to the pre-forecast logic
    (naive_eventual-based decisions) for backward compatibility.
    """
    current_basal = float(profile.get("current_basal", 1.0))
    sens = float(profile.get("sens", 50.0))
    min_bg = float(profile.get("min_bg", 90.0))
    max_bg = float(profile.get("max_bg", 120.0))
    max_iob = float(profile.get("max_iob", 3.0))
    temp_duration = int(profile.get("temp_basal_duration_min", 30))
    max_safe_basal = float(profile.get("max_safe_basal", current_basal * 3.0))

    # guards
    min_bg = max(60.0, min_bg)
    max_bg = max(min_bg + 5.0, max_bg)
    target_bg = (min_bg + max_bg) / 2.0

    # BGI (mg/dL per 5m) — oref0 equivalent: round(-activity * sens * 5, 2)
    bgi = round(-activity_u_per_min * sens * 5.0, 2)

    # Deviation (30-min projection using glucose delta vs predicted BGI)
    mdelta = min_delta if min_delta is not None else 0.0
    savg = short_avgdelta if short_avgdelta is not None else mdelta
    lavg = long_avgdelta if long_avgdelta is not None else savg
    min_avg = min(savg, lavg)
    deviation = round((30.0 / 5.0) * (mdelta - bgi))
    if deviation < 0:
        deviation = round((30.0 / 5.0) * (min_avg - bgi))
        if deviation < 0:
            deviation = round((30.0 / 5.0) * (lavg - bgi))

    # Naive eventual BG (oref0 signature — still computed for reference)
    if iob_u > 0:
        naive_eventual = round(glucose_mgdl - (iob_u * sens))
    else:
        naive_eventual = round(glucose_mgdl - (iob_u * min(sens, profile.get("sens", sens))))
    eventual_bg = naive_eventual + deviation

    threshold = min_bg - 0.5 * (min_bg - 40.0)

    # Expected delta for "rising>expected" check
    target_delta = target_bg - eventual_bg
    expected_delta = round(bgi + (target_delta / 24.0), 2)

    # ================================================================
    # oref0-style prediction loop (if forecast provided)
    # ================================================================
    if iob_forecast_ticks is not None and len(iob_forecast_ticks) > 0:
        # Per-tick 5-min carb impact (oref0 uses `ci = deviation / 2`,
        # but because our deviation is already over 30-min window we divide
        # by 6 to get per-5-min impact.)
        ci = deviation / 6.0 if deviation > 0 else 0.0

        # Carb impact duration (cid) in 5-min ticks: roughly 3h of absorption,
        # longer if cob is large. Simplified from oref0.
        cid_ticks = 36  # 3h absorption default
        if cob_g and cob_g > 0:
            # Assume ~12g/h absorption → cid = cob/12 hours → × 12 to get ticks
            cid_ticks = int(max(12, min(60, (cob_g / 12.0) * 12)))

        # Remaining carb impact peak & time for the late-absorption bump
        remaining_ca_time = 3.0  # hours
        remaining_ci_peak = 0.0  # simplified: no extra late-absorption bump
        # (oref0 computes this from carbimpactratio and remainingCarbs;
        # leaving at 0 under-estimates COB effect slightly but is conservative)

        predict_cob = cob_g is not None and cob_g > 0

        iob_pred_bgs: List[float] = [glucose_mgdl]
        cob_pred_bgs: List[float] = [glucose_mgdl]

        # Minimum tracking
        min_iob_pred_bg = glucose_mgdl
        min_cob_pred_bg = glucose_mgdl
        min_iob_guard_bg = glucose_mgdl
        min_cob_guard_bg = glucose_mgdl

        # Insulin peak delay: oref0 only starts tracking "min main pred" after
        # insulin peak effect has occurred (~90 min). Before that, any prediction
        # dip is expected-decay behavior, not a true low.
        insulin_peak_5m = 18  # 90 / 5

        for i, tick in enumerate(iob_forecast_ticks):
            pred_bgi = round(-tick.activity * sens * 5.0, 2)

            # predDev: deviation decays linearly to 0 over first 60 min (12 ticks)
            pred_dev = ci * max(0.0, 1.0 - (i + 1) / 12.0)

            prev_iob = iob_pred_bgs[-1]
            iob_pred_bg = prev_iob + pred_bgi + pred_dev
            iob_pred_bgs.append(iob_pred_bg)

            if predict_cob:
                # predCI: active carb impact decays linearly to 0 over cid_ticks*2
                pred_ci = max(0.0, ci * (1.0 - (i + 1) / max(cid_ticks * 2.0, 1.0)))
                # remainingCI: triangular bump (peak at half absorption time)
                intervals = min(i + 1, int(remaining_ca_time * 12) - (i + 1))
                remaining_ci = max(
                    0.0,
                    intervals / (remaining_ca_time / 2.0 * 12.0) * remaining_ci_peak,
                )
                prev_cob = cob_pred_bgs[-1]
                cob_pred_bg = prev_cob + pred_bgi + min(0.0, pred_dev) + pred_ci + remaining_ci
                cob_pred_bgs.append(cob_pred_bg)

            # Main min (with 90-min insulin-peak delay)
            if i + 1 > insulin_peak_5m:
                if iob_pred_bg < min_iob_pred_bg:
                    min_iob_pred_bg = iob_pred_bg
                if predict_cob and cob_pred_bg < min_cob_pred_bg:
                    min_cob_pred_bg = cob_pred_bg

            # Guard BG (no delay) — used for hypo safety
            if iob_pred_bg < min_iob_guard_bg:
                min_iob_guard_bg = iob_pred_bg
            if predict_cob and cob_pred_bg < min_cob_guard_bg:
                min_cob_guard_bg = cob_pred_bg

        min_iob_pred_bg = round(min_iob_pred_bg)
        min_iob_guard_bg = round(min_iob_guard_bg)
        if predict_cob:
            min_cob_pred_bg = round(min_cob_pred_bg)
            min_cob_guard_bg = round(min_cob_guard_bg)

        # Select aggregate minima
        if predict_cob:
            # Blend: oref0 uses fractionCarbsLeft, we use simple mean since we
            # don't track fractionCarbsLeft here
            min_pred_bg = max(min_iob_pred_bg, min_cob_pred_bg)
            min_guard_bg = min_cob_guard_bg  # COB guard is more conservative
        else:
            min_pred_bg = min_iob_pred_bg
            min_guard_bg = min_iob_guard_bg

        # ============================================================
        # Decisions based on minGuardBG / minPredBG (oref0 style)
        # ============================================================
        diag = dict(
            min_iob_pred_bg=float(min_iob_pred_bg),
            min_iob_guard_bg=float(min_iob_guard_bg),
            min_cob_pred_bg=float(min_cob_pred_bg) if predict_cob else None,
            min_cob_guard_bg=float(min_cob_guard_bg) if predict_cob else None,
            min_pred_bg=float(min_pred_bg),
            min_guard_bg=float(min_guard_bg),
            eventual_bg=float(eventual_bg),
            naive_eventual_bg=float(naive_eventual),
            deviation=float(deviation),
        )

        # Low-glucose suspend (hypo prevention) — uses minGuardBG, not eventual_bg.
        # This is the core fix: with forecast, min_guard_bg reflects the actual
        # minimum of the projected glucose curve given IOB decay, which is much
        # higher than `bg - iob*sens` that assumed all IOB acts instantly.
        if glucose_mgdl < threshold or min_guard_bg < threshold:
            bg_undershoot = target_bg - max(min_guard_bg, glucose_mgdl)
            worst_req_u = bg_undershoot / sens
            duration = int(round(60.0 * worst_req_u / max(current_basal, 1e-6)))
            duration = max(30, min(120, (duration // 30) * 30 if duration > 0 else 30))
            return BasalAdvice(
                rate_u_per_hr=0.0, duration_min=duration,
                reason=f"low-suspend (minGuardBG={min_guard_bg})", **diag,
            )

        # Rising faster than expected → hold at scheduled basal (conservative)
        if min_delta is not None and min_delta > expected_delta and min_delta > 0:
            return BasalAdvice(
                rate_u_per_hr=current_basal, duration_min=temp_duration,
                reason="rising>expected", **diag,
            )

        # Eventual BG below min_bg → low temp (not suspend, but reduce basal)
        if eventual_bg < min_bg:
            # Rate based on undershoot; oref0 uses insulinReq = 2 * min(0, (eventualBG - target)/sens)
            insulin_req_low = 2.0 * min(0.0, (eventual_bg - target_bg) / sens)
            rate = max(0.0, current_basal + 2.0 * insulin_req_low)
            return BasalAdvice(
                rate_u_per_hr=rate, duration_min=temp_duration,
                reason=f"low-temp (eventual={eventual_bg})", **diag,
            )

        # In range (and min_pred_bg in range too) → neutral
        if min_pred_bg < max_bg and min(eventual_bg, glucose_mgdl) < max_bg:
            return BasalAdvice(
                rate_u_per_hr=current_basal, duration_min=temp_duration,
                reason="in-range", **diag,
            )

        # High temp — oref0 uses min(minPredBG, eventualBG) to decide insulin_req
        insulin_req = max(0.0, (min(min_pred_bg, eventual_bg) - target_bg) / sens)
        if insulin_req > max_iob - iob_u:
            insulin_req = max(0.0, max_iob - iob_u)
        rate = current_basal + 2.0 * insulin_req
        rate = min(max_safe_basal, rate)
        return BasalAdvice(
            rate_u_per_hr=rate, duration_min=temp_duration,
            reason=f"high-temp (minPredBG={min_pred_bg})", **diag,
        )

    # ================================================================
    # Fallback: legacy pre-forecast logic (backward compatibility)
    # ================================================================
    diag_fallback = dict(
        eventual_bg=float(eventual_bg),
        naive_eventual_bg=float(naive_eventual),
        deviation=float(deviation),
    )
    if glucose_mgdl < threshold or eventual_bg < threshold:
        bg_undershoot = target_bg - max(eventual_bg, glucose_mgdl)
        worst_req_u = bg_undershoot / sens
        duration = int(round(60.0 * worst_req_u / max(current_basal, 1e-6)))
        duration = max(30, min(120, (duration // 30) * 30 if duration > 0 else 30))
        return BasalAdvice(
            rate_u_per_hr=0.0, duration_min=duration,
            reason="low-suspend (legacy naive)", **diag_fallback,
        )

    if min_delta is not None and min_delta > expected_delta and min_delta > 0:
        return BasalAdvice(
            rate_u_per_hr=current_basal, duration_min=temp_duration,
            reason="rising>expected", **diag_fallback,
        )

    if min(eventual_bg, glucose_mgdl) < max_bg:
        return BasalAdvice(
            rate_u_per_hr=current_basal, duration_min=temp_duration,
            reason="in-range", **diag_fallback,
        )

    insulin_req = max(0.0, (min(eventual_bg, glucose_mgdl) - target_bg) / sens)
    if insulin_req > max_iob - iob_u:
        insulin_req = max(0.0, max_iob - iob_u)
    rate = current_basal + 2.0 * insulin_req
    rate = min(max_safe_basal, rate)
    return BasalAdvice(
        rate_u_per_hr=rate, duration_min=temp_duration,
        reason="high-temp (legacy)", **diag_fallback,
    )
