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
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class BasalAdvice:
    rate_u_per_hr: float
    duration_min: int
    reason: Optional[str] = None


def determine_basal(
    glucose_mgdl: float,
    iob_u: float,
    activity_u_per_min: float,
    profile: Dict,
    min_delta: Optional[float] = None,
    short_avgdelta: Optional[float] = None,
    long_avgdelta: Optional[float] = None,
) -> BasalAdvice:
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

    # BGI (mg/dL per 5m)
    bgi = round(-activity_u_per_min * sens * 5.0, 2)

    # Deviation (30m projectie, conservatief):
    mdelta = min_delta if min_delta is not None else 0.0
    savg = short_avgdelta if short_avgdelta is not None else mdelta
    lavg = long_avgdelta if long_avgdelta is not None else savg
    min_avg = min(savg, lavg)
    deviation = round((30.0 / 5.0) * (mdelta - bgi))
    if deviation < 0:
        deviation = round((30.0 / 5.0) * (min_avg - bgi))
        if deviation < 0:
            deviation = round((30.0 / 5.0) * (lavg - bgi))

    # Naive eventual BG
    if iob_u > 0:
        naive_eventual = round(glucose_mgdl - (iob_u * sens))
    else:
        naive_eventual = round(glucose_mgdl - (iob_u * min(sens, profile.get("sens", sens))))
    eventual_bg = naive_eventual + deviation

    threshold = min_bg - 0.5 * (min_bg - 40.0)

    # expectedDelta volgens oref0: bgi + (target_delta / 24) over 2 uur
    target_bg = (min_bg + max_bg) / 2.0
    target_delta = target_bg - eventual_bg
    expected_delta = round(bgi + (target_delta / 24.0), 2)

    # Low suspend & predicted low
    if glucose_mgdl < threshold or eventual_bg < threshold:
        # Determine duration based on undershoot
        bg_undershoot = target_bg - max(eventual_bg, glucose_mgdl)
        worst_req_u = bg_undershoot / sens
        duration = int(round(60.0 * worst_req_u / max(current_basal, 1e-6)))
        duration = max(30, min(120, (duration // 30) * 30 if duration > 0 else 30))
        return BasalAdvice(rate_u_per_hr=0.0, duration_min=duration, reason="low-suspend")

    # If BG rises faster than expected, neutralize (conservative)
    if min_delta is not None and min_delta > expected_delta and min_delta > 0:
        return BasalAdvice(
            rate_u_per_hr=current_basal, duration_min=temp_duration, reason="rising>expected"
        )

    # Within range? neutral temp
    if min(eventual_bg, glucose_mgdl) < max_bg:
        return BasalAdvice(
            rate_u_per_hr=current_basal, duration_min=temp_duration, reason="in-range"
        )

    # High – calculate extra insulin requirement and temp rate
    insulin_req = max(0.0, (min(eventual_bg, glucose_mgdl) - target_bg) / sens)
    # cap aan max_iob
    if insulin_req > max_iob - iob_u:
        insulin_req = max(0.0, max_iob - iob_u)

    rate = current_basal + 2.0 * insulin_req
    rate = min(max_safe_basal, rate)

    return BasalAdvice(rate_u_per_hr=rate, duration_min=temp_duration, reason="high-temp")
