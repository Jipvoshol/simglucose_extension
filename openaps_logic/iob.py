"""
IOB based on oref0/oref1.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import exp
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


@dataclass
class Treatment:
    date_ms: int  # epoch millis
    insulin: float  # units


@dataclass
class IOBTotal:
    iob: float
    activity: float
    basaliob: float
    bolusiob: float
    netbasalinsulin: float
    bolusinsulin: float
    time: datetime


def _round(value: float, digits: int) -> float:
    scale = 10 ** digits
    return round(value * scale) / scale


def _as_epoch_ms(dt: Union[int, float, datetime]) -> int:
    if isinstance(dt, (int, float)):
        return int(dt)
    return int(dt.timestamp() * 1000)


def _now_ms(now: Optional[datetime]) -> int:
    return _as_epoch_ms(now or datetime.utcnow())


def _ensure_treatments(objs: Iterable[Dict[str, Any]]) -> List[Treatment]:
    treatments: List[Treatment] = []
    for o in objs:
        if o is None:
            continue
        date_val = o.get("date") or o.get("date_ms") or o.get("timestamp_ms")
        insulin = o.get("insulin")
        if date_val is None or insulin is None:
            continue
        treatments.append(Treatment(date_ms=int(date_val), insulin=float(insulin)))
    return treatments


def _iob_calc_bilinear(insulin_u: float, mins_ago: float, dia_h: float) -> Tuple[float, float]:
    default_dia_h = 3.0
    peak_min = 75.0
    end_min = 180.0

    time_scalar = default_dia_h / max(dia_h, 1e-6)
    scaled = time_scalar * mins_ago

    activity_contrib = 0.0
    iob_contrib = 0.0

    activity_peak = 2.0 / (dia_h * 60.0)
    slope_up = activity_peak / peak_min
    slope_down = -activity_peak / (end_min - peak_min)

    if scaled < peak_min:
        activity_contrib = insulin_u * (slope_up * scaled)
        x1 = (scaled / 5.0) + 1.0
        iob_contrib = insulin_u * ((-0.001852 * x1 * x1) + (0.001852 * x1) + 1.0)
    elif scaled < end_min:
        mins_past_peak = scaled - peak_min
        activity_contrib = insulin_u * (activity_peak + (slope_down * mins_past_peak))
        x2 = ((scaled - peak_min) / 5.0)
        iob_contrib = insulin_u * ((0.001323 * x2 * x2) + (-0.054233 * x2) + 0.555560)

    return activity_contrib, iob_contrib


def _iob_calc_exponential(
    insulin_u: float,
    mins_ago: float,
    dia_h: float,
    peak_min: float,
    profile: Dict[str, Any],
) -> Tuple[float, float]:
    curve = str(profile.get("curve", "rapid-acting")).lower()
    if curve == "rapid-acting":
        if profile.get("useCustomPeakTime") and (profile.get("insulinPeakTime") is not None):
            ipt = int(profile["insulinPeakTime"])
            if ipt > 120:
                peak_min = 120.0
            elif ipt < 50:
                peak_min = 50.0
            else:
                peak_min = float(ipt)
        else:
            peak_min = 75.0
    elif curve == "ultra-rapid":
        if profile.get("useCustomPeakTime") and (profile.get("insulinPeakTime") is not None):
            ipt = int(profile["insulinPeakTime"])
            if ipt > 100:
                peak_min = 100.0
            elif ipt < 35:
                peak_min = 35.0
            else:
                peak_min = float(ipt)
        else:
            peak_min = 55.0

    end_min = dia_h * 60.0

    activity_contrib = 0.0
    iob_contrib = 0.0

    if mins_ago < end_min and end_min > 0:
        tau = peak_min * (1 - peak_min / end_min) / (1 - 2 * peak_min / end_min)
        a = 2 * tau / end_min
        S = 1.0 / (1 - a + (1 + a) * exp(-end_min / tau))

        activity_contrib = insulin_u * (S / (tau * tau)) * mins_ago * (1 - mins_ago / end_min) * exp(-mins_ago / tau)
        iob_contrib = insulin_u * (
            1
            - S
            * (1 - a)
            * (
                ((mins_ago * mins_ago) / (tau * end_min * (1 - a)) - mins_ago / tau - 1)
                * exp(-mins_ago / tau)
                + 1
            )
        )

    return activity_contrib, iob_contrib


def _iob_calc_for_treatment(
    treatment: Treatment, now_ms: int, curve: str, dia_h: float, peak_min: float, profile: Dict[str, Any]
) -> Tuple[float, float]:
    if treatment.insulin is None:
        return 0.0, 0.0
    mins_ago = max(0.0, (now_ms - treatment.date_ms) / 1000.0 / 60.0)
    if curve == "bilinear":
        return _iob_calc_bilinear(treatment.insulin, mins_ago, dia_h)
    else:
        return _iob_calc_exponential(treatment.insulin, mins_ago, dia_h, peak_min, profile)


def iob_total(
    treatments_input: Iterable[Dict[str, Any]],
    profile: Dict[str, Any],
    now: Optional[datetime] = None,
) -> IOBTotal:
    now_ms = _now_ms(now)
    treatments = _ensure_treatments(treatments_input)
    if not treatments:
        return IOBTotal(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, datetime.utcfromtimestamp(now_ms / 1000))

    dia = float(profile.get("dia", 6.0))
    if dia < 3.0:
        dia = 3.0

    curve_defaults = {
        "bilinear": {"requireLongDia": False, "peak": 75.0},
        "rapid-acting": {"requireLongDia": True, "peak": 75.0, "tdMin": 300},
        "ultra-rapid": {"requireLongDia": True, "peak": 55.0, "tdMin": 300},
    }
    curve = str(profile.get("curve", "bilinear")).lower()
    if curve not in curve_defaults:
        curve = "rapid-acting"
    defaults = curve_defaults[curve]
    if defaults.get("requireLongDia") and dia < 5.0:
        dia = 5.0
    peak_min = float(defaults.get("peak", 75.0))

    iob_sum = 0.0
    activity_sum = 0.0
    basaliob = 0.0
    bolusiob = 0.0
    netbasalinsulin = 0.0
    bolusinsulin = 0.0

    dia_ago_ms = now_ms - int(dia * 60 * 60 * 1000)

    for tr in treatments:
        if tr.date_ms > now_ms or tr.date_ms <= dia_ago_ms:
            continue
        a_contrib, iob_contrib = _iob_calc_for_treatment(tr, now_ms, curve, dia, peak_min, profile)
        if iob_contrib:
            iob_sum += iob_contrib
            activity_sum += a_contrib or 0.0
            if tr.insulin < 0.1:
                basaliob += iob_contrib
                netbasalinsulin += tr.insulin
            else:
                bolusiob += iob_contrib
                bolusinsulin += tr.insulin

    return IOBTotal(
        iob=_round(iob_sum, 3),
        activity=_round(activity_sum, 4),
        basaliob=_round(basaliob, 3),
        bolusiob=_round(bolusiob, 3),
        netbasalinsulin=_round(netbasalinsulin, 3),
        bolusinsulin=_round(bolusinsulin, 3),
        time=datetime.utcfromtimestamp(now_ms / 1000.0),
    )


