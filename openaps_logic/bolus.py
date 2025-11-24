"""
determine_bolus (SMB) – simplified oref1 port.

Implements core ideas:
- threshold / target / naive_eventualBG
- insulinReq ~ (eventualBG - target)/ISF
- SMB caps: maxSMBBasalMinutes, SMBInterval, bolus_increment

Note: this is a first, conservative subset without UAM/COB-predBG.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class BolusAdvice:
    units: float
    reason: Optional[str] = None


def _round_to_increment(value: float, increment: float) -> float:
    if increment <= 0:
        return max(0.0, value)
    steps = int((value + 1e-9) / increment)
    return max(0.0, steps * increment)


def determine_bolus(
    glucose_mgdl: float,
    iob_u: float,
    cob_g: float,
    profile: Dict,
    minutes_since_last_bolus: Optional[float] = None,
    min_delta: Optional[float] = None,
) -> BolusAdvice:
    # Profile values (defaults if not specified)
    min_bg = float(profile.get("min_bg", 90))
    max_bg = float(profile.get("max_bg", 120))
    target_bg = (min_bg + max_bg) / 2.0
    sens = float(profile.get("sens", 50.0))  # mg/dL per U
    current_basal = float(profile.get("current_basal", 1.0))  # U/hr
    max_iob = float(profile.get("max_iob", 3.0))
    bolus_increment = float(profile.get("bolus_increment", 0.1))
    maxSMBBasalMinutes = float(profile.get("maxSMBBasalMinutes", 30.0))
    SMBInterval = float(profile.get("SMBInterval", 3.0))  # min
    high_bg_target = float(profile.get("enableSMB_high_bg_target", 160.0))
    enableSMB_always = bool(profile.get("enableSMB_always", False))
    enableSMB_with_COB = bool(profile.get("enableSMB_with_COB", True))
    enableSMB_after_carbs = bool(profile.get("enableSMB_after_carbs", False))
    enableSMB_with_temptarget = bool(profile.get("enableSMB_with_temptarget", False))
    enableSMB_high_bg = bool(profile.get("enableSMB_high_bg", True))
    temptargetSet = bool(profile.get("temptargetSet", False))
    maxDelta_bg_threshold = float(profile.get("maxDelta_bg_threshold", 0.2))

    # Guard rails
    min_bg = max(60.0, min_bg)
    max_bg = max(min_bg + 5.0, max_bg)
    target_bg = (min_bg + max_bg) / 2.0

    # Threshold definition (oref0: min_bg - 0.5*(min_bg-40))
    threshold = min_bg - 0.5 * (min_bg - 40.0)

    # Naive eventual BG (without CI or UAM corrections)
    naive_eventualBG = glucose_mgdl - (iob_u * sens)

    # Safety: no SMB below threshold
    if glucose_mgdl < threshold:
        return BolusAdvice(0.0, reason=f"BG {glucose_mgdl:.0f}<thr {threshold:.0f}")

    # If IOB above max_iob: no extra bolus
    if iob_u > max_iob:
        return BolusAdvice(0.0, reason=f"IOB {iob_u:.2f}>max_iob {max_iob:.2f}")

    # Only SMB if interval has elapsed
    if minutes_since_last_bolus is not None and minutes_since_last_bolus < SMBInterval:
        return BolusAdvice(0.0, reason=f"wait {SMBInterval-minutes_since_last_bolus:.0f}m")

    # Disable SMB for sudden spikes
    if min_delta is not None and glucose_mgdl > 0:
        if min_delta > maxDelta_bg_threshold * glucose_mgdl:
            return BolusAdvice(0.0, reason="spike->noSMB")

    # Enablement rules (oref1 subset)
    smb_enabled = False
    if enableSMB_always:
        smb_enabled = True
    if enableSMB_with_COB and cob_g and cob_g > 0:
        smb_enabled = True
    if enableSMB_after_carbs and cob_g and cob_g > 0:
        smb_enabled = True
    if enableSMB_with_temptarget and temptargetSet and target_bg < 100:
        smb_enabled = True
    if enableSMB_high_bg and glucose_mgdl >= high_bg_target:
        smb_enabled = True

    # If projected or actual above target → SMB if enabled
    if smb_enabled and (naive_eventualBG > max_bg or glucose_mgdl > max_bg):
        # Use the higher of actual and projected BG vs. target
        proj = max(glucose_mgdl, naive_eventualBG)
        insulin_req = (proj - target_bg) / sens
        insulin_req = max(0.0, insulin_req)
        # COB can justify higher SMB within caps
        max_bolus = current_basal * (maxSMBBasalMinutes / 60.0)
        # COB factor (max 1.5×)
        cob_factor = 1.0 + min(0.5, cob_g / 60.0)
        micro_bolus = min(insulin_req * 0.5 * cob_factor, max_bolus)
        micro_bolus = _round_to_increment(micro_bolus, bolus_increment)
        return BolusAdvice(micro_bolus, reason=f"SMB insulinReq {insulin_req:.2f}")

    # Within range: no SMB
    return BolusAdvice(0.0, reason="in range")
