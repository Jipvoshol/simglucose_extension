"""
OpenAPSController: Simplified controller based on OpenAPS oref0 principles.

Implements core OpenAPS logic (determine_basal, IOB tracking, safety limits)
but excludes advanced features (autosens, UAM, complex absorption models).

Features:
- Temp basal adjustments based on BGI/deviation predictions
- IOB tracking with bilinear/rapid/ultra-rapid decay curves
- COB tracking with linear 3-hour absorption
- Safety: low-suspend, max IOB, max safe basal

Simplifications vs most advanced OpenAPS:
- No autosens (automatic sensitivity adjustment)
- No UAM (unannounced meal detection)
- Linear COB absorption (vs adaptive curves)
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
from datetime import datetime, timedelta, timezone

from simglucose.controller.base import Controller, Action

from openaps_logic.basal import determine_basal
from openaps_logic.bolus import determine_bolus


class OpenAPSController(Controller):
    def __init__(self, profile: Dict[str, Any]):
        self.profile = dict(profile)
        self._bg_hist = []  # store recent CGM values (mg/dL)
        self._last_bolus_mins = 999.0
        self._sample_time_min = 5.0
        # IOB/treatments bookkeeping (controller-local timeline)
        self._treatments: List[Dict[str, Any]] = []
        self._now: datetime | None = None
        # COB tracking: list of (datetime, carbs_g)
        self._cob_entries: List[Tuple[datetime, float]] = []
        self._absorption_hours = 3.0  # Linear absorption over 3 hours

    def _compute_cob(self) -> float:
        """Calculate Carbs On Board with linear absorption (3 hours)."""
        if self._now is None:
            return 0.0

        total_cob = 0.0

        for entry_time, carbs in self._cob_entries:
            elapsed_hours = (self._now - entry_time).total_seconds() / 3600.0

            if elapsed_hours >= self._absorption_hours:
                continue  # Fully absorbed

            # Linear decay: remaining = carbs × (1 - elapsed/total)
            remaining_fraction = max(0.0, 1.0 - (elapsed_hours / self._absorption_hours))
            total_cob += carbs * remaining_fraction

        return total_cob

    def policy(self, observation, reward, done, **info):
        # SimGlucose observation can be a namedtuple with CGM field
        try:
            glucose = float(observation.CGM)  # type: ignore[attr-defined]
        except Exception:
            glucose = float(observation)

        # Track controller time for IOB/Activity calculation
        # Init timestamp (arbitrary: realtime start). Then discrete steps of sample_time.
        if self._now is None:
            self._now = datetime.now(timezone.utc)

        # Track new meal intake
        meal_flow = float(info.get("meal", 0.0))
        sample_time = float(info.get("sample_time", self._sample_time_min))
        if meal_flow > 0:
            carbs_ingested = meal_flow * sample_time
            self._cob_entries.append((self._now, carbs_ingested))

        # Calculate persistent COB with absorption
        cob = self._compute_cob()
        self.last_cob = float(cob)

        # Determine IOB/Activity via openaps_py.iob over already delivered insulin
        from openaps_logic.iob import iob_total, iob_forecast  # local import to avoid cycles

        iobres = iob_total(self._treatments, self.profile, now=self._now)
        iob = float(iobres.iob)
        activity = float(iobres.activity)  # U/min
        self.last_iob = iob
        self.last_activity = activity
        self.last_basaliob = float(iobres.basaliob)
        self.last_bolusiob = float(iobres.bolusiob)
        self.last_netbasalinsulin = float(iobres.netbasalinsulin)
        self.last_bolusinsulin = float(iobres.bolusinsulin)

        # 4-hour IOB forecast for oref0-style prediction loop (48 ticks × 5 min)
        iob_forecast_ticks = iob_forecast(
            self._treatments, self.profile, now=self._now,
            horizon_min=240, tick_min=5,
        )

        # Compute deltas uit interne geschiedenis (5-min stap)
        self._bg_hist.append(glucose)
        if len(self._bg_hist) > 12:
            self._bg_hist.pop(0)
        deltas = [self._bg_hist[i] - self._bg_hist[i - 1] for i in range(1, len(self._bg_hist))]
        min_delta = deltas[-1] if deltas else 0.0
        short_avgdelta = sum(deltas[-3:]) / max(1, len(deltas[-3:])) if deltas else 0.0
        long_avgdelta = sum(deltas[-9:]) / max(1, len(deltas[-9:])) if deltas else 0.0

        basal_advice = determine_basal(
            glucose,
            iob,
            activity,
            self.profile,
            min_delta=min_delta,
            short_avgdelta=short_avgdelta,
            long_avgdelta=long_avgdelta,
            iob_forecast_ticks=iob_forecast_ticks,
            cob_g=cob,
        )
        # Expose the basal_advice's diagnostics on the controller for logging
        self.last_basal_advice = basal_advice
        self.last_basal_rate_u_per_hr = float(basal_advice.rate_u_per_hr)
        self.last_basal_duration_min = float(basal_advice.duration_min)
        self.last_basal_reason = basal_advice.reason
        self.last_min_guard_bg = basal_advice.min_guard_bg
        self.last_min_pred_bg = basal_advice.min_pred_bg
        self.last_eventual_bg = basal_advice.eventual_bg
        self.last_naive_eventual_bg = basal_advice.naive_eventual_bg
        self.last_deviation = basal_advice.deviation
        # sample_time via kwargs
        st = info.get("sample_time") or self._sample_time_min
        try:
            self._sample_time_min = float(st)
        except Exception:
            pass

        bolus_advice = determine_bolus(
            glucose,
            iob,
            cob,
            self.profile,
            self._last_bolus_mins,
            min_delta=min_delta,
            min_pred_bg=basal_advice.min_pred_bg,
            min_guard_bg=basal_advice.min_guard_bg,
            eventual_bg=basal_advice.eventual_bg,
            naive_eventual_bg=basal_advice.naive_eventual_bg,
        )
        self.last_bolus_advice = bolus_advice
        self.last_bolus_units = float(bolus_advice.units)
        self.last_bolus_reason = bolus_advice.reason
        self.last_smb_projection_bg = bolus_advice.smb_projection_bg
        self.last_smb_insulin_req = bolus_advice.smb_insulin_req
        self.last_smb_enabled = bool(bolus_advice.smb_enabled)
        self.last_smb_veto_reason = bolus_advice.smb_veto_reason

        # SimGlucose expects U/min; profile values are often U/hr → divide by 60
        basal_u_per_min = float(basal_advice.rate_u_per_hr) / 60.0
        # SMB-output (Units) omzetten naar U/min over de stepduur
        bolus_u_per_min = float(bolus_advice.units) / max(1.0, self._sample_time_min)
        action = Action(basal=basal_u_per_min, bolus=bolus_u_per_min)

        # Log delivered insulin as treatment for next IOB calculation
        # NB: IOB for current decision only includes previous treatments (not this one)
        try:
            # 1. Bolus treatment
            units_bolus = bolus_u_per_min * self._sample_time_min
            if units_bolus > 0:
                self._treatments.append(
                    {
                        "date": int(self._now.timestamp() * 1000),  # epoch ms
                        "insulin": float(units_bolus),
                        "type": "bolus",
                    }
                )

            # 2. Temp Basal treatment (net difference from scheduled)
            # CRITICAL FIX: OpenAPS tracks IOB based on deviation from scheduled basal.
            # Without this, high temp basals aren't seen as IOB (stacking risk)
            # and suspends aren't seen as negative IOB (slow recovery).
            scheduled_basal_u_hr = float(self.profile.get("current_basal", 0.0))
            scheduled_basal_u_min = scheduled_basal_u_hr / 60.0

            net_basal_u_min = basal_u_per_min - scheduled_basal_u_min
            units_net_basal = net_basal_u_min * self._sample_time_min

            # Only log if deviation is significant (e.g. > 0.001 U)
            # This handles both positive (high temp) and negative (suspend) IOB
            if abs(units_net_basal) > 0.001:
                self._treatments.append(
                    {
                        "date": int(self._now.timestamp() * 1000),
                        "insulin": float(units_net_basal),
                        "type": "temp_basal",
                    }
                )

        except Exception:
            pass

        # advance discrete controller time
        self._now = self._now + timedelta(minutes=self._sample_time_min)

        # update bolus-timer
        if action.bolus and action.bolus > 0:
            self._last_bolus_mins = 0.0
        else:
            self._last_bolus_mins += self._sample_time_min

        return action

    def reset(self):
        self._bg_hist.clear()
        self._last_bolus_mins = 999.0
        self._treatments.clear()
        self._cob_entries.clear()
        self._now = None
        self.last_cob = 0.0
        self.last_iob = 0.0
        self.last_activity = 0.0
        self.last_basaliob = 0.0
        self.last_bolusiob = 0.0
        self.last_netbasalinsulin = 0.0
        self.last_bolusinsulin = 0.0
        self.last_smb_projection_bg = None
        self.last_smb_insulin_req = None
        self.last_smb_enabled = False
        self.last_smb_veto_reason = None
        return None
