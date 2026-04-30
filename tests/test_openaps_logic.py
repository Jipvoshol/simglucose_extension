"""
Unit tests for OpenAPS logic (basal, bolus, IOB).

Tests cover core decision paths without requiring simglucose.
"""

import pytest
from datetime import datetime, timedelta, timezone

from openaps_logic.basal import determine_basal, BasalAdvice
from openaps_logic.bolus import determine_bolus, BolusAdvice
from openaps_logic.iob import iob_total, IOBTotal
from simglucose_ctx.openaps_controller import OpenAPSController


# ---------------------------------------------------------------------------
# Shared test profile
# ---------------------------------------------------------------------------

@pytest.fixture
def profile():
    return {
        "current_basal": 1.0,  # U/hr
        "sens": 50.0,  # mg/dL per U
        "min_bg": 90,
        "max_bg": 120,
        "max_iob": 3.0,
        "dia": 6.0,
        "curve": "bilinear",
        "temp_basal_duration_min": 30,
        "max_safe_basal": 4.0,
        "bolus_increment": 0.1,
        "maxSMBBasalMinutes": 30,
        "SMBInterval": 3,
        "enableSMB_always": False,
        "enableSMB_with_COB": True,
        "enableSMB_high_bg": True,
        "enableSMB_high_bg_target": 160,
        "maxDelta_bg_threshold": 0.2,
    }


# ===========================================================================
# determine_basal tests
# ===========================================================================

class TestDetermineBasal:
    """Test core temp basal logic."""

    def test_low_glucose_suspends(self, profile):
        """When glucose is below threshold, basal should be suspended."""
        advice = determine_basal(60.0, iob_u=0.0, activity_u_per_min=0.0, profile=profile)
        assert advice.rate_u_per_hr == 0.0
        assert "low-suspend" in advice.reason

    def test_very_low_glucose_suspends(self, profile):
        """Even with IOB, very low glucose should suspend."""
        advice = determine_basal(50.0, iob_u=2.0, activity_u_per_min=0.001, profile=profile)
        assert advice.rate_u_per_hr == 0.0

    def test_in_range_returns_current_basal(self, profile):
        """Glucose in range with no IOB should return current basal."""
        advice = determine_basal(105.0, iob_u=0.0, activity_u_per_min=0.0, profile=profile)
        assert advice.rate_u_per_hr == profile["current_basal"]
        assert advice.reason in ("in-range", "rising>expected")

    def test_high_glucose_increases_rate(self, profile):
        """High glucose should increase temp basal rate."""
        advice = determine_basal(200.0, iob_u=0.0, activity_u_per_min=0.0, profile=profile)
        assert advice.rate_u_per_hr > profile["current_basal"]
        assert "high-temp" in advice.reason

    def test_high_glucose_capped_by_max_safe(self, profile):
        """Temp basal should never exceed max_safe_basal."""
        advice = determine_basal(400.0, iob_u=0.0, activity_u_per_min=0.0, profile=profile)
        assert advice.rate_u_per_hr <= profile["max_safe_basal"]

    def test_high_glucose_capped_by_max_iob(self, profile):
        """When IOB is at max, no additional insulin should be requested."""
        advice = determine_basal(200.0, iob_u=3.0, activity_u_per_min=0.0, profile=profile)
        # Should be at or near current basal since max_iob is reached
        assert advice.rate_u_per_hr <= profile["current_basal"] + 0.01

    def test_returns_basal_advice(self, profile):
        """Output should always be a BasalAdvice dataclass."""
        advice = determine_basal(120.0, iob_u=0.5, activity_u_per_min=0.0, profile=profile)
        assert isinstance(advice, BasalAdvice)
        assert isinstance(advice.rate_u_per_hr, float)
        assert isinstance(advice.duration_min, int)
        assert advice.duration_min > 0

    def test_eventual_bg_below_threshold_suspends(self, profile):
        """If eventual BG (glucose - IOB*sens) is low, should suspend."""
        # glucose=100, IOB=2.0, sens=50 → eventual = 100 - 100 = 0 (below threshold)
        advice = determine_basal(100.0, iob_u=2.0, activity_u_per_min=0.0, profile=profile)
        assert advice.rate_u_per_hr == 0.0


# ===========================================================================
# determine_bolus (SMB) tests
# ===========================================================================

class TestDetermineBolus:
    """Test SMB decision logic."""

    def test_low_glucose_no_smb(self, profile):
        """No SMB when glucose is below threshold."""
        advice = determine_bolus(60.0, iob_u=0.0, cob_g=30.0, profile=profile)
        assert advice.units == 0.0

    def test_max_iob_no_smb(self, profile):
        """No SMB when IOB exceeds max_iob."""
        advice = determine_bolus(200.0, iob_u=4.0, cob_g=30.0, profile=profile)
        assert advice.units == 0.0

    def test_smb_interval_respected(self, profile):
        """No SMB if last bolus was too recent."""
        advice = determine_bolus(
            200.0, iob_u=0.0, cob_g=30.0, profile=profile,
            minutes_since_last_bolus=1.0,
        )
        assert advice.units == 0.0

    def test_smb_with_cob_and_high_bg(self, profile):
        """SMB should fire when COB present and glucose is high."""
        advice = determine_bolus(
            200.0, iob_u=0.0, cob_g=30.0, profile=profile,
            minutes_since_last_bolus=10.0,
        )
        assert advice.units > 0.0

    def test_smb_high_bg_enabled(self, profile):
        """SMB should fire on high BG even without COB when enableSMB_high_bg=True."""
        advice = determine_bolus(
            180.0, iob_u=0.0, cob_g=0.0, profile=profile,
            minutes_since_last_bolus=10.0,
        )
        assert advice.units > 0.0

    def test_no_smb_in_range_no_cob(self, profile):
        """No SMB when glucose is in range and no COB."""
        advice = determine_bolus(
            110.0, iob_u=0.0, cob_g=0.0, profile=profile,
            minutes_since_last_bolus=10.0,
        )
        assert advice.units == 0.0

    def test_smb_increment_rounding(self, profile):
        """SMB should be rounded to bolus_increment."""
        advice = determine_bolus(
            250.0, iob_u=0.0, cob_g=50.0, profile=profile,
            minutes_since_last_bolus=10.0,
        )
        # Should be a multiple of 0.1
        assert abs(advice.units * 10 - round(advice.units * 10)) < 1e-9

    def test_returns_bolus_advice(self, profile):
        """Output should always be a BolusAdvice dataclass."""
        advice = determine_bolus(120.0, iob_u=0.5, cob_g=0.0, profile=profile)
        assert isinstance(advice, BolusAdvice)
        assert isinstance(advice.units, float)
        assert advice.units >= 0.0

    def test_spike_protection(self, profile):
        """Large min_delta should trigger spike protection."""
        # glucose=150, min_delta=50 → 50 > 0.2*150=30 → spike
        advice = determine_bolus(
            150.0, iob_u=0.0, cob_g=30.0, profile=profile,
            minutes_since_last_bolus=10.0, min_delta=50.0,
        )
        assert advice.units == 0.0
        assert "spike" in advice.reason

    def test_forecast_below_target_suppresses_smb_despite_high_current_bg(self, profile):
        """Forecast-aware SMB should not anchor on current BG when predictions are safe/low."""
        profile = dict(profile, enableSMB_always=True)
        advice = determine_bolus(
            220.0,
            iob_u=0.0,
            cob_g=0.0,
            profile=profile,
            minutes_since_last_bolus=10.0,
            min_pred_bg=95.0,
            min_guard_bg=85.0,
            eventual_bg=100.0,
        )
        assert advice.units == 0.0
        assert advice.smb_veto_reason == "forecast_not_high"

    def test_min_guard_bg_below_threshold_vetoes_smb(self, profile):
        """Mirror oref0's minGuardBG safety veto for SMB."""
        profile = dict(profile, enableSMB_always=True)
        advice = determine_bolus(
            220.0,
            iob_u=0.0,
            cob_g=0.0,
            profile=profile,
            minutes_since_last_bolus=10.0,
            min_pred_bg=180.0,
            min_guard_bg=60.0,
            eventual_bg=190.0,
        )
        assert advice.units == 0.0
        assert advice.smb_veto_reason == "min_guard_bg_below_threshold"

    def test_forecast_above_target_doses_smb_with_cap_and_increment(self, profile):
        """Forecast-aware SMB uses min(minPredBG, eventualBG), cap, and bolus increment."""
        profile = dict(profile, enableSMB_always=True)
        advice = determine_bolus(
            220.0,
            iob_u=0.0,
            cob_g=0.0,
            profile=profile,
            minutes_since_last_bolus=10.0,
            min_pred_bg=180.0,
            min_guard_bg=100.0,
            eventual_bg=190.0,
        )
        assert advice.units == 0.5
        assert advice.smb_projection_bg == 180.0
        assert advice.smb_insulin_req == pytest.approx(1.5)

    def test_higher_sens_reduces_forecast_smb_insulin_req(self, profile):
        """The patched ISF should propagate through SMB insulinReq, not only basal."""
        p50 = dict(profile, sens=50.0, current_basal=2.0, maxSMBBasalMinutes=60, enableSMB_always=True)
        p100 = dict(p50, sens=100.0)
        common = dict(
            glucose_mgdl=220.0,
            iob_u=0.0,
            cob_g=0.0,
            minutes_since_last_bolus=10.0,
            min_pred_bg=180.0,
            min_guard_bg=100.0,
            eventual_bg=190.0,
        )
        adv50 = determine_bolus(profile=p50, **common)
        adv100 = determine_bolus(profile=p100, **common)
        assert adv100.smb_insulin_req < adv50.smb_insulin_req
        assert adv100.units < adv50.units

    def test_high_temp_target_disables_smb_unless_allowed(self, profile):
        """oref0 disables SMB under high temp target unless explicitly allowed."""
        high_tt = dict(
            profile,
            min_bg=140,
            max_bg=160,
            temptargetSet=True,
            enableSMB_always=True,
            allowSMB_with_high_temptarget=False,
        )
        advice = determine_bolus(
            240.0,
            iob_u=0.0,
            cob_g=0.0,
            profile=high_tt,
            minutes_since_last_bolus=10.0,
            min_pred_bg=220.0,
            min_guard_bg=150.0,
            eventual_bg=230.0,
        )
        assert advice.units == 0.0
        assert advice.smb_veto_reason == "high_temp_target"

        allowed = dict(high_tt, allowSMB_with_high_temptarget=True)
        allowed_advice = determine_bolus(
            240.0,
            iob_u=0.0,
            cob_g=0.0,
            profile=allowed,
            minutes_since_last_bolus=10.0,
            min_pred_bg=220.0,
            min_guard_bg=150.0,
            eventual_bg=230.0,
        )
        assert allowed_advice.units > 0.0

    def test_controller_passes_basal_forecast_diagnostics_to_bolus(self, profile, monkeypatch):
        """OpenAPSController should pass basal minPred/minGuard into determine_bolus."""
        captured = {}

        def fake_bolus(*args, **kwargs):
            captured.update(kwargs)
            return BolusAdvice(0.0, reason="captured")

        monkeypatch.setattr(
            "simglucose_ctx.openaps_controller.determine_bolus",
            fake_bolus,
        )
        ctrl = OpenAPSController(profile)

        class Obs:
            CGM = 180.0

        ctrl.policy(Obs(), reward=0.0, done=False, sample_time=5.0)
        assert "min_pred_bg" in captured
        assert "min_guard_bg" in captured
        assert "eventual_bg" in captured
        assert captured["min_pred_bg"] is not None
        assert captured["min_guard_bg"] is not None


# ===========================================================================
# IOB calculation tests
# ===========================================================================

class TestIOBTotal:
    """Test IOB tracking with bilinear and exponential curves."""

    def test_no_treatments_zero_iob(self, profile):
        """No treatments should give zero IOB."""
        now = datetime.now(timezone.utc)
        result = iob_total([], profile, now=now)
        assert result.iob == 0.0
        assert result.activity == 0.0

    def test_recent_bolus_has_iob(self, profile):
        """A recent bolus should have remaining IOB."""
        now = datetime.now(timezone.utc)
        treatment_time = now - timedelta(minutes=30)
        treatments = [{"date": int(treatment_time.timestamp() * 1000), "insulin": 2.0}]

        result = iob_total(treatments, profile, now=now)
        assert result.iob > 0.0, "30-min-old bolus should have remaining IOB"
        assert result.iob < 2.0, "IOB should be less than original dose"

    def test_old_bolus_no_iob(self, profile):
        """A bolus older than DIA should have zero IOB."""
        now = datetime.now(timezone.utc)
        treatment_time = now - timedelta(hours=7)  # DIA=6h, so this is expired
        treatments = [{"date": int(treatment_time.timestamp() * 1000), "insulin": 2.0}]

        result = iob_total(treatments, profile, now=now)
        assert result.iob == 0.0

    def test_iob_decreases_over_time(self, profile):
        """IOB should decrease monotonically over time."""
        base_time = datetime.now(timezone.utc)
        treatment_time = base_time - timedelta(minutes=10)
        treatments = [{"date": int(treatment_time.timestamp() * 1000), "insulin": 3.0}]

        iobs = []
        for minutes_later in range(0, 180, 15):
            now = base_time + timedelta(minutes=minutes_later)
            result = iob_total(treatments, profile, now=now)
            iobs.append(result.iob)

        # IOB should generally decrease (allow small floating point noise)
        for i in range(1, len(iobs)):
            assert iobs[i] <= iobs[i - 1] + 0.01, (
                f"IOB increased at step {i}: {iobs[i-1]:.3f} -> {iobs[i]:.3f}"
            )

    def test_multiple_treatments_accumulate(self, profile):
        """Multiple treatments should accumulate IOB."""
        now = datetime.now(timezone.utc)
        treatments = [
            {"date": int((now - timedelta(minutes=20)).timestamp() * 1000), "insulin": 1.0},
            {"date": int((now - timedelta(minutes=10)).timestamp() * 1000), "insulin": 1.5},
        ]

        result = iob_total(treatments, profile, now=now)
        # Combined IOB should be positive and meaningful
        assert result.iob > 0.5

    def test_returns_iobtotal(self, profile):
        """Output should always be an IOBTotal dataclass."""
        now = datetime.now(timezone.utc)
        result = iob_total([], profile, now=now)
        assert isinstance(result, IOBTotal)
        assert isinstance(result.iob, float)
        assert isinstance(result.activity, float)
        assert isinstance(result.time, datetime)

    def test_exponential_curve(self, profile):
        """Test IOB with rapid-acting exponential curve."""
        profile_exp = dict(profile)
        profile_exp["curve"] = "rapid-acting"
        profile_exp["dia"] = 6.0

        now = datetime.now(timezone.utc)
        treatment_time = now - timedelta(minutes=60)
        treatments = [{"date": int(treatment_time.timestamp() * 1000), "insulin": 2.0}]

        result = iob_total(treatments, profile_exp, now=now)
        assert result.iob > 0.0
        assert result.iob < 2.0

    def test_activity_positive_for_recent_treatment(self, profile):
        """Recent treatment should have positive activity (insulin action rate)."""
        now = datetime.now(timezone.utc)
        treatment_time = now - timedelta(minutes=45)
        treatments = [{"date": int(treatment_time.timestamp() * 1000), "insulin": 2.0}]

        result = iob_total(treatments, profile, now=now)
        assert result.activity > 0.0

    def test_none_treatment_skipped(self, profile):
        """None entries in treatment list should be skipped gracefully."""
        now = datetime.now(timezone.utc)
        treatments = [None, {"date": int(now.timestamp() * 1000), "insulin": 1.0}, None]
        result = iob_total(treatments, profile, now=now)
        # Should not crash
        assert isinstance(result, IOBTotal)
