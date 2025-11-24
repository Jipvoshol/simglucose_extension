"""
Integration tests for ContextAwareT1DSimEnv.

Tests cover:
- Reset idempotency (deterministic behavior)
- Context on/off equivalence (context_stream=None)
- Multi-day stability (no NaN/Inf)
- Environment consistency
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario import CustomScenario
from simglucose.controller.base import Action

from simglucose_ctx.context import ContextStream, ContextConfig
from simglucose_ctx.env_wrapper import ContextAwareT1DSimEnv


@pytest.fixture
def patient():
    """Create a test patient"""
    return T1DPatient.withName("adolescent#001")


@pytest.fixture
def sensor():
    """Create a CGM sensor"""
    return CGMSensor.withName("Dexcom", seed=1)


@pytest.fixture
def pump():
    """Create an insulin pump"""
    return InsulinPump.withName("Insulet")


@pytest.fixture
def scenario():
    """Create a simple scenario (breakfast only)"""
    start_time = datetime(2025, 1, 1, 6, 0, 0)
    return CustomScenario(start_time=start_time, scenario=[(7, 45)])  # Breakfast at 7am


@pytest.fixture
def context_stream():
    """Create a simple context stream for testing"""
    start = pd.Timestamp("2025-01-01 06:00")
    idx = pd.date_range(start, periods=288, freq="5min")  # 24 hours

    # Simple pattern: low HR (rest), then exercise spike, then back to rest
    hr_vals = [70] * 60 + [140] * 48 + [70] * 180  # 5h rest, 4h exercise, 15h rest
    hr = pd.Series(hr_vals, index=idx)
    eda = pd.Series([0.1] * 288, index=idx)

    return ContextStream(hr, eda, preprocess=False)


class TestResetIdempotency:
    """Test that reset() produces identical results"""

    def test_reset_produces_identical_results(
        self, patient, sensor, pump, scenario, context_stream
    ):
        """Multiple resets should give same results (deterministic)"""
        env = ContextAwareT1DSimEnv(
            patient,
            sensor,
            pump,
            scenario,
            context_stream=context_stream,
            modulate_vm0=True,
            modulate_vmx=True,
        )

        # Run 1
        env.reset()
        results1 = []
        for _ in range(50):
            action = Action(basal=patient._params.u2ss, bolus=0)
            obs, reward, done, info = env.step(action)
            results1.append(obs.CGM)  # Fixed: use CGM not Gsub
            if done:
                break

        # Run 2 (same seed implied by patient seed)
        env.reset()
        results2 = []
        for _ in range(50):
            action = Action(basal=patient._params.u2ss, bolus=0)
            obs, reward, done, info = env.step(action)
            results2.append(obs.CGM)  # Fixed: use CGM not Gsub
            if done:
                break

        # Should be identical
        np.testing.assert_array_almost_equal(results1, results2, decimal=6)

    def test_context_log_cleared_on_reset(self, patient, sensor, pump, scenario, context_stream):
        """Context log should be cleared after reset"""
        env = ContextAwareT1DSimEnv(patient, sensor, pump, scenario, context_stream=context_stream)

        # Run for 20 steps (each step = 3 mini_steps by default)
        env.reset()
        for _ in range(20):
            action = Action(basal=patient._params.u2ss, bolus=0)
            env.step(action)

        log1 = env.get_context_log()
        assert log1 is not None
        # Each step consists of 3 mini_steps (sample_time=5min / patient.SAMPLE_TIME=1min)
        # But logging happens per mini_step, so we expect ~60 entries
        assert len(log1) > 0  # Just check it has data

        # Reset and run again
        env.reset()
        for _ in range(10):
            action = Action(basal=patient._params.u2ss, bolus=0)
            env.step(action)

        log2 = env.get_context_log()
        assert log2 is not None
        # Should be new log (shorter than log1)
        assert len(log2) < len(log1)


class TestContextOnOffEquivalence:
    """Test that context_stream=None behaves like original SimGlucose"""

    def test_none_context_no_modulation(self, patient, sensor, pump, scenario):
        """context_stream=None should not modulate parameters"""
        env = ContextAwareT1DSimEnv(
            patient,
            sensor,
            pump,
            scenario,
            context_stream=None,
            modulate_vm0=True,
            modulate_vmx=True,
        )

        env.reset()

        # Check that baselines are unchanged
        vmx_base = env._vmx_base
        vm0_base = env._vm0_base

        # Step once
        action = Action(basal=patient._params.u2ss, bolus=0)
        env.step(action)

        # Parameters should still be at baseline
        assert patient._params.Vmx == vmx_base
        assert patient._params.Vm0 == vm0_base

    def test_context_log_empty_when_none(self, patient, sensor, pump, scenario):
        """get_context_log() should return None if no context stream"""
        env = ContextAwareT1DSimEnv(patient, sensor, pump, scenario, context_stream=None)

        env.reset()
        for _ in range(10):
            action = Action(basal=patient._params.u2ss, bolus=0)
            env.step(action)

        log = env.get_context_log()
        assert log is None or len(log) == 0


class TestMultiDayStability:
    """Test numerical stability over long simulations"""

    def test_24hour_simulation_no_nan(self, patient, sensor, pump, scenario, context_stream):
        """24-hour simulation should not produce NaN or Inf"""
        env = ContextAwareT1DSimEnv(
            patient,
            sensor,
            pump,
            scenario,
            context_stream=context_stream,
            modulate_vm0=True,
            modulate_vmx=True,
        )

        env.reset()
        glucose_values = []

        # Run for 24 hours (288 steps at 5min intervals)
        for _ in range(288):
            action = Action(basal=patient._params.u2ss, bolus=0)
            obs, reward, done, info = env.step(action)
            glucose_values.append(obs.CGM)  # Fixed: use CGM not Gsub

            if done:
                break

        # Check for NaN/Inf
        assert not any(np.isnan(glucose_values)), "NaN detected in glucose values"
        assert not any(np.isinf(glucose_values)), "Inf detected in glucose values"

        # Check physiological bounds (glucose should stay in [0, 600] mg/dL)
        assert all(
            0 <= g <= 600 for g in glucose_values
        ), f"Glucose out of physiological range: {min(glucose_values):.1f} - {max(glucose_values):.1f}"

    def test_buffer_limit_respected(self, patient, sensor, pump, scenario):
        """Log buffer should not exceed max_log_size"""
        # Create long context stream (48 hours)
        start = pd.Timestamp("2025-01-01 06:00")
        idx = pd.date_range(start, periods=576, freq="5min")  # 48 hours
        hr = pd.Series([70] * 576, index=idx)
        eda = pd.Series([0.1] * 576, index=idx)
        ctx = ContextStream(hr, eda, preprocess=False)

        env = ContextAwareT1DSimEnv(
            patient, sensor, pump, scenario, context_stream=ctx, max_log_size=100  # Small buffer
        )

        env.reset()

        # Run for 200 steps (more than buffer size)
        for _ in range(200):
            action = Action(basal=patient._params.u2ss, bolus=0)
            obs, reward, done, info = env.step(action)
            if done:
                break

        log = env.get_context_log()

        # Should only keep most recent 100 entries
        assert len(log) <= 100, f"Buffer size exceeded: {len(log)}"


class TestEnvironmentConsistency:
    """Test environment maintains consistency"""

    def test_baseline_params_restored_on_reset(
        self, patient, sensor, pump, scenario, context_stream
    ):
        """Baseline parameters should be restored after reset"""
        env = ContextAwareT1DSimEnv(
            patient,
            sensor,
            pump,
            scenario,
            context_stream=context_stream,
            modulate_vm0=True,
            modulate_vmx=True,
        )

        # Store original baselines
        vmx_orig = env._vmx_base
        vm0_orig = env._vm0_base

        # Run simulation (parameters will be modulated)
        env.reset()
        for _ in range(50):
            action = Action(basal=patient._params.u2ss, bolus=0)
            env.step(action)

        # Reset again
        env.reset()

        # Baselines should be unchanged
        assert env._vmx_base == vmx_orig
        assert env._vm0_base == vm0_orig

        # Current parameters should be reset to baseline
        assert patient._params.Vmx == vmx_orig
        assert patient._params.Vm0 == vm0_orig

    def test_info_dict_contains_context(self, patient, sensor, pump, scenario, context_stream):
        """step() info dict should contain context information"""
        env = ContextAwareT1DSimEnv(patient, sensor, pump, scenario, context_stream=context_stream)

        env.reset()
        action = Action(basal=patient._params.u2ss, bolus=0)
        obs, reward, done, info = env.step(action)

        # Check info dict has context fields
        assert "context_m" in info
        assert "context_hr" in info
        assert "context_eda" in info

        # Values should be valid
        assert isinstance(info["context_m"], float)
        assert 0.6 <= info["context_m"] <= 2.5
