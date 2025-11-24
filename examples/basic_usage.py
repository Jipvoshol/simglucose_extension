"""
Basic usage example for SimGlucose Context Patch.

Demonstrates:
- Creating synthetic HR/EDA data
- Setting up context stream
- Running a simple 24-hour simulation
- Accessing context information
"""

import pandas as pd
import numpy as np
from datetime import datetime

from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario import CustomScenario
from simglucose.controller.base import Action

from simglucose_ctx.context import ContextStream, ContextConfig
from simglucose_ctx.env_wrapper import ContextAwareT1DSimEnv


def main():
    print("=" * 60)
    print("SimGlucose Context Patch - Basic Usage Example")
    print("=" * 60)

    # 1. Create synthetic HR/EDA data for 24 hours
    print("\n1. Creating synthetic HR/EDA data...")
    start_time = pd.Timestamp("2025-01-01 06:00")
    idx = pd.date_range(start_time, periods=288, freq="5min")  # 24h

    # Simple pattern: rest → moderate exercise → rest
    hr_vals = [70] * 60 + [120] * 48 + [70] * 180  # 5h rest, 4h exercise, 15h rest
    hr = pd.Series(hr_vals, index=idx)
    eda = pd.Series([0.1] * 288, index=idx)  # Low baseline stress

    print(f"   - HR range: {hr.min():.0f}-{hr.max():.0f} bpm")
    print(f"   - Duration: {len(idx)} samples ({len(idx)*5/60:.1f} hours)")

    # 2. Create context stream with default configuration
    print("\n2. Creating context stream...")
    cfg = ContextConfig()
    ctx = ContextStream(hr, eda, cfg, preprocess=False)
    print(f"   - Config: alpha={cfg.alpha}, beta={cfg.beta}")
    print(f"   - Vm0 exponent: {cfg.vm0_exponent_exercise}")

    # 3. Create SimGlucose components
    print("\n3. Setting up SimGlucose environment...")
    patient = T1DPatient.withName("adolescent#001")
    sensor = CGMSensor.withName("Dexcom", seed=1)
    pump = InsulinPump.withName("Insulet")
    scenario = CustomScenario(start_time=datetime(2025, 1, 1, 6, 0, 0), scenario=[(7, 0)])

    print(f"   - Patient: {patient.name}")
    print(f"   - Sensor: Dexcom")
    print(f"   - Scenario: Simple (no meals)")

    # 4. Create context-aware environment
    print("\n4. Creating context-aware environment...")
    env = ContextAwareT1DSimEnv(
        patient,
        sensor,
        pump,
        scenario,
        context_stream=ctx,
        modulate_vm0=True,  # Enable exercise effect
        modulate_vmx=True,  # Enable stress effect
        max_log_size=300,
    )

    # 5. Run simulation
    print("\n5. Running 24-hour simulation...")
    print("   Time     | Glucose | m(t)  | HR  | Vmx      | Vm0")
    print("   " + "-" * 60)

    env.reset()
    glucose_values = []

    for i in range(100):  # First 100 steps (~8 hours)
        # Basal-only (no bolus)
        action = Action(basal=patient._params.u2ss, bolus=0)
        obs, reward, done, info = env.step(action)

        glucose_values.append(obs.CGM)

        # Print every 12 steps (1 hour)
        if i % 12 == 0:
            current_time = start_time + pd.Timedelta(minutes=i * 5)
            print(
                f"   {current_time.strftime('%H:%M')}   | "
                f"{obs.CGM:6.1f}  | "
                f"{info['context_m']:5.2f} | "
                f"{info.get('context_hr', np.nan):3.0f} | "
                f"{info.get('Vmx', env._vmx_base):8.6f} | "
                f"{info.get('Vm0', env._vm0_base):8.4f}"
            )

        if done:
            print(f"\n   Simulation ended early at step {i}")
            break

    # 6. Show results summary
    print("\n6. Results Summary:")
    print(f"   - Mean glucose: {np.mean(glucose_values):.1f} mg/dL")
    print(f"   - Min glucose: {np.min(glucose_values):.1f} mg/dL")
    print(f"   - Max glucose: {np.max(glucose_values):.1f} mg/dL")
    print(f"   - Std glucose: {np.std(glucose_values):.1f} mg/dL")

    # 7. Get context log
    log = env.get_context_log()
    if log is not None:
        print(f"\n7. Context Log:")
        print(f"   - Logged {len(log)} samples")
        print(f"   - m(t) range: {log['m'].min():.2f} - {log['m'].max():.2f}")
        print(f"   - Mean m(t): {log['m'].mean():.2f}")

    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
