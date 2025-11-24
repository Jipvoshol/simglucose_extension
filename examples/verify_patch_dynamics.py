import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# Ensure we can import simglucose_ctx
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simglucose_ctx.context import ContextStream, ContextConfig
from simglucose_ctx.sim_hook import apply_vmx_multiplier

# 1. Mock Patient
class MockParams:
    def __init__(self, Vmx):
        self.Vmx = Vmx

class MockPatient:
    def __init__(self, Vmx_base=50.0):
        self._params = MockParams(Vmx_base)
        self.name = "MockPatient"

# 2. Setup Scenario
def run_verification():
    print("Running Patch Dynamics Verification...")
    
    # Time: 4 hours, 1 min resolution
    start_time = pd.Timestamp('2025-01-01 08:00:00')
    duration_hours = 4
    freq = '1min'
    time_index = pd.date_range(start=start_time, periods=duration_hours*60, freq=freq)
    
    # Synthetic HR: 
    # 08:00 - 09:00: Rest (60 bpm)
    # 09:00 - 10:00: Exercise (150 bpm)
    # 10:00 - 12:00: Rest (60 bpm)
    hr_values = np.full(len(time_index), 60.0)
    exercise_mask = (time_index >= start_time + timedelta(hours=1)) & \
                    (time_index < start_time + timedelta(hours=2))
    hr_values[exercise_mask] = 150.0
    
    hr_series = pd.Series(hr_values, index=time_index)
    
    # Synthetic EDA (Constant low stress for isolation)
    eda_series = pd.Series(np.full(len(time_index), 0.1), index=time_index)
    
    # Initialize Context
    # Use small dt_minutes to see smooth updates if we call it every step
    cfg = ContextConfig(dt_minutes=5, ema_half_life_min=15.0)
    ctx = ContextStream(hr_series, eda_series, cfg, preprocess=True)
    
    patient = MockPatient(Vmx_base=50.0)
    vmx_base = 50.0
    
    results = {
        'time': [],
        'hr': [],
        'm': [],
        'vmx': [],
        'vm0': []
    }
    
    # Simulation Loop (step by step, e.g. every 1 minute)
    # Note: ContextStream resamples to 5min internally, but we can query anytime.
    # Ideally we query at the 5min grid points or just every minute.
    # Let's query every minute to see the interpolation/step behavior.
    
    for t in time_index:
    # Hook call
        # We pass None for m_value so it calculates it fresh
        from simglucose_ctx.sim_hook import apply_vm0_multiplier
        
        # Test Vmx (should be minimal)
        vmx_eff = apply_vmx_multiplier(patient, t, ctx, vmx_base)
        m_vmx = vmx_eff / vmx_base
        
        # Test Vm0 (should be massive)
        vm0_base = 5.0
        vm0_eff = apply_vm0_multiplier(patient, t, ctx, vm0_base)
        m_vm0 = vm0_eff / vm0_base
        
        results['time'].append(t)
        results['hr'].append(hr_series.loc[t])
        results['m'].append(m_vmx) # Log the Vmx multiplier as 'm' for continuity
        results['vmx'].append(vmx_eff)
        results['vm0'].append(vm0_eff)
        
    # Plotting
    df_res = pd.DataFrame(results).set_index('time')
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    # Plot 1: HR Input
    axes[0].plot(df_res.index, df_res['hr'], color='blue', label='Heart Rate (bpm)')
    axes[0].set_ylabel('HR (bpm)')
    axes[0].set_title('Input: Synthetic Heart Rate Profile')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Vmx Multiplier (Minimal)
    axes[1].plot(df_res.index, df_res['m'], color='purple', label='Vmx Multiplier (m^0.2)')
    axes[1].axhline(1.0, color='gray', linestyle='--')
    axes[1].set_ylabel('Factor')
    axes[1].set_title('Vmx Scaling: Minimal (Hybrid Strategy)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot 3: Vm0 Multiplier (Massive)
    axes[2].plot(df_res.index, df_res['vm0'] / 5.0, color='red', label='Vm0 Multiplier (m^3.5)')
    axes[2].axhline(1.0, color='gray', linestyle='--')
    axes[2].set_ylabel('Factor')
    axes[2].set_title('Vm0 Scaling: Massive AMPK Boost')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Plot 4: Effective Parameters
    axes[3].plot(df_res.index, df_res['vmx'], color='purple', label='Vmx')
    axes[3].plot(df_res.index, df_res['vm0'], color='red', label='Vm0')
    axes[3].set_ylabel('Value')
    axes[3].set_xlabel('Time')
    axes[3].set_title('Absolute Parameter Values')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    
    plt.tight_layout()
    output_path = 'verify_dynamics_plot.png'
    plt.savefig(output_path)
    print(f"Verification plot saved to {output_path}")
    
    # Basic Assertions
    max_m_vmx = df_res['m'].max()
    max_m_vm0 = (df_res['vm0'] / 5.0).max()
    
    print(f"Max Vmx Factor: {max_m_vmx:.3f} (Expected ~1.1-1.2)")
    print(f"Max Vm0 Factor: {max_m_vm0:.3f} (Expected > 8.0)")
    
    if max_m_vmx < 1.3 and max_m_vm0 > 8.0:
        print("SUCCESS: Hybrid Strategy Verified (Vmx stable, Vm0 boosted).")
    else:
        print("WARNING: Scaling factors not matching hybrid strategy expectations.")

if __name__ == "__main__":
    run_verification()
