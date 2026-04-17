import os
import sys
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Fix relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../simglucose')))

from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario import CustomScenario
from simglucose_ctx.env_wrapper import ContextAwareT1DSimEnv
from simglucose_ctx.context import ContextConfig, ContextStream
from simglucose_ctx.context_aware_openaps import ContextAwareOpenAPSController

print("=========================================================")
print("THESIS EXPERIMENT: CLOSED-LOOP CAUSAL EVALUATION - 2307")
print("Patch: manchester/F9_cap25_patch (cap_25 preprocessing)")
print("=========================================================")

# ============================================================
# 1. SETUP CONTEXT SCENARIO (Simulated PID 2306 pattern)
# ============================================================
# IMPORTANT: ContextStream expects ABSOLUTE heart rate and raw EDA.
# The ContextAwareOpenAPSController handles the conversion to
# HR_norm (relative) and stress_level internally.

start_time = datetime(2025, 6, 1, 0, 0, 0)
time_index = pd.date_range(start_time, periods=480, freq="3min")  # 24 hours

np.random.seed(42)

# --- HR: absolute heart rate in BPM ---
HR_REST = 81.5  # Resting heart rate for 2307 (from per_patient/2307_profile.md)
hr_data = pd.Series(HR_REST + np.random.randn(480) * 3, index=time_index)

# --- EDA/Stress: On a 0-100 stress_level scale (matching Manchester) ---
# Manchester uses Garmin stress_level (0-100), NOT raw EDA in µS
stress_base = pd.Series(15 + np.random.randn(480) * 3, index=time_index)  # Low baseline stress

# Event 1: Morning Run (08:00 - 09:30) — HR up, stress stays low
hr_data.iloc[160:190] += 65   # HR: 65 → 130 bpm (vigorous exercise)
stress_base.iloc[160:190] += 10  # Mild stress response during exercise

# Event 2: Afternoon High Stress / Exam (14:00 - 16:00) — moderate HR, HIGH stress
hr_data.iloc[280:320] += 25   # HR: 65 → 90 bpm (elevated but not exercise)
stress_base.iloc[280:320] += 60  # Stress: 15 → 75 (high stress on 0-100 scale)

# Event 3: Evening Walk (20:00 - 21:00) — moderate HR, low stress
hr_data.iloc[400:420] += 30   # HR: 65 → 95 bpm
stress_base.iloc[400:420] += 5   # Minimal stress during walk

cfg = ContextConfig(
    alpha=1.2, beta=0.8, mmax=2.5, mmin=0.6,
    hr_rest=HR_REST, hr_max=185.0, ema_half_life_min=5.0
)
ctx = ContextStream(hr_data, stress_base, cfg, preprocess=False)

# Meals: Breakfast (07:30 - before run), Lunch (12:30), Dinner (19:00)
meals = [
    (datetime(2025, 6, 1, 7, 30, 0), 45),
    (datetime(2025, 6, 1, 12, 30, 0), 60),
    (datetime(2025, 6, 1, 19, 0, 0), 75),
]
scenario = CustomScenario(start_time=start_time, scenario=meals)


# ============================================================
# 2. RUN SIMULATION FUNCTION
# ============================================================
def run_simulation(arm_name, use_context_in_env=True, controller_mode='baseline'):
    print(f"\nRunning {arm_name} (Mode: {controller_mode})...")

    patient = T1DPatient.withName("adolescent#001", seed=10)
    sensor = CGMSensor.withName("Dexcom", seed=10)
    pump = InsulinPump.withName("Insulet")

    env_ctx = ctx if use_context_in_env else None
    env = ContextAwareT1DSimEnv(
        patient, sensor, pump, scenario,
        context_stream=env_ctx, modulate_vm0=True, modulate_vmx=True
    )

    # Standard OpenAPS Profile
    profile = {
        'max_iob': 4.0, 'max_daily_basal': 2.5, 'current_basal': 0.8,
        'max_basal': 3.0, 'dia': 4.0, 'sens': 50.0, 'target_bg': 100,
        'min_bg': 95, 'max_bg': 105, 'carb_ratio': 10.0
    }

    # 2307-specific patch trained on cap_25 preprocessing
    patch_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '../../manchester/F9_cap25_patch/patch/isf_patch_2307.pkl'))

    if controller_mode == 'baseline':
        controller = ContextAwareOpenAPSController(profile, model_path=None, hr_rest=HR_REST)
    elif controller_mode == 'patch':
        controller = ContextAwareOpenAPSController(
            profile, model_path=patch_path, placebo_mode=False, hr_rest=HR_REST)
    elif controller_mode == 'placebo':
        controller = ContextAwareOpenAPSController(
            profile, model_path=patch_path, placebo_mode=True, hr_rest=HR_REST)

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    results = []

    for _ in range(480):
        info_dict = {"sample_time": 3.0}

        if use_context_in_env:
            current_time = env.env.time if hasattr(env, 'env') else env.time
            ts = pd.Timestamp(current_time)
            idx_index = ctx.hr.index
            pos = idx_index.searchsorted(ts, side="right")
            idx = idx_index[0] if pos == 0 else idx_index[pos - 1]

            # Pass ABSOLUTE HR and stress_level (0-100) to the controller.
            # The controller handles conversion to HR_norm = HR - hr_rest internally.
            info_dict['context_hr'] = float(ctx.hr.loc[idx])
            info_dict['context_eda'] = float(ctx.eda.loc[idx])
            # Flag: EDA is stress_level (0-100), not raw µS
            info_dict['context_eda_is_stress_level'] = True
            # Flag: HR is absolute (bpm), not already relative
            info_dict['context_hr_is_relative'] = False

        action = controller.policy(obs, reward=0, done=False, **info_dict)
        step_result = env.step(action)

        bg = float(step_result.observation.CGM) if hasattr(step_result.observation, 'CGM') else float(step_result.observation)
        # m_t stored on controller after policy call (see context_aware_openaps.py)
        m_t = getattr(controller, 'last_m_t', 1.0)

        results.append({
            'time': env.env.time if hasattr(env, 'env') else env.time,
            'glucose': bg,
            'basal': float(action.basal),
            'bolus': float(action.bolus),
            'm_t': m_t,
            'hr': info_dict.get('context_hr', HR_REST),
            'stress': info_dict.get('context_eda', 15.0)
        })
        obs = step_result.observation

    df = pd.DataFrame(results).set_index('time')
    df['insulin'] = df['basal'] + df['bolus']
    lam = np.exp(-3.0 / (240.0 / 3.0))  # DIA 4h decay
    iob = []
    remain = 0.0
    for r in df['insulin'].values:
        remain = remain * lam + (r * 3.0)
        iob.append(remain)
    df['IOB'] = iob

    return df


# ============================================================
# 3. EXECUTE ARMS
# ============================================================
df_A = run_simulation("Arm A (Blind OpenAPS)", controller_mode='baseline')
df_B = run_simulation("Arm B (Proactive Patch)", controller_mode='patch')
df_C = run_simulation("Arm C (Placebo Control)", controller_mode='placebo')


# ============================================================
# 4. METRICS & PLOTTING
# ============================================================
def calc_metrics(df):
    tir = ((df['glucose'] >= 70) & (df['glucose'] <= 180)).mean() * 100
    tbr = (df['glucose'] < 70).mean() * 100
    tar = (df['glucose'] > 180).mean() * 100
    ins = df['insulin'].sum() * 3.0  # Total Units
    return tir, tbr, tar, ins


metrics_A = calc_metrics(df_A)
metrics_B = calc_metrics(df_B)
metrics_C = calc_metrics(df_C)

print("\n================ FINAL METRICS ================")
print(f"| Arm | TIR (%) | TBR (<70) | Hyper (>180) | Total Ins |")
print(f"|---|---|---|---|---|")
print(f"| Arm A (Blind)   | {metrics_A[0]:.1f} | {metrics_A[1]:.1f} | {metrics_A[2]:.1f} | {metrics_A[3]:.1f} U |")
print(f"| Arm B (Patch)   | {metrics_B[0]:.1f} | {metrics_B[1]:.1f} | {metrics_B[2]:.1f} | {metrics_B[3]:.1f} U |")
print(f"| Arm C (Placebo) | {metrics_C[0]:.1f} | {metrics_C[1]:.1f} | {metrics_C[2]:.1f} | {metrics_C[3]:.1f} U |")
print("===============================================\n")

# Plot
fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
hours = [(t - df_A.index[0]).total_seconds() / 3600 for t in df_A.index]

# Panel 1: Glucose
ax1 = axes[0]
ax1.plot(hours, df_A['glucose'], 'r--', label='Arm A (Blind)')
ax1.plot(hours, df_B['glucose'], 'g-', linewidth=2.5, label='Arm B (Patch Active)')
ax1.plot(hours, df_C['glucose'], 'k:', alpha=0.5, label='Arm C (Placebo)')
ax1.axhspan(70, 180, color='green', alpha=0.1)
ax1.axhline(70, color='red', linestyle='-', alpha=0.3)
ax1.axvspan(8, 9.5, color='orange', alpha=0.2, label='Morning Run')
ax1.axvspan(14, 16, color='purple', alpha=0.2, label='Exam Stress')
ax1.axvspan(20, 21, color='orange', alpha=0.1, label='Evening Walk')
for m in meals:
    ax1.axvline((m[0] - start_time).total_seconds() / 3600, color='blue', alpha=0.5, linestyle='-.')
ax1.set_ylabel('Glucose (mg/dL)')
ax1.set_title('Closed-Loop Causal Evaluation — Patient 2307 (cap_25 patch)')
ax1.legend()

# Panel 2: m(t) prediction
ax2 = axes[1]
ax2.plot(hours, df_B['m_t'], 'g-', label='XGBoost Predicted m(t)')
ax2.plot(hours, df_C['m_t'], 'k:', alpha=0.5, label='Placebo Random m(t)')
ax2.axhline(1.0, color='gray', linestyle='--')
ax2.set_ylabel('ISF Multiplier m(t)')
ax2.legend()

# Panel 3: HR context (for visual validation)
ax3 = axes[2]
hours_hr = [(t - hr_data.index[0]).total_seconds() / 3600 for t in hr_data.index]
ax3.plot(hours_hr, hr_data.values, 'b-', alpha=0.7, label='Heart Rate (bpm)')
ax3.axhline(HR_REST, color='gray', linestyle='--', alpha=0.5, label=f'Resting HR ({HR_REST})')
ax3.set_ylabel('Heart Rate (bpm)')
ax3.legend()

# Panel 4: IOB
ax4 = axes[3]
ax4.plot(hours, df_A['IOB'], 'r--', label='Arm A IOB')
ax4.plot(hours, df_B['IOB'], 'g-', label='Arm B IOB')
ax4.set_ylabel('Insulin On Board (U)')
ax4.set_xlabel('Time (Hours)')
ax4.legend()

plt.tight_layout()
os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
fig_path = os.path.join(os.path.dirname(__file__), 'results', 'closed_loop_2307_cap25.png')
plt.savefig(fig_path, dpi=150)
print(f"Successfully saved evaluation plot to: {fig_path}")

# Also save per-arm CSV for downstream analysis
out_dir = os.path.join(os.path.dirname(__file__), 'results')
df_A.to_csv(os.path.join(out_dir, 'closed_loop_2307_arm_A_baseline.csv'))
df_B.to_csv(os.path.join(out_dir, 'closed_loop_2307_arm_B_patch.csv'))
df_C.to_csv(os.path.join(out_dir, 'closed_loop_2307_arm_C_placebo.csv'))
print(f"Saved per-arm CSVs to {out_dir}/closed_loop_2307_arm_*.csv")
