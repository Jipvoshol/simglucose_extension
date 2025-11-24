"""
Custom HR/EDA Source Example

Shows how to create context data from various sources:
1. Synthetic data (for testing)
2. CSV files (common use case)
3. Real-time streaming (advanced)

Most users won't have real-life data, so this demonstrates practical alternatives.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_synthetic_context(start_time, duration_hours, freq='5min'):
    """
    Generate realistic HR/EDA patterns for testing.
    
    Simulates:
    - Circadian rhythm (lower HR at night)
    - Random exercise periods
    - Stress events
    - Realistic noise
    """
    print(f"Generating {duration_hours}h of synthetic HR/EDA data...")
    
    time_index = pd.date_range(start_time, periods=int(duration_hours*12), freq=freq)
    n_samples = len(time_index)
    
    # 1. Circadian HR pattern (lower at night, higher during day)
    hours = np.array([t.hour + t.minute/60 for t in time_index])
    circadian = 15 * np.sin(2 * np.pi * (hours - 6) / 24)  # Peak at 18:00, trough at 06:00
    
    # 2. Base HR with realistic noise
    hr_base = 70 + circadian + np.random.randn(n_samples) * 3
    
    # 3. Add exercise periods (random 1-2 hour sessions)
    hr = hr_base.copy()
    for _ in range(int(duration_hours / 24 * 2)):  # ~2 exercise sessions per day
        # Random exercise start (avoid nighttime 22:00-06:00)
        start_hour = np.random.randint(7, 21)
        start_idx = int(start_hour * 12)
        duration_steps = np.random.randint(12, 24)  # 1-2 hours
        
        if start_idx + duration_steps < n_samples:
            # Gradual ramp up, plateau, ramp down
            ramp_up = np.linspace(0, 60, 6)  # 30 min ramp
            plateau = np.ones(duration_steps - 12) * 60
            ramp_down = np.linspace(60, 0, 6)  # 30 min recovery
            
            exercise_profile = np.concatenate([ramp_up, plateau, ramp_down])
            hr[start_idx:start_idx+duration_steps] += exercise_profile[:duration_steps]
    
    # 4. Add stress events (shorter, EDA-based)
    eda_base = 0.1 + np.random.randn(n_samples) * 0.02  # Baseline with noise
    eda = np.clip(eda_base, 0.05, 0.15)
    
    for _ in range(int(duration_hours / 24)):  # ~1 stress event per day
        start_idx = np.random.randint(0, n_samples - 12)
        duration_steps = np.random.randint(6, 18)  # 30min - 1.5h
        eda[start_idx:start_idx+duration_steps] += 0.5 + np.random.rand() * 0.3
    
    # 5. Clip to physiological ranges
    hr = np.clip(hr, 45, 200)
    eda = np.clip(eda, 0.05, 2.0)
    
    hr_series = pd.Series(hr, index=time_index)
    eda_series = pd.Series(eda, index=time_index)
    
    print(f"  ✓ Generated {n_samples} samples")
    print(f"  ✓ HR range: {hr.min():.0f}-{hr.max():.0f} bpm")
    print(f"  ✓ EDA range: {eda.min():.3f}-{eda.max():.3f} μS")
    
    return hr_series, eda_series


def load_from_csv(csv_path, time_col='timestamp', hr_col='hr', eda_col='eda'):
    """
    Load HR/EDA from generic CSV file.
    
    Expected format:
        timestamp,hr,eda
        2025-01-01 06:00:00,70,0.1
        2025-01-01 06:05:00,72,0.11
        ...
    
    Handles:
    - Missing data (interpolation)
    - Non-standard column names
    - Timezone issues
    """
    print(f"Loading HR/EDA from CSV: {csv_path}")
    
    df = pd.read_csv(csv_path, parse_dates=[time_col])
    df = df.set_index(time_col)
    df = df.sort_index()
    
    # Extract HR and EDA
    hr = df[hr_col]
    eda = df[eda_col]
    
    # Handle missing data
    hr_missing = hr.isna().sum()
    eda_missing = eda.isna().sum()
    
    if hr_missing > 0 or eda_missing > 0:
        print(f"  ⚠ Missing data: HR={hr_missing}, EDA={eda_missing}")
        print(f"  ✓ Interpolating gaps (linear, max 3 consecutive NaNs)...")
        hr = hr.interpolate(method='linear', limit=3)
        eda = eda.interpolate(method='linear', limit=3)
        
        # Fill remaining NaNs with defaults
        hr = hr.fillna(70.0)
        eda = eda.fillna(0.1)
    
    print(f"  ✓ Loaded {len(hr)} samples")
    print(f"  ✓ Time range: {hr.index[0]} to {hr.index[-1]}")
    print(f"  ✓ HR range: {hr.min():.0f}-{hr.max():.0f} bpm")
    print(f"  ✓ EDA range: {eda.min():.3f}-{eda.max():.3f} μS")
    
    return hr, eda


def create_csv_example(csv_path):
    """
    Create an example CSV file for demonstration.
    """
    print(f"Creating example CSV: {csv_path}")
    
    idx = pd.date_range('2025-01-01 06:00', periods=100, freq='5min')
    hr = 70 + np.random.randn(100) * 5
    eda = 0.1 + np.random.randn(100) * 0.02
    
    df = pd.DataFrame({
        'timestamp': idx,
        'hr': hr,
        'eda': eda
    })
    
    # Add some missing values
    df.loc[10:12, 'hr'] = np.nan
    df.loc[50:51, 'eda'] = np.nan
    
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Created {csv_path} with {len(df)} samples")


class RealTimeHRBuffer:
    """
    Buffer for real-time HR/EDA streaming (advanced use case).
    
    Example for integrating with real-time wearable devices.
    """
    def __init__(self, buffer_size=100):
        self.buffer_size = buffer_size
        self.hr_buffer = []
        self.eda_buffer = []
        self.time_buffer = []
    
    def add_sample(self, timestamp, hr, eda):
        """Add a new sample to the buffer."""
        self.time_buffer.append(timestamp)
        self.hr_buffer.append(hr)
        self.eda_buffer.append(eda)
        
        # Keep only recent samples
        if len(self.time_buffer) > self.buffer_size:
            self.time_buffer.pop(0)
            self.hr_buffer.pop(0)
            self.eda_buffer.pop(0)
    
    def get_series(self):
        """Get current buffered data as pandas Series."""
        if not self.time_buffer:
            return None, None
        
        hr = pd.Series(self.hr_buffer, index=pd.DatetimeIndex(self.time_buffer))
        eda = pd.Series(self.eda_buffer, index=pd.DatetimeIndex(self.time_buffer))
        
        return hr, eda
    
    def simulate_stream(self, start_time, n_samples=50):
        """Simulate real-time streaming for demonstration."""
        print(f"Simulating real-time stream ({n_samples} samples)...")
        
        for i in range(n_samples):
            timestamp = start_time + timedelta(minutes=i*5)
            hr = 70 + np.random.randn() * 5 + (10 if i > 20 else 0)  # Exercise at t>20
            eda = 0.1 + np.random.randn() * 0.02
            
            self.add_sample(timestamp, hr, eda)
            
            if i % 10 == 0:
                hr_series, eda_series = self.get_series()
                print(f"  {timestamp.strftime('%H:%M')} | Buffer size: {len(hr_series)} | "
                      f"HR: {hr:.0f} bpm")
        
        print(f"  ✓ Streaming complete")
        return self.get_series()


def main():
    print("=" * 70)
    print("Custom HR/EDA Source Examples")
    print("=" * 70)
    
    # Example 1: Synthetic data
    print("\n" + "=" * 70)
    print("Example 1: Synthetic Data Generation")
    print("=" * 70)
    hr_synth, eda_synth = generate_synthetic_context(
        start_time=datetime(2025, 1, 1, 6, 0),
        duration_hours=24
    )
    
    # Example 2: CSV loading
    print("\n" + "=" * 70)
    print("Example 2: Load from CSV")
    print("=" * 70)
    csv_path = '/tmp/example_hr_eda.csv'
    create_csv_example(csv_path)
    hr_csv, eda_csv = load_from_csv(csv_path)
    
    # Example 3: Real-time streaming
    print("\n" + "=" * 70)
    print("Example 3: Real-Time Streaming Buffer")
    print("=" * 70)
    buffer = RealTimeHRBuffer(buffer_size=50)
    hr_stream, eda_stream = buffer.simulate_stream(
        start_time=datetime(2025, 1, 1, 6, 0),
        n_samples=50
    )
    
    # Show how to use with ContextStream
    print("\n" + "=" * 70)
    print("Usage with ContextStream")
    print("=" * 70)
    
    from simglucose_ctx.context import ContextStream, ContextConfig
    
    print("\nOption 1: Synthetic data (for testing)")
    print("```python")
    print("hr, eda = generate_synthetic_context(start_time, duration_hours=24)")
    print("ctx = ContextStream(hr, eda, ContextConfig())")
    print("```")
    
    print("\nOption 2: CSV data (most common)")
    print("```python")
    print("hr, eda = load_from_csv('my_data.csv')")
    print("ctx = ContextStream(hr, eda, ContextConfig())")
    print("```")
    
    print("\nOption 3: Real-time streaming (advanced)")
    print("```python")
    print("buffer = RealTimeHRBuffer()")
    print("# ... continuously add samples ...")
    print("hr, eda = buffer.get_series()")
    print("ctx = ContextStream(hr, eda, ContextConfig())")
    print("```")
    
    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
