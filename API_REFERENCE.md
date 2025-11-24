# API Reference

Complete reference documentation for **SimGlucose Context Patch** v1.0.

---

## ContextConfig

Configuration dataclass containing all hyperparameters for context modulation.

### Basic Parameters

#### `dt_minutes: int = 5`
**Time resolution for context processing.**

- Resampling frequency for HR/EDA data
- Should match simulation step size (typically 5 minutes)
- **Range:** 1-15 minutes (5 is standard for SimGlucose)

#### `hr_rest: float = 60.0`
**Resting heart rate in beats per minute (bpm).**

- Used as baseline for intensity normalization
- **Typical range:** 50-80 bpm (lower for athletes, higher for sedentary)
- **Physiological basis:** Determines zero-point for exercise detection
- **Example:** Athletic adolescent: `hr_rest=55`, sedentary adult: `hr_rest=75`

#### `hr_max: float = 180.0`
**Maximum expected heart rate in bpm.**

- Upper bound for intensity normalization
- **Rule of thumb:** 220 - age
- **Must be:** `hr_max > hr_rest` (validated in `__post_init__`)
- **Example:** 40-year-old: `hr_max=180`, 60-year-old: `hr_max=160`

### Modulation Gains

#### `alpha: float = 1.25`
*Exercise gain — determines the sensitivity increase at max intensity.*

*   **Formula:** $m_{exercise} = 1.0 + \alpha \cdot I$, where $I \in [0, 1]$ is normalized HR intensity.
*   **Calibration Logic:** Chosen to reach $m \approx 2.25$ at max intensity ($I=1$). This target $m$ is required to achieve the literature-based $\sim 17\times$ increase in glucose uptake ($V_{m0}$) via the power law $m^{3.5}$.
*   **Physiological Basis:** Exercise increases insulin sensitivity 1.5-2.5× (Riddell et al. 2017).

#### `beta: float = 0.3`
*Stress gain — determines the sensitivity decrease at max stress.*

*   **Formula:** $m_{stress} = 1.0 - \beta \cdot S$, where $S \in [0, 1]$ is normalized EDA.
*   **Calibration Logic:** Chosen to reach $m \approx 0.7$ at max stress ($S=1$), matching the 30-40% reduction in sensitivity observed in clinical studies.
*   **Physiological Basis:** Stress decreases sensitivity to 0.6-0.8× baseline (cortisol effect).

### Safety Bounds

#### `mmin: float = 0.6`
**Minimum allowed m(t) value.**

- Prevents excessive insulin resistance
- **Physiological basis:** Even severe stress rarely reduces sensitivity below 60% baseline
- **Range:** 0.5-0.8
- **Must satisfy:** `0 < mmin < mmax`

#### `mmax: float = 2.5`
**Maximum allowed m(t) value.**

- Prevents unrealistic sensitivity boosts
- **Physiological basis:** Exercise typically increases sensitivity 1.5-2.5×
- **Range:** 1.5-3.0
- **Must satisfy:** `mmin < mmax`

#### `max_delta_per_step: float = 0.1`
**Maximum change in m(t) per time step (rate limiter).**

- Prevents sudden jumps in sensitivity
- **Example:** With 5-min steps, can change by 0.1 per step = 1.2 per hour maximum
- **Range:** 0.05-0.2
- **Must be:** > 0
- **Effect:** Lower values → smoother transitions, slower response to physiological changes

### Smoothing & Filtering

#### `ema_half_life_min: float = 15.0`
**Half-life for exponential moving average (EMA) smoothing.**

- Applied to HR/EDA before computing m(t)
- **Formula:** `alpha_ema = 1 - exp(-ln(2) / (half_life / dt))`
- **Range:** 10-30 minutes
- **Effect:**
  - 10 min: Responsive but noisy
  - 15 min: Standard (recommended)
  - 30 min: Very smooth but slow

#### `hysteresis_steps: int = 2`
**Number of consecutive steps above threshold to trigger exercise/stress state.**

- Prevents false alarms from HR/EDA spikes
- **Example:** With `hysteresis_steps=2`, HR must stay elevated for 2 steps (10 min) to count as exercise
- **Range:** 1-5
- **Effect:** Higher values → fewer false positives, slower response

### Advanced Features

#### `night_cap: Optional[float] = None`
**Optional cap on |m(t) - 1.0| during nighttime hours.**

- Limits modulation magnitude at night (conservative safety feature)
- **Example:** `night_cap=0.15` → m(t) ∈ [0.85, 1.15] at night
- **Range:** 0.1-0.3 or None (disabled)
- **Must be:** `0 < night_cap < 1` if set
- **Use case:** Prevent nocturnal hypoglycemia from nighttime exercise (e.g., sleepwalking)

#### `night_hours: tuple = (23, 6)`
**Time range (hour_start, hour_end) for night cap application.**

- 24-hour format, local time
- **Default:** 23:00-06:00 (11 PM to 6 AM)
- **Example:** Evening shift worker: `night_hours=(8, 14)` (8 AM - 2 PM)

#### `stress_hr_supp_pow: float = 2.0`
**Power for HR-based stress suppression.**

- **Formula:** `effective_beta = beta × (1 - I)^power`
- Suppresses stress effects when HR is high (likely exercise, not pure stress)
- **Range:** 1.0-3.0
- **Effect:** Higher → more aggressive stress suppression during elevated HR

#### `stress_hr_off_threshold: Optional[float] = 0.65`
**Hard cutoff: ignore stress if HR intensity >= threshold.**

- **Example:** `stress_hr_off_threshold=0.65` → ignore stress when HR >65% of max
- **Range:** 0.5-0.8 or None (disabled)
- **Use case:** Disambiguate exercise (HR+EDA high) vs. exam stress (EDA high, HR normal)

### Hybrid Strategy Parameters

#### `vm0_exponent_exercise: float = 3.5`
**Exponential boost for insulin-independent uptake (Vm0) during exercise.**

- **Formula:** `vm0_factor = m^K` when `m > 1` (exercise)
- **Calibration:** K=3.5 targets 10-20× uptake at theoretical max intensity
  - With alpha=1.25, theoretical m_max = 2.25
  - 2.25^3.5 ≈ 17× boost (matches AMPK physiology literature)
- **Physiological basis:** Exercise increases muscle glucose uptake 10-20× via AMPK pathway (Sylow et al. 2017)
- **Range:**
  - Conservative: 2.5-3.0 (mild boost)
  - Standard: 3.0-4.0 (literature-based)
  - Aggressive: 4.0-4.5 (strong athletes only)
- **Critical:** Must be > 0
-  **Warning:** If you change `alpha` or `mmax`, recalibrate this exponent!
- **Recalibration formula:** `K = log(target_boost) / log(m_theoretical)`

#### `vmx_exponent_exercise: float = 0.2`
**Minimal scaling for insulin-dependent uptake (Vmx) during exercise.**

- **Formula:** `vmx_factor = m^0.2` when `m > 1` (exercise)
- **Rationale:** Prevents negative uptake artifact when insulin < basal (pump suspend)
- **Effect:** At m=2.0 (high exercise), vmx_factor ≈ 1.15 (only +15%)
- **Range:** 0.0-0.5 (keep conservative)
- **Must be:** >= 0

### Validation

#### `strict_validation: bool = False`
**Enable strict parameter validation (raises ValueError on invalid config).**

- **Default (False):** Log warnings, allow invalid configs (v1.0 behavior)
- **Strict (True):** Raise ValueError on invalid parameters
- **Use case:** Set to True during development/testing to catch misconfigurations early
- **Note:** Will become default True in v2.0

---

## ContextStream

Processes HR/EDA time series into smooth m(t) multiplier.

### Constructor

```python
ContextStream(
    hr: pd.Series,
    eda: pd.Series,
    cfg: ContextConfig = ContextConfig(),
    preprocess: bool = True,
    eda_min_max: Optional[tuple] = None
)
```

**Parameters:**
- **hr** (`pd.Series`): Heart rate in bpm, indexed by datetime
  - Index must be monotonic increasing
  - Can be timezone-aware or naive (but must match `eda`)
- **eda** (`pd.Series`): Electrodermal activity (skin conductance), same index as `hr`
  - Units: typically microsiemens (μS), but normalized internally
- **cfg** (`ContextConfig`): Configuration hyperparameters
- **preprocess** (`bool`): If True, resample, clip outliers, smooth with EMA
- **eda_min_max** (`Optional[tuple]`): Pre-computed (min, max) for EDA normalization
  - If None, computed from data (per-subject normalization)
  - Use same values for train/test consistency



### Methods

#### `m(now_ts: pd.Timestamp, steps_above_thresh: Optional[dict] = None) -> float`
**Compute m(t) multiplier for given timestamp.**

**Parameters:**
- **now_ts**: Query timestamp (datetime)
- **steps_above_thresh**: Optional dict for hysteresis tracking `{"exercise": int, "stress": int}`
  - Maintained across calls to track sustained elevation
  - If None, no hysteresis (immediate response)

**Returns:**
- **float**: Modulation factor m(t) ∈ [mmin, mmax]

**Behavior:**
- Uses forward-fill (last observation carried forward) for timestamps between data points
- Warns if timestamp >30min outside data range
- Handles NaN by filling with defaults (hr_rest, 0.0)
- Applies EMA smoothing, rate limiting, night cap (if enabled)

**Example:**
```python
ctx = ContextStream(hr, eda, ContextConfig())
hyst = {"exercise": 0, "stress": 0}
for ts in simulation_timestamps:
    m_value = ctx.m(ts, steps_above_thresh=hyst)
    # m_value used to modulate patient parameters
```

---

## ContextAwareT1DSimEnv

Wrapper for `simglucose.simulation.env.T1DSimEnv` with context modulation.

### Constructor

```python
ContextAwareT1DSimEnv(
    patient, sensor, pump, scenario,
    context_stream: Optional[ContextStream] = None,
    modulate_vmx: bool = True,
    modulate_p2u: bool = False,
    modulate_vm0: bool = False,
    max_log_size: int = 300
)
```

**Parameters:**
- **patient, sensor, pump, scenario**: Standard SimGlucose components
- **context_stream** (`Optional[ContextStream]`): Context stream providing m(t)
  - If None, behaves like original T1DSimEnv (no modulation)
- **modulate_vmx** (`bool`): Enable Vmx modulation (insulin-dependent uptake)
- **modulate_p2u** (`bool`): Enable p2u modulation (insulin action kinetics)
- **modulate_vm0** (`bool`): Enable Vm0 modulation (insulin-independent uptake)
  -  Recommended: Enable for exercise simulation
- **max_log_size** (`int`): Maximum log buffer entries
  - Default: 300 (~25 hours at 5min intervals)
  - For 7-day simulations: use 500-1000 or disable logging

### Methods

#### `step(action) -> (obs, reward, done, info)`
**Standard Gym API step function with added context info.**

**Returns:**
- **info dict** includes:
  - `context_m`: Current m(t) value
  - `context_hr`: Current HR (if available)
  - `context_eda`: Current EDA (if available)

#### `reset() -> (obs, reward, done, info)`
**Reset environment and clear context log.**

- Restores baseline parameters (Vmx, Vm0, p2u)
- Clears logging buffer
- Resets hysteresis state

#### `get_context_log() -> Optional[pd.DataFrame]`
**Retrieve logged context modulation history.**

**Returns:**
- **DataFrame** with columns: `m`, `Vmx`, `p2u`, `Vm0`, indexed by timestamp
- **None** if no context stream or empty log
- **Note:** Limited to last `max_log_size` entries (most recent)

---

## Common Usage Patterns

### Pattern 1: Conservative Config (First-Time Users)

```python
cfg = ContextConfig(
    alpha=0.8,                  # Mild exercise response
    beta=0.2,                   # Mild stress response
    vm0_exponent_exercise=2.5,  # Conservative boost
    mmax=1.8,                   # Cap at 80% increase
    mmin=0.8,                   # Cap at 20% decrease
    night_cap=0.15              # Very conservative at night
)
```

**Use case:** Initial testing, risk-averse scenarios, children

### Pattern 2: Aggressive Config (Athletes)

```python
cfg = ContextConfig(
    alpha=1.5,                  # Strong exercise response
    beta=0.3,                   # Moderate stress response
    vm0_exponent_exercise=4.0,  # Higher boost for trained muscles
    mmax=2.5,                   # Allow up to 2.5× sensitivity
    stress_hr_off_threshold=0.7 # Ignore stress at very high HR
)
```

**Use case:** Athletic patients, high-intensity training simulations

### Pattern 3: Stress-Only (Disable Exercise)

```python
cfg = ContextConfig(
    alpha=0.0,                  # No exercise response
    beta=0.4,                   # Strong stress response
    modulate_vm0=False          # Skip Vm0 (exercise mechanism)
)
```

**Use case:** Studying psychological stress effects only

### Pattern 4: Exercise-Only (Disable Stress)

```python
cfg = ContextConfig(
    alpha=1.25,                 # Standard exercise response
    beta=0.0,                   # No stress response
    modulate_vmx=False          # Skip Vmx (stress mechanism)
)

env = ContextAwareT1DSimEnv(
    ...,
    modulate_vm0=True,
    modulate_vmx=False  # Disable Vmx modulation
)
```

**Use case:** Studying exercise effects only



## See Also

- [README.md](README.md) - Quick start and overview
- [examples/](examples/) - Complete runnable examples
