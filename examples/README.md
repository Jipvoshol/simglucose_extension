# Examples

This directory contains complete, runnable examples demonstrating various use cases of the SimGlucose Context Patch.

## Quick Start

Run any example directly:

```bash
cd examples
python basic_usage.py
python custom_hr_source.py
```

## Available Examples

### 1. `basic_usage.py`
**Minimal working example** 

- Creates synthetic HR/EDA data (rest → exercise → rest pattern)
- Sets up context stream with default configuration
- Runs 24-hour simulation with basal insulin only
- Displays glucose, m(t), and modulated parameters

### 2. `custom_hr_source.py`
**Creating context data from various sources**

Demonstrates three approaches:
1. **Synthetic generation** with realistic circadian rhythm, exercise, and stress patterns
2. **CSV loading** with missing data handling
3. **Real-time streaming** buffer (for wearable device integration)


## Notes

- All examples use `adolescent#001` patient by default
- Synthetic data is used for reproducibility
- For real-world use, replace with actual HR/EDA from wearables (see `custom_hr_source.py`)

## Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'simglucose_ctx'`

**Solution:** Install the patch first:
```bash
cd ..
pip install -e .
```

**Issue:** Examples run but glucose values seem off

**Check:**
- Are you using the correct patient baseline?
- Is the HR/EDA data timezone-aware vs. naive?
- Try running with `strict_validation=True` to catch config errors
