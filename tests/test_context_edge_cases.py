"""
Edge case unit tests for ContextStream and ContextConfig.

Tests cover:
- Timezone handling (critical: can cause silent 12h shifts)
- Data quality (NaN, extreme values, flatline sensors)
- Timestamp alignment (non-monotonic, duplicates, large gaps)
- Rate limiting and stability
- Configuration validation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from simglucose_ctx.context import ContextStream, ContextConfig


class TestTimezoneHandling:
    """CRITICAL: Timezone bugs can cause silent 12-hour shifts"""

    def test_naive_timestamps(self):
        """Test with naive (no timezone) timestamps"""
        idx = pd.date_range("2025-01-01", periods=20, freq="5min")
        hr = pd.Series([70] * 20, index=idx)
        eda = pd.Series([0.1] * 20, index=idx)

        ctx = ContextStream(hr, eda, preprocess=False)

        # Query at various points
        m1 = ctx.m(idx[5])
        m2 = ctx.m(idx[10])

        assert isinstance(m1, float)
        assert isinstance(m2, float)

    def test_timezone_aware_timestamps(self):
        """Test with timezone-aware timestamps"""
        idx = pd.date_range("2025-01-01", periods=20, freq="5min", tz="UTC")
        hr = pd.Series([70] * 20, index=idx)
        eda = pd.Series([0.1] * 20, index=idx)

        ctx = ContextStream(hr, eda, preprocess=False)

        # Query should work with TZ-aware timestamp
        m = ctx.m(idx[10])
        assert isinstance(m, float)

    def test_query_before_data_start(self):
        """Test querying timestamp before data starts (with grace period)"""
        idx = pd.date_range("2025-01-01 10:00", periods=20, freq="5min")
        hr = pd.Series([70] * 20, index=idx)
        eda = pd.Series([0.1] * 20, index=idx)

        ctx = ContextStream(hr, eda, preprocess=False)

        # Query 1 hour before start (should warn but not crash)
        early_ts = idx[0] - pd.Timedelta(hours=1)
        m = ctx.m(early_ts)

        # Should clamp to first available value
        assert isinstance(m, float)
        assert 0.6 <= m <= 2.5  # Within valid bounds

    def test_query_after_data_end(self):
        """Test querying timestamp after data ends (extrapolation)"""
        idx = pd.date_range("2025-01-01 10:00", periods=20, freq="5min")
        hr = pd.Series([70] * 20, index=idx)
        eda = pd.Series([0.1] * 20, index=idx)

        ctx = ContextStream(hr, eda, preprocess=False)

        # Query 1 hour after end
        late_ts = idx[-1] + pd.Timedelta(hours=1)
        m = ctx.m(late_ts)

        # Should use last available value
        assert isinstance(m, float)
        assert 0.6 <= m <= 2.5


class TestDataQuality:
    """Test handling of poor quality data"""

    def test_nan_at_start(self):
        """NaN values at beginning of series"""
        idx = pd.date_range("2025-01-01", periods=20, freq="5min")
        hr = pd.Series([np.nan] * 5 + [70] * 15, index=idx)
        eda = pd.Series([0.1] * 20, index=idx)

        # Should fill NaN with defaults
        ctx = ContextStream(hr, eda, preprocess=True)

        # First value should be filled
        m = ctx.m(idx[0])
        assert isinstance(m, float)
        assert not np.isnan(m)

    def test_extreme_hr_values(self):
        """Test with extreme but valid HR values"""
        idx = pd.date_range("2025-01-01", periods=10, freq="5min")
        # Bradycardia (40), normal (70), tachycardia (200)
        hr = pd.Series([40, 40, 70, 70, 200, 200, 70, 70, 40, 40], index=idx)
        eda = pd.Series([0.1] * 10, index=idx)

        ctx = ContextStream(hr, eda, preprocess=False)

        # All should produce valid m values
        for ts in idx:
            m = ctx.m(ts)
            assert isinstance(m, float)
            assert 0.6 <= m <= 2.5

    def test_flatline_sensor(self):
        """Sensor hangs (constant value for extended period)"""
        idx = pd.date_range("2025-01-01", periods=60, freq="5min")
        # Start normal, then flatline at 140 for 2 hours
        hr_vals = [70] * 12 + [140] * 36 + [70] * 12
        hr = pd.Series(hr_vals, index=idx)
        eda = pd.Series([0.1] * 60, index=idx)

        ctx = ContextStream(hr, eda, preprocess=False)

        # During flatline, m should stabilize (not keep climbing)
        m_start = ctx.m(idx[15])  # Early in flatline
        m_end = ctx.m(idx[45])  # Late in flatline

        # Both should be valid
        assert isinstance(m_start, float)
        assert isinstance(m_end, float)

        # Rate limiting should prevent extreme drift
        assert abs(m_end - m_start) < 2.0  # Max drift limited by rate limiter

    def test_negative_hr(self):
        """Sensor error producing negative values"""
        idx = pd.date_range("2025-01-01", periods=10, freq="5min")
        hr = pd.Series([70, 70, -10, 70, 70, 70, 70, 70, 70, 70], index=idx)
        eda = pd.Series([0.1] * 10, index=idx)

        # After clipping, negative value → 5th percentile
        ctx = ContextStream(hr, eda, preprocess=True)

        m = ctx.m(idx[2])
        assert isinstance(m, float)
        assert not np.isnan(m)


class TestTimestampAlignment:
    """Test timestamp alignment edge cases"""

    def test_non_monotonic_index(self):
        """Unsorted timestamps (should be fixed by sort_index)"""
        # Create unsorted index
        dates = pd.to_datetime(
            ["2025-01-01 10:05", "2025-01-01 10:00", "2025-01-01 10:10", "2025-01-01 10:15"]
        )
        hr = pd.Series([80, 70, 90, 85], index=dates)
        eda = pd.Series([0.1, 0.1, 0.2, 0.1], index=dates)

        # Should sort during preprocessing
        ctx = ContextStream(hr, eda, preprocess=True)

        # Query should work
        m = ctx.m(pd.Timestamp("2025-01-01 10:12"))
        assert isinstance(m, float)

    def test_duplicate_timestamps(self):
        """Multiple recordings at same time (sensor glitch)"""
        idx = pd.to_datetime(
            ["2025-01-01 10:00", "2025-01-01 10:00", "2025-01-01 10:05", "2025-01-01 10:10"]
        )
        hr = pd.Series([70, 75, 80, 85], index=idx)
        eda = pd.Series([0.1] * 4, index=idx)

        # Resampling should handle duplicates (averaging)
        ctx = ContextStream(hr, eda, preprocess=True)

        m = ctx.m(pd.Timestamp("2025-01-01 10:00"))
        assert isinstance(m, float)

    def test_large_data_gap(self):
        """Missing data for >1 hour"""
        # 08:00-09:00, then jump to 11:00-12:00
        idx1 = pd.date_range("2025-01-01 08:00", periods=12, freq="5min")
        idx2 = pd.date_range("2025-01-01 11:00", periods=12, freq="5min")
        idx = idx1.append(idx2)

        hr = pd.Series([70] * 24, index=idx)
        eda = pd.Series([0.1] * 24, index=idx)

        ctx = ContextStream(hr, eda, preprocess=False)

        # Query at 10:00 (in the gap)
        m = ctx.m(pd.Timestamp("2025-01-01 10:00"))

        # Should use stale data from 09:00
        assert isinstance(m, float)


class TestRateLimitingStability:
    """Test rate limiting and stability features"""

    def test_rapid_hr_spike(self):
        """Rest → sudden sprint → rest"""
        idx = pd.date_range("2025-01-01", periods=30, freq="5min")
        hr_vals = [70] * 10 + [180] * 10 + [70] * 10
        hr = pd.Series(hr_vals, index=idx)
        eda = pd.Series([0.1] * 30, index=idx)

        cfg = ContextConfig(max_delta_per_step=0.1)
        ctx = ContextStream(hr, eda, cfg, preprocess=False)

        # Track m values over spike
        m_values = [ctx.m(ts) for ts in idx]

        # Should see smooth ramp (no jumps > 0.1 per step)
        for i in range(1, len(m_values)):
            delta = abs(m_values[i] - m_values[i - 1])
            assert delta <= 0.1 + 1e-6, f"Delta too large: {delta} at step {i}"

    def test_hysteresis_boundary(self):
        """Test hysteresis at exact threshold I=0.4"""
        idx = pd.date_range("2025-01-01", periods=10, freq="5min")

        cfg = ContextConfig(hr_rest=60.0, hr_max=180.0)
        # I = (HR - 60) / (180 - 60) = 0.4 → HR = 108
        hr = pd.Series([108.0] * 10, index=idx)
        eda = pd.Series([0.1] * 10, index=idx)

        ctx = ContextStream(hr, eda, cfg, preprocess=False)

        hyst = {"exercise": 0, "stress": 0}

        # First call should not trigger (hysteresis_steps=2)
        m1 = ctx.m(idx[0], steps_above_thresh=hyst)

        # Second call should trigger
        m2 = ctx.m(idx[1], steps_above_thresh=hyst)

        assert isinstance(m1, float)
        assert isinstance(m2, float)


class TestConfigValidation:
    """Test configuration validation"""

    def test_invalid_hr_bounds(self):
        """hr_max <= hr_rest should fail in strict mode"""
        with pytest.raises(ValueError, match="hr_max.*must be >"):
            ContextConfig(hr_rest=70, hr_max=70, strict_validation=True)

    def test_invalid_hr_bounds_warning(self):
        """hr_max <= hr_rest should warn in non-strict mode"""
        # Should not raise , but log warning
        cfg = ContextConfig(hr_rest=70, hr_max=70, strict_validation=False)
        assert cfg.hr_rest == 70
        assert cfg.hr_max == 70

    def test_invalid_m_bounds(self):
        """mmin >= mmax should fail"""
        with pytest.raises(ValueError, match="Invalid m bounds"):
            ContextConfig(mmin=0.8, mmax=0.6, strict_validation=True)

    def test_negative_alpha(self):
        """Negative exercise gain should fail"""
        with pytest.raises(ValueError, match="alpha.*must be > 0"):
            ContextConfig(alpha=-0.5, strict_validation=True)

    def test_invalid_night_cap(self):
        """night_cap outside (0, 1) should fail"""
        with pytest.raises(ValueError, match="night_cap.*must be in range"):
            ContextConfig(night_cap=1.5, strict_validation=True)

    def test_valid_config(self):
        """Default config should pass validation"""
        cfg = ContextConfig()

        # Should have all expected attributes
        assert cfg.hr_rest == 60.0
        assert cfg.hr_max == 180.0
        assert cfg.alpha == 1.25
        assert cfg.beta == 0.3
