"""
test_hmm.py — Unit tests for the HMM regime detection engine.

Tests cover:
  - Model fitting with synthetic data.
  - BIC model selection (best n_states is recoverable from known data).
  - Regime labelling consistency (LOW_VOL / MID_VOL / HIGH_VOL assignment).
  - Flicker suppression: rapid state switches are clamped.
  - Stability detection: regime is not confirmed until stability_bars hold.
  - Confidence gating: low-confidence predictions yield Regime.UNKNOWN.
"""

import numpy as np
import pandas as pd
import pytest

from core.hmm_engine import HMMEngine, HMMConfig, Regime, RegimeState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config() -> HMMConfig:
    """Return a fast-running HMMConfig for testing (small n_init)."""
    return HMMConfig(
        n_candidates=[2, 3],
        n_init=3,
        covariance_type="full",
        min_train_bars=50,
        stability_bars=2,
        flicker_window=10,
        flicker_threshold=3,
        min_confidence=0.55,
    )


@pytest.fixture
def synthetic_features(default_config: HMMConfig) -> pd.DataFrame:
    """
    Generate synthetic feature data with two clearly distinct regimes:
    low-vol (small returns) and high-vol (large returns).
    """
    rng = np.random.default_rng(42)
    n = 200
    low_vol = rng.normal(0, 0.005, (n // 2, 2))
    high_vol = rng.normal(0, 0.03, (n // 2, 2))
    data = np.vstack([low_vol, high_vol])
    return pd.DataFrame(data, columns=["log_return", "realised_vol"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHMMEngineFit:
    def test_fit_returns_self(self, default_config, synthetic_features):
        """fit() should return the engine instance for method chaining."""
        ...

    def test_fit_marks_engine_as_fitted(self, default_config, synthetic_features):
        """After fit(), is_fitted() should return True."""
        ...

    def test_fit_raises_on_insufficient_data(self, default_config):
        """fit() should raise ValueError when fewer than min_train_bars rows are provided."""
        ...

    def test_bic_selects_reasonable_n_states(self, default_config, synthetic_features):
        """BIC selection should choose n_states ≥ 2 on bimodal synthetic data."""
        ...


class TestHMMEnginePredict:
    def test_predict_length_matches_input(self, default_config, synthetic_features):
        """predict() should return one RegimeState per input row."""
        ...

    def test_predict_current_is_last_state(self, default_config, synthetic_features):
        """predict_current() result should match the last element of predict()."""
        ...

    def test_regime_labels_are_valid(self, default_config, synthetic_features):
        """All predicted regimes should be members of the Regime enum."""
        ...

    def test_confidence_in_unit_interval(self, default_config, synthetic_features):
        """All confidence values should be in [0.0, 1.0]."""
        ...


class TestFlickerSuppression:
    def test_flicker_reduces_transitions(self, default_config):
        """_apply_flicker_filter should reduce the number of state transitions."""
        ...

    def test_flicker_does_not_alter_stable_runs(self, default_config):
        """A stable run of identical states should pass through unchanged."""
        ...


class TestStabilityDetection:
    def test_stability_requires_consecutive_bars(self, default_config):
        """is_stable should only be True after stability_bars consecutive same-state bars."""
        ...
