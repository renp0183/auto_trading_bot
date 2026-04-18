"""
test_strategies.py — Unit tests for regime allocation strategies.

Tests cover:
  - Correct allocation fractions for each regime × trend combination.
  - Leverage capping at max_leverage.
  - Uncertainty scaling when HMM confidence is below min_confidence.
  - Rebalance threshold logic (trigger vs. no-trigger).
"""

import pandas as pd
import pytest

from core.hmm_engine import Regime, RegimeState
from core.regime_strategies import RegimeStrategy, StrategyConfig, AllocationTarget

import numpy as np


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_strategy() -> RegimeStrategy:
    return RegimeStrategy(StrategyConfig())


def make_regime_state(
    regime: Regime,
    confidence: float = 0.80,
    is_stable: bool = True,
) -> RegimeState:
    """Helper to construct a minimal RegimeState for testing."""
    return RegimeState(
        regime=regime,
        confidence=confidence,
        state_index=0,
        n_states=3,
        transition_matrix=np.eye(3),
        is_stable=is_stable,
        is_flickering=False,
    )


def make_trending_prices(n: int = 100) -> pd.Series:
    """Rising price series — close will be above 50-bar SMA."""
    return pd.Series(range(n, 2 * n), dtype=float)


def make_flat_prices(n: int = 100) -> pd.Series:
    """Flat price series — close will be at / below 50-bar SMA."""
    return pd.Series([100.0] * n)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAllocationFractions:
    def test_low_vol_allocation(self, default_strategy):
        """LOW_VOL regime should return 0.95 allocation."""
        ...

    def test_mid_vol_with_trend(self, default_strategy):
        """MID_VOL + trend confirmed → 0.95 allocation."""
        ...

    def test_mid_vol_no_trend(self, default_strategy):
        """MID_VOL + no trend → 0.60 allocation."""
        ...

    def test_high_vol_allocation(self, default_strategy):
        """HIGH_VOL regime should return 0.60 allocation."""
        ...


class TestLeverage:
    def test_low_vol_leverage(self, default_strategy):
        """LOW_VOL regime should apply 1.25× leverage."""
        ...

    def test_high_vol_no_leverage(self, default_strategy):
        """HIGH_VOL regime should use 1.0× leverage (no margin)."""
        ...


class TestUncertaintyScaling:
    def test_low_confidence_reduces_allocation(self, default_strategy):
        """Confidence below min_confidence should scale allocation by uncertainty_size_mult."""
        ...

    def test_high_confidence_unaffected(self, default_strategy):
        """Confidence at or above min_confidence should not apply uncertainty scaling."""
        ...


class TestRebalanceLogic:
    def test_no_rebalance_within_threshold(self, default_strategy):
        """needs_rebalance returns False when drift < rebalance_threshold."""
        ...

    def test_rebalance_triggered_beyond_threshold(self, default_strategy):
        """needs_rebalance returns True when any symbol exceeds drift threshold."""
        ...
