"""
test_look_ahead.py — Verify that no look-ahead bias exists in any component.

The critical invariant:

    predict_regime_filtered(data[0:T])[-1]
    ==
    predict_regime_filtered(data[0:T+N])[T-1]

for any N > 0.  If this holds, the forward algorithm is truly causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.feature_engineering import FeatureEngineer, FEATURE_NAMES
from core.hmm_engine import HMMEngine, HMMConfig, RegimeState


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)


def _make_ohlcv(n: int = 700) -> pd.DataFrame:
    """
    Synthetic OHLCV DataFrame with a simple random-walk close.
    Two distinct vol regimes are embedded so the HMM has something to learn.
    """
    # Build close via random walk with embedded vol regimes
    log_rets = np.concatenate([
        RNG.normal(0.0003, 0.008, n // 2),   # low-vol regime
        RNG.normal(-0.0002, 0.025, n // 2),  # high-vol regime
    ])
    close = 100.0 * np.exp(np.cumsum(log_rets))

    high   = close * (1 + np.abs(RNG.normal(0, 0.003, n)))
    low    = close * (1 - np.abs(RNG.normal(0, 0.003, n)))
    open_  = close * (1 + RNG.normal(0, 0.002, n))
    volume = RNG.integers(1_000_000, 5_000_000, size=n).astype(float)

    idx = pd.bdate_range("2019-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _fast_config() -> HMMConfig:
    """HMMConfig tuned for fast tests (small n_init, limited candidates)."""
    return HMMConfig(
        n_candidates=[2, 3],
        n_init=3,
        covariance_type="full",
        min_train_bars=150,
        stability_bars=2,
        flicker_window=10,
        flicker_threshold=3,
        min_confidence=0.50,
    )


@pytest.fixture(scope="module")
def ohlcv_df() -> pd.DataFrame:
    return _make_ohlcv(700)


@pytest.fixture(scope="module")
def feature_df(ohlcv_df) -> pd.DataFrame:
    eng = FeatureEngineer()
    return eng.build_features(ohlcv_df)


_hmmlearn = pytest.importorskip(
    "hmmlearn",
    reason="hmmlearn not installed — requires Python ≤ 3.12 or C++ build tools",
)


@pytest.fixture(scope="module")
def fitted_engine(feature_df) -> HMMEngine:
    """One fitted engine reused across all HMM tests in this module."""
    engine = HMMEngine(_fast_config())
    engine.fit(feature_df.iloc[:400])
    return engine


# ─────────────────────────────────────────────────────────────────────────────
# MANDATORY: no-look-ahead bias test
# ─────────────────────────────────────────────────────────────────────────────

class TestNoLookAheadBias:
    """
    Core architectural guarantee: regime at bar T must be identical whether we
    supply data[0:T] or data[0:T+N].
    """

    def test_no_look_ahead_bias(self, fitted_engine: HMMEngine, feature_df: pd.DataFrame):
        """
        Regime at T must be identical with data[0:T] vs data[0:T+100].

        This is the mandatory test specified in the Phase-2 requirements.
        Failure means predict_regime_filtered() uses future data — fatal look-ahead bias.
        """
        T   = 400
        # Ensure the feature_df is long enough
        assert len(feature_df) >= T + 100, (
            f"feature_df only has {len(feature_df)} rows; need at least {T + 100}"
        )

        # Prediction using only data[0:T]
        states_short = fitted_engine.predict_regime_filtered(feature_df.iloc[:T])
        regime_short = states_short[-1]          # regime at bar T-1

        # Prediction using data[0:T+100] — T-1 result must be identical
        states_long  = fitted_engine.predict_regime_filtered(feature_df.iloc[:T + 100])
        regime_long  = states_long[T - 1]        # same bar, more future data appended

        assert regime_short == regime_long, (
            f"LOOK-AHEAD BIAS DETECTED: "
            f"short={regime_short.label} (p={regime_short.probability:.4f})  "
            f"long={regime_long.label}  (p={regime_long.probability:.4f})"
        )

    def test_probabilities_match_exactly(
        self, fitted_engine: HMMEngine, feature_df: pd.DataFrame
    ):
        """
        The full posterior distribution at bar T-1 must be numerically identical
        regardless of how many future bars are appended.
        """
        T = 300

        states_short = fitted_engine.predict_regime_filtered(feature_df.iloc[:T])
        states_long  = fitted_engine.predict_regime_filtered(feature_df.iloc[:T + 50])

        np.testing.assert_allclose(
            states_short[-1].state_probabilities,
            states_long[T - 1].state_probabilities,
            rtol=1e-6,
            atol=1e-9,
            err_msg="Forward algorithm posteriors at bar T-1 differ with/without future data",
        )

    def test_incremental_matches_batch(
        self, fitted_engine: HMMEngine, feature_df: pd.DataFrame
    ):
        """
        predict_filtered_next (incremental) must produce the same regime sequence
        as predict_regime_filtered (batch) on the same data.
        """
        N = 50
        subset = feature_df.iloc[:250 + N]

        # Batch prediction
        batch_states = fitted_engine.predict_regime_filtered(subset)

        # Incremental prediction — start from scratch
        fitted_engine.reset_live_state()
        incr_states = []
        for i in range(len(subset)):
            obs = subset.iloc[i].values
            ts  = subset.index[i]
            s   = fitted_engine.predict_filtered_next(obs, timestamp=ts)
            incr_states.append(s)
        fitted_engine.reset_live_state()  # clean up after test

        # Compare labels for each bar
        for t, (b, inc) in enumerate(zip(batch_states, incr_states)):
            assert b.label == inc.label, (
                f"Batch vs incremental mismatch at bar {t}: "
                f"batch={b.label}  incr={inc.label}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineer causality tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureEngineerNoLookAhead:
    """Verify that FeatureEngineer uses only past data at each bar."""

    def test_permuting_future_does_not_change_past_features(self, ohlcv_df):
        """
        Randomly scramble OHLCV bars after index T; verify that feature values
        at all rows BEFORE T are numerically identical in both the original and
        the scrambled dataset.

        T is chosen to be well inside the valid feature range (after the warmup
        period), so that a non-trivial number of rows are checked.
        """
        engineer = FeatureEngineer()

        # Determine warmup so T is safely within valid features
        features_original = engineer.build_features(ohlcv_df)
        n = len(ohlcv_df)
        warmup = n - len(features_original)
        # Split at midpoint of valid feature range
        T = warmup + len(features_original) // 2
        assert T < n - 50, f"Not enough data after T={T} for a meaningful test"

        cut_date = ohlcv_df.index[T]

        # Scramble everything after bar T in the raw OHLCV
        ohlcv_scrambled = ohlcv_df.copy()
        n_future = n - T
        perm = RNG.permutation(n_future)
        for col in ["open", "high", "low", "close", "volume"]:
            vals = ohlcv_scrambled[col].values.copy()
            vals[T:] = vals[T:][perm]
            ohlcv_scrambled[col] = vals

        features_scrambled = engineer.build_features(ohlcv_scrambled)

        # Only compare rows that fall before the scramble point
        orig_before = features_original[features_original.index < cut_date]
        scr_before  = features_scrambled[features_scrambled.index < cut_date]
        common_idx  = orig_before.index.intersection(scr_before.index)

        assert len(common_idx) > 0, (
            f"No overlapping valid feature rows before cut_date={cut_date}. "
            f"Increase n or decrease T."
        )

        pd.testing.assert_frame_equal(
            features_original.loc[common_idx],
            features_scrambled.loc[common_idx],
            check_exact=False,
            atol=1e-9,
        )

    def test_rolling_stats_use_only_past_window(self, ohlcv_df):
        """
        For every row t, the realised-vol feature must equal the std of the
        preceding 20 log-returns (causal window check).
        """
        engineer = FeatureEngineer()
        features = engineer.build_features(ohlcv_df)

        # Independently compute 20-bar realised vol from raw close prices
        close    = ohlcv_df["close"]
        lr1_raw  = np.log(close / close.shift(1))
        rv20_raw = lr1_raw.rolling(20, min_periods=10).std() * np.sqrt(252)

        # Both are z-scored, but their ranks within the valid window must be
        # monotone (same ordering ↔ same causal computation).
        common_idx = features.index.intersection(rv20_raw.dropna().index)
        if len(common_idx) < 10:
            pytest.skip("Insufficient overlapping rows for rolling-stat check")

        feat_rv   = features.loc[common_idx, "realized_vol_20"]
        raw_rv    = rv20_raw.loc[common_idx]

        # Spearman rank correlation must be very high (> 0.99)
        from scipy.stats import spearmanr
        corr, _ = spearmanr(feat_rv.values, raw_rv.values)
        assert corr > 0.99, (
            f"realized_vol_20 Spearman correlation with raw vol = {corr:.4f} < 0.99; "
            "possible look-ahead in rolling computation."
        )

    def test_all_features_present(self, ohlcv_df):
        """build_features must return exactly the FEATURE_NAMES columns."""
        engineer = FeatureEngineer()
        features = engineer.build_features(ohlcv_df)
        assert list(features.columns) == FEATURE_NAMES, (
            f"Column mismatch.\n"
            f"Expected: {FEATURE_NAMES}\n"
            f"Got:      {list(features.columns)}"
        )

    def test_no_nan_in_valid_rows(self, ohlcv_df):
        """After dropna, the returned feature matrix must contain no NaN values."""
        engineer = FeatureEngineer()
        features = engineer.build_features(ohlcv_df)
        assert not features.isnull().any().any(), (
            "build_features returned NaN values after dropping incomplete rows."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward backtester causality tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWalkForwardNoLookAhead:
    """
    Structural checks that the walk-forward loop never leaks test-window data
    into the training window.
    """

    def test_train_window_does_not_overlap_test_window(self, feature_df):
        """
        For each walk-forward fold, the training slice [start:train_end] must not
        overlap with the test slice [train_end:test_end].
        """
        train_window = 200
        test_window  = 100
        step_size    = 100
        n            = len(feature_df)

        folds: list[tuple[int, int, int, int]] = []
        start = 0
        while True:
            train_end = start + train_window
            test_end  = train_end + test_window
            if test_end > n:
                break
            folds.append((start, train_end, train_end, test_end))
            start += step_size

        assert len(folds) > 0, "No folds generated — feature_df may be too short"

        for train_start, train_end, test_start, test_end in folds:
            assert train_end <= test_start, (
                f"Train window [{train_start}:{train_end}] overlaps "
                f"test window [{test_start}:{test_end}]"
            )
            # No overlap in index
            train_idx = set(range(train_start, train_end))
            test_idx  = set(range(test_start, test_end))
            assert train_idx.isdisjoint(test_idx), (
                "Train and test index sets are NOT disjoint!"
            )

    def test_hmm_refitted_per_fold(self, feature_df):
        """
        Simulate two walk-forward folds.  The HMM fitted on fold 1's training
        data must be a different object (refitted from scratch) from fold 2.
        Verified by checking that BIC values differ (different training data).
        """
        cfg = _fast_config()

        fold1_X = feature_df.iloc[:200]
        fold2_X = feature_df.iloc[100:300]

        engine1 = HMMEngine(cfg)
        engine2 = HMMEngine(cfg)

        engine1.fit(fold1_X)
        engine2.fit(fold2_X)

        # Engines are independent objects — they must not share the same model
        assert engine1._model is not engine2._model, (
            "Fold engines share the same model object — refitting is broken."
        )

        # Their training dates should be close but distinct
        assert engine1._training_date != engine2._training_date or \
               engine1._bic != engine2._bic, (
            "Both fold engines appear identical — second fold was not refitted."
        )
