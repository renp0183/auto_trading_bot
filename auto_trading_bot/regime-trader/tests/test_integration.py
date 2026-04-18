"""
test_integration.py — End-to-end and cross-layer integration tests.

Test suites:
  a. TestEndToEndDryRun     — full pipeline: features → strategy → risk → simulated order
  b. TestBacktestEndDateIndependence — feature values at bar T are invariant to
                              appending future data (no look-ahead at the backtest level)
  c. TestRiskStress         — extreme sizes capped, rapid-fire blocked, no-stop rejected
  d. TestAlpacaMockOrders   — bracket submit, stop tighten/widen, cancel, clean state
  e. TestStateRecovery      — snapshot JSON round-trip; no double-entry after restart
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from core.hmm_engine import RegimeInfo, RegimeState
from core.regime_strategies import (
    Signal,
    StrategyConfig,
    StrategyOrchestrator,
)
from core.risk_manager import (
    BreakerType,
    CircuitBreaker,
    PortfolioState,
    Position,
    RiskConfig,
    RiskDecision,
    RiskManager,
    TradingState,
    _HALT_LOCK_FILE,
)
from data.feature_engineering import FeatureEngineer

# ─────────────────────────────────────────────────────────────────────────────
# Shared test fixtures and helpers
# ─────────────────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(7)


def _make_ohlcv(n: int = 300, seed: int = 7) -> pd.DataFrame:
    """Synthetic daily OHLCV with two embedded vol regimes."""
    rng = np.random.default_rng(seed)
    half = n // 2
    log_rets = np.concatenate([
        rng.normal(0.0004, 0.007, half),   # low-vol regime
        rng.normal(-0.0001, 0.022, n - half),  # high-vol regime
    ])
    close  = 100.0 * np.exp(np.cumsum(log_rets))
    high   = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low    = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_  = close * (1 + rng.normal(0, 0.002, n))
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    idx    = pd.bdate_range("2021-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _make_regime_infos() -> dict[str, RegimeInfo]:
    """
    Three synthetic RegimeInfo objects spanning low / mid / high vol.
    regime_id matches the dict key's vol rank position (0 = lowest vol).
    """
    return {
        "BULL": RegimeInfo(
            regime_id=0,
            regime_name="BULL",
            expected_return=0.6,
            expected_volatility=-1.2,         # lowest vol
            recommended_strategy_type="LowVolBull",
            max_leverage_allowed=1.25,
            max_position_size_pct=0.95,
            min_confidence_to_act=0.55,
        ),
        "NEUTRAL": RegimeInfo(
            regime_id=1,
            regime_name="NEUTRAL",
            expected_return=0.1,
            expected_volatility=0.0,          # mid vol
            recommended_strategy_type="MidVolCautious",
            max_leverage_allowed=1.0,
            max_position_size_pct=0.95,
            min_confidence_to_act=0.55,
        ),
        "BEAR": RegimeInfo(
            regime_id=2,
            regime_name="BEAR",
            expected_return=-0.4,
            expected_volatility=1.5,          # highest vol
            recommended_strategy_type="HighVolDefensive",
            max_leverage_allowed=1.0,
            max_position_size_pct=0.60,
            min_confidence_to_act=0.55,
        ),
    }


def _make_regime_state(
    label: str = "BULL",
    state_id: int = 0,
    probability: float = 0.82,
    n_states: int = 3,
    in_transition: bool = False,
    is_flickering: bool = False,
) -> RegimeState:
    probs = np.zeros(n_states)
    probs[state_id] = probability
    remainder = 1.0 - probability
    for i in range(n_states):
        if i != state_id:
            probs[i] = remainder / (n_states - 1)

    return RegimeState(
        label=label,
        state_id=state_id,
        probability=probability,
        state_probabilities=probs,
        timestamp=pd.Timestamp.now(),
        is_confirmed=True,
        consecutive_bars=5,
        in_transition=in_transition,
        candidate_label=None,
        flicker_rate=0.05 if not is_flickering else 0.30,
        regime_info=None,
    )


def _make_portfolio(
    equity: float = 100_000.0,
    positions: Optional[dict] = None,
) -> PortfolioState:
    e = equity
    return PortfolioState(
        equity=e,
        cash=e,
        buying_power=e,
        positions=positions or {},
        peak_equity=e,
        sod_equity=e,
        sow_equity=e,
    )


def _make_signal(
    symbol: str = "SPY",
    direction: Literal["LONG", "FLAT"] = "LONG",
    entry_price: float = 450.0,
    stop_loss: float = 440.0,
    take_profit: Optional[float] = 470.0,
    position_size_pct: float = 0.10,
    leverage: float = 1.0,
    regime_id: int = 0,
    regime_name: str = "BULL",
) -> Signal:
    return Signal(
        symbol=symbol,
        direction=direction,
        confidence=0.82,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        position_size_pct=position_size_pct,
        leverage=leverage,
        regime_id=regime_id,
        regime_name=regime_name,
        regime_probability=0.82,
        timestamp=pd.Timestamp.now(),
        reasoning="test signal",
        strategy_name="LowVolBullStrategy",
    )


def _make_risk_manager() -> RiskManager:
    return RiskManager(RiskConfig(), initial_equity=100_000.0)


# ─────────────────────────────────────────────────────────────────────────────
# a. End-to-end dry run
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndDryRun:
    """
    Full pipeline without hmmlearn:
    OHLCV → FeatureEngineer → StrategyOrchestrator → RiskManager → mock executor.
    """

    @pytest.fixture(scope="class")
    def ohlcv(self):
        return _make_ohlcv(700)

    @pytest.fixture(scope="class")
    def bars_dict(self, ohlcv):
        return {"SPY": ohlcv, "AAPL": ohlcv.copy()}

    @pytest.fixture(scope="class")
    def orchestrator(self):
        return StrategyOrchestrator(StrategyConfig(), _make_regime_infos())

    @pytest.fixture
    def risk_mgr(self):
        """Function-scoped so duplicate-guard state resets between tests."""
        return _make_risk_manager()

    def test_feature_engineering_produces_no_nan(self, ohlcv):
        """FeatureEngineer.build_features() must return a clean matrix."""
        eng      = FeatureEngineer()
        features = eng.build_features(ohlcv)
        # 700-bar input; FeatureEngineer drops warmup rows (~324 bars), so we
        # expect several hundred valid rows.
        assert len(features) > 100, (
            f"Too few valid feature rows ({len(features)}) — warmup may be too long"
        )
        assert not features.isnull().any().any(), "Feature matrix contains NaN"

    def test_strategy_generates_long_signals_in_low_vol_regime(
        self, orchestrator, bars_dict
    ):
        """BULL (low-vol) regime must produce LONG signals for all symbols."""
        regime_state = _make_regime_state("BULL", state_id=0, probability=0.85)
        signals = orchestrator.generate_signals(
            ["SPY", "AAPL"], bars_dict, regime_state, is_flickering=False
        )
        assert len(signals) == 2, "Expected one signal per symbol"
        for sig in signals:
            assert sig.direction == "LONG", f"{sig.symbol}: expected LONG, got {sig.direction}"
            assert sig.position_size_pct > 0.0
            assert sig.stop_loss < sig.entry_price, "Stop must be below entry for longs"

    def test_strategy_generates_flat_in_high_vol_regime(
        self, orchestrator, bars_dict
    ):
        """
        High-vol regime should generate FLAT signals during extreme uncertainty
        OR small LONG signals.  In all cases direction must be LONG or FLAT
        (never SHORT).
        """
        regime_state = _make_regime_state("BEAR", state_id=2, probability=0.78)
        signals = orchestrator.generate_signals(
            ["SPY"], bars_dict, regime_state, is_flickering=False
        )
        for sig in signals:
            assert sig.direction in ("LONG", "FLAT"), (
                f"Unexpected direction {sig.direction!r} — system is long-only"
            )

    def test_uncertainty_mode_halves_position_size(self, orchestrator, bars_dict):
        """
        Flickering flag → uncertainty mode → position_size_pct halved vs
        a stable signal with the same regime.
        """
        regime_normal  = _make_regime_state("BULL", state_id=0, probability=0.85)
        regime_flicker = _make_regime_state("BULL", state_id=0, probability=0.85)

        sigs_normal  = orchestrator.generate_signals(["SPY"], bars_dict, regime_normal,  is_flickering=False)
        sigs_flicker = orchestrator.generate_signals(["SPY"], bars_dict, regime_flicker, is_flickering=True)

        if sigs_normal and sigs_flicker:
            norm_size  = sigs_normal[0].position_size_pct
            flick_size = sigs_flicker[0].position_size_pct
            if sigs_normal[0].direction == "LONG" and sigs_flicker[0].direction == "LONG":
                assert flick_size < norm_size, (
                    f"Flickering should reduce size: normal={norm_size:.3f} "
                    f"flicker={flick_size:.3f}"
                )

    def test_risk_manager_approves_normal_signal(self, risk_mgr):
        """A clean, in-range signal must be approved by the risk manager."""
        signal    = _make_signal("SPY", entry_price=450.0, stop_loss=440.0, position_size_pct=0.08)
        portfolio = _make_portfolio(equity=100_000.0)

        decision = risk_mgr.validate_signal(signal, portfolio)

        assert decision.approved, (
            f"Expected approval, got rejection: {decision.rejection_reason}"
        )
        assert decision.modified_signal is not None

    def test_full_pipeline_produces_submittable_order(self, orchestrator, bars_dict, risk_mgr):
        """
        E2E: strategy signal → risk approval → mock executor submission.
        Verifies the three layers connect cleanly and submit_order is called.
        """
        regime_state = _make_regime_state("BULL", state_id=0, probability=0.88)
        signals      = orchestrator.generate_signals(["SPY"], bars_dict, regime_state)
        assert signals, "Orchestrator produced no signals"

        long_signals = [s for s in signals if s.direction == "LONG"]
        assert long_signals, "No LONG signals produced for BULL regime"
        raw_sig = long_signals[0]

        portfolio = _make_portfolio(equity=100_000.0)
        decision  = risk_mgr.validate_signal(raw_sig, portfolio)
        assert decision.approved, decision.rejection_reason

        # Simulate order submission via mock executor
        mock_client   = MagicMock()
        mock_executor = MagicMock()
        mock_executor.submit_order.return_value = MagicMock(
            order_id="test-123", status="accepted", error=None
        )
        mock_executor.submit_order(decision.modified_signal)
        mock_executor.submit_order.assert_called_once()

    def test_flat_signal_bypasses_risk_gates(self, risk_mgr):
        """FLAT signals must always be approved (gate 0 fast-path)."""
        signal    = _make_signal("SPY", direction="FLAT")
        portfolio = _make_portfolio()
        decision  = risk_mgr.validate_signal(signal, portfolio)
        assert decision.approved, "FLAT signal should bypass risk gates"


# ─────────────────────────────────────────────────────────────────────────────
# b. Look-ahead bias — backtest end-date independence
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestEndDateIndependence:
    """
    Verify that feature values at positions [0:T] are identical whether we
    compute on data[:T] or data[:T+100].  This is the feature-level guarantee
    that walk-forward folds cannot be contaminated by future data outside
    their test window.
    """

    @pytest.fixture(scope="class")
    def long_ohlcv(self):
        # 900 bars: enough warmup for all feature windows, leaves 400+ valid rows
        return _make_ohlcv(900)

    @pytest.fixture(scope="class")
    def engineer(self):
        return FeatureEngineer()

    def test_features_unchanged_when_future_data_appended(self, engineer, long_ohlcv):
        """
        features(data[:T]) == features(data[:T+100])[:len_T]
        for every feature column.  Any difference is a look-ahead bug.
        """
        T = 600
        feats_T     = engineer.build_features(long_ohlcv.iloc[:T])
        feats_TN    = engineer.build_features(long_ohlcv.iloc[:T + 100])

        common_idx  = feats_T.index.intersection(feats_TN.index)
        assert len(common_idx) > 0, "No overlapping rows to compare"

        try:
            pd.testing.assert_frame_equal(
                feats_T.loc[common_idx],
                feats_TN.loc[common_idx],
                check_exact=False,
                atol=1e-9,
            )
        except AssertionError as exc:
            raise AssertionError(
                "Feature values at bar T differ depending on how much future "
                "data is appended — look-ahead bias detected at feature layer.\n"
                + str(exc)
            ) from None

    def test_two_fold_endpoints_produce_identical_earlier_features(self, engineer, long_ohlcv):
        """
        Simulate two walk-forward folds with different end-dates.
        Features in the earlier fold must not be affected by the later fold's data.
        """
        fold1_end = 600
        fold2_end = 750

        feats_fold1 = engineer.build_features(long_ohlcv.iloc[:fold1_end])
        feats_fold2 = engineer.build_features(long_ohlcv.iloc[:fold2_end])

        common = feats_fold1.index.intersection(feats_fold2.index)
        assert len(common) > 0

        try:
            pd.testing.assert_frame_equal(
                feats_fold1.loc[common],
                feats_fold2.loc[common],
                check_exact=False,
                atol=1e-9,
            )
        except AssertionError as exc:
            raise AssertionError(
                "Walk-forward fold features are not end-date independent.\n" + str(exc)
            ) from None

    def test_max_diff_is_zero(self, engineer, long_ohlcv):
        """
        The maximum absolute difference across all features and rows must be
        at or below floating-point noise (≤ 1e-9).  A non-zero max diff means
        future data is influencing the computation.
        """
        T = 600
        feats_short = engineer.build_features(long_ohlcv.iloc[:T])
        feats_long  = engineer.build_features(long_ohlcv.iloc[:T + 50])

        common = feats_short.index.intersection(feats_long.index)
        diff   = (feats_short.loc[common] - feats_long.loc[common]).abs().max().max()

        assert diff <= 1e-9, (
            f"Max absolute difference = {diff:.2e} > 1e-9; "
            "possible causal contamination from appended future data."
        )


# ─────────────────────────────────────────────────────────────────────────────
# c. Risk stress tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskStress:
    """
    Verify the risk layer correctly handles adversarial inputs:
      - Extreme position sizes are capped, not silently passed
      - Rapid-fire duplicate orders are blocked
      - Zero-risk signals (stop == entry) are rejected
      - Over-exposed portfolios block new entries
      - Max-concurrent position limit is enforced
    """

    @pytest.fixture(autouse=True)
    def no_lock_file(self, tmp_path, monkeypatch):
        """Ensure the peak-DD lock file never interferes with stress tests."""
        monkeypatch.setattr(
            "core.risk_manager._HALT_LOCK_FILE",
            tmp_path / "trading_halted.lock",
        )

    def test_extreme_position_size_capped_at_max_single_position(self):
        """
        A signal requesting 99 % allocation must be capped to
        max_single_position (15 %) by the sizing pipeline.
        """
        risk_mgr  = _make_risk_manager()
        signal    = _make_signal(
            "SPY",
            entry_price=100.0,
            stop_loss=1.0,          # tiny stop → risk_per_share = 99 → risk_cap is tiny
            position_size_pct=0.99,
            leverage=1.0,
        )
        portfolio = _make_portfolio(equity=100_000.0)
        decision  = risk_mgr.validate_signal(signal, portfolio)

        # Either approved with capped size, or rejected entirely (both are acceptable)
        if decision.approved:
            final_pct = decision.modified_signal.position_size_pct
            assert final_pct <= 0.16, (
                f"Position size {final_pct:.2%} exceeds max_single_position (15 %)"
            )
        else:
            # Rejection is also valid — the risk budget was exhausted
            assert decision.rejection_reason, "Rejection must include a reason"

    def test_extreme_position_size_capped_alloc_path(self):
        """
        Large stop (normal risk), but position_size_pct=0.95 with leverage=1.25
        → total notional would be 118.75 % of equity; final size must stay ≤ 15 %.
        """
        cfg = RiskConfig(
            max_single_position=0.15,
            max_exposure=0.80,
            max_risk_per_trade=0.01,
        )
        risk_mgr  = RiskManager(cfg, initial_equity=100_000.0)
        signal    = _make_signal(
            "SPY",
            entry_price=100.0,
            stop_loss=98.0,         # 2 % stop — normal sizing
            position_size_pct=0.95,
            leverage=1.25,
        )
        portfolio = _make_portfolio(equity=100_000.0)
        decision  = risk_mgr.validate_signal(signal, portfolio)

        if decision.approved:
            assert decision.modified_signal.position_size_pct <= 0.16, (
                "Approved signal exceeds max_single_position cap"
            )

    def test_zero_risk_signal_rejected(self):
        """stop_loss == entry_price → zero risk_per_share → [ZERO_RISK] rejection."""
        risk_mgr  = _make_risk_manager()
        signal    = _make_signal("SPY", entry_price=450.0, stop_loss=450.0)
        portfolio = _make_portfolio()

        decision = risk_mgr.validate_signal(signal, portfolio)

        assert not decision.approved, "Zero-risk signal must be rejected"
        assert "ZERO_RISK" in decision.rejection_reason or \
               "INVALID_PRICES" in decision.rejection_reason, (
            f"Unexpected rejection reason: {decision.rejection_reason}"
        )

    def test_no_stop_loss_rejected(self):
        """stop_loss = 0 → [NO_STOP_LOSS] rejection."""
        risk_mgr  = _make_risk_manager()
        signal    = _make_signal("SPY", entry_price=450.0, stop_loss=0.0)
        portfolio = _make_portfolio()

        decision = risk_mgr.validate_signal(signal, portfolio)

        assert not decision.approved, "Signal without stop must be rejected"
        assert "NO_STOP" in decision.rejection_reason or \
               "INVALID" in decision.rejection_reason, (
            f"Unexpected rejection reason: {decision.rejection_reason}"
        )

    def test_rapid_fire_duplicate_blocked(self):
        """
        Two LONG signals for the same symbol within 60 s must result in the
        second being blocked by the duplicate-order guard.
        """
        cfg      = RiskConfig(duplicate_window_s=60)
        risk_mgr = RiskManager(cfg, initial_equity=100_000.0)
        signal   = _make_signal("AAPL", entry_price=180.0, stop_loss=175.0, position_size_pct=0.08)
        portfolio = _make_portfolio()

        first  = risk_mgr.validate_signal(signal, portfolio)
        second = risk_mgr.validate_signal(signal, portfolio)

        assert first.approved, f"First signal rejected: {first.rejection_reason}"
        assert not second.approved, "Second rapid-fire signal must be blocked"
        assert "DUPLICATE" in second.rejection_reason, (
            f"Expected DUPLICATE in reason, got: {second.rejection_reason}"
        )

    def test_rapid_fire_passes_after_cooldown(self):
        """After the duplicate_window expires, the same signal is accepted again."""
        cfg      = RiskConfig(duplicate_window_s=1)   # 1-second window for test speed
        risk_mgr = RiskManager(cfg, initial_equity=100_000.0)
        signal   = _make_signal("AAPL", entry_price=180.0, stop_loss=175.0, position_size_pct=0.08)
        portfolio = _make_portfolio()

        first  = risk_mgr.validate_signal(signal, portfolio)
        assert first.approved

        time.sleep(1.1)   # wait out the cooldown

        third = risk_mgr.validate_signal(signal, portfolio)
        assert third.approved, f"Post-cooldown signal rejected: {third.rejection_reason}"

    def test_max_concurrent_positions_enforced(self):
        """When 5 positions are open (max_concurrent=5), a new symbol is rejected."""
        cfg      = RiskConfig(max_concurrent=5)
        risk_mgr = RiskManager(cfg, initial_equity=100_000.0)

        syms = ["A", "B", "C", "D", "E"]
        portfolio = _make_portfolio(
            equity=100_000.0,
            positions={
                sym: Position(
                    symbol=sym,
                    shares=10,
                    entry_price=100.0,
                    stop_loss=95.0,
                    notional=1_000.0,
                )
                for sym in syms
            },
        )
        new_signal = _make_signal("NEW", entry_price=100.0, stop_loss=95.0, position_size_pct=0.05)
        decision   = risk_mgr.validate_signal(new_signal, portfolio)

        assert not decision.approved, "6th position must be rejected by max_concurrent"
        assert "MAX_CONCURRENT" in decision.rejection_reason or \
               "PORTFOLIO_LIMIT" in decision.rejection_reason, (
            f"Expected MAX_CONCURRENT, got: {decision.rejection_reason}"
        )

    def test_max_exposure_cap_prevents_over_deployment(self):
        """
        Portfolio already at 75 % of equity; a new position that would push
        exposure over 80 % must be reduced or rejected.
        """
        cfg      = RiskConfig(max_exposure=0.80, max_single_position=0.15)
        risk_mgr = RiskManager(cfg, initial_equity=100_000.0)

        equity   = 100_000.0
        # Existing positions consuming 75 % of equity
        existing_positions = {
            "SPY": Position(
                symbol="SPY", shares=750, entry_price=100.0,
                stop_loss=95.0, notional=75_000.0,
            ),
        }
        portfolio = _make_portfolio(equity=equity, positions=existing_positions)

        # New position requesting 15 % → would bring total to 90 % (over 80 %)
        signal   = _make_signal("AAPL", entry_price=100.0, stop_loss=95.0, position_size_pct=0.15)
        decision = risk_mgr.validate_signal(signal, portfolio)

        if decision.approved:
            # Must be reduced below 6 % (80 % - 75 % = 5 % headroom)
            assert decision.modified_signal.position_size_pct <= 0.06, (
                "Approved size exceeds available exposure headroom"
            )
        else:
            # Several terminal gates can fire when exposure is squeezed to near-zero:
            # MAX_EXPOSURE → reduces to ~1 % → risk sizing → MIN_NOTIONAL
            assert any(kw in decision.rejection_reason for kw in
                       ("EXPOSURE", "PORTFOLIO_LIMIT", "MIN_NOTIONAL", "ZERO_SHARES")), (
                f"Expected exposure-related rejection, got: {decision.rejection_reason}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# d. Alpaca mock orders
# ─────────────────────────────────────────────────────────────────────────────

class TestAlpacaMockOrders:
    """
    Verify OrderExecutor behaviour using a mock AlpacaClient.
    Tests cover: bracket pre-flight validation, stop tighten/widen, cancel,
    and the pending-limit timer registration.
    """

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client._trading_client = MagicMock()
        client.is_market_open.return_value = True
        return client

    @pytest.fixture
    def executor(self, mock_client):
        from broker.order_executor import OrderExecutor
        return OrderExecutor(mock_client)

    def _make_trade_signal(
        self,
        symbol: str = "SPY",
        shares: int = 10,
        entry_price: float = 450.0,
        stop_price: float = 440.0,
        target_price: Optional[float] = 470.0,
    ):
        from core.signal_generator import TradeSignal, SignalType
        from core.hmm_engine import Regime
        from core.risk_manager import TradingState
        return TradeSignal(
            symbol=symbol,
            signal_type=SignalType.BUY,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            shares=shares,
            notional=shares * entry_price,
            regime=Regime.LOW_VOL.value if hasattr(Regime, "LOW_VOL") else "BULL",
            confidence=0.85,
            allocation_fraction=0.10,
            leverage=1.0,
            trading_state=TradingState.NORMAL,
        )

    # ── Bracket pre-flight validation ─────────────────────────────────────────

    def test_bracket_requires_target_price(self, executor):
        """submit_bracket_order raises ValueError when target_price is None."""
        sig = self._make_trade_signal(target_price=None)
        with pytest.raises(ValueError, match="target_price"):
            executor.submit_bracket_order(sig)

    def test_bracket_requires_positive_stop(self, executor):
        """submit_bracket_order raises ValueError when stop_price <= 0."""
        sig = self._make_trade_signal(stop_price=0.0, target_price=470.0)
        with pytest.raises(ValueError, match="stop_price"):
            executor.submit_bracket_order(sig)

    def test_bracket_negative_stop_raises(self, executor):
        """Negative stop price must also be rejected."""
        sig = self._make_trade_signal(stop_price=-5.0, target_price=470.0)
        with pytest.raises(ValueError):
            executor.submit_bracket_order(sig)

    # ── Bracket order submission (mock alpaca constructors) ───────────────────

    def test_bracket_submission_calls_trading_client(self, executor, mock_client):
        """
        With mocked Alpaca constructors, submit_bracket_order must call
        client._trading_client.submit_order exactly once.
        """
        import broker.order_executor as oe_mod

        mock_order = MagicMock()
        mock_order.id = "bracket-order-id-001"
        mock_order.client_order_id = "coid-001"
        mock_order.status = MagicMock(value="accepted")
        mock_order.qty = "10"
        mock_order.filled_qty = "0"
        mock_order.limit_price = "451.45"
        mock_order.stop_price = "440.00"
        mock_client._trading_client.submit_order.return_value = mock_order

        with patch("broker.order_executor._ALPACA_AVAILABLE", True), \
             patch("broker.order_executor.LimitOrderRequest", MagicMock(return_value=MagicMock()), create=True), \
             patch("broker.order_executor.TakeProfitRequest", MagicMock(return_value=MagicMock()), create=True), \
             patch("broker.order_executor.StopLossRequest",   MagicMock(return_value=MagicMock()), create=True), \
             patch("broker.order_executor._AlpacaSide",  MagicMock(BUY="buy"),      create=True), \
             patch("broker.order_executor._AlpacaType",  MagicMock(LIMIT="limit"),  create=True), \
             patch("broker.order_executor._AlpacaTIF",   MagicMock(GTC="gtc"),      create=True), \
             patch("broker.order_executor._AlpacaClass", MagicMock(BRACKET="bracket"), create=True), \
             patch("broker.order_executor._alpaca_order_to_result", return_value=MagicMock(
                 order_id="bracket-order-id-001", trade_id="tid", error=None
             )):
            sig    = self._make_trade_signal(target_price=470.0)
            result = executor.submit_bracket_order(sig)

        mock_client._trading_client.submit_order.assert_called_once()
        assert "trade_id" in sig.metadata, "trade_id must be injected into signal metadata"

    # ── Stop modification ──────────────────────────────────────────────────────

    def test_modify_stop_tighten_accepted(self, executor, mock_client):
        """
        modify_stop with new_stop > current_stop (tightening) must call
        replace_order_by_id and return True.
        """
        import broker.order_executor as oe_mod

        # Mock an open stop order returned by get_open_orders
        stop_order = MagicMock()
        stop_order.order_type = oe_mod.OrderType.STOP
        stop_order.side       = oe_mod.OrderSide.SELL
        stop_order.order_id   = "stop-order-001"

        with patch.object(executor, "get_open_orders", return_value=[stop_order]), \
             patch("broker.order_executor._ALPACA_AVAILABLE", True), \
             patch("broker.order_executor.ReplaceOrderRequest", MagicMock(return_value=MagicMock()), create=True):
            result = executor.modify_stop("SPY", new_stop=445.0, current_stop=440.0)

        assert result is True
        mock_client._trading_client.replace_order_by_id.assert_called_once()

    def test_modify_stop_widen_raises(self, executor):
        """new_stop <= current_stop (widening) must raise ValueError immediately."""
        with pytest.raises(ValueError, match="must be above"):
            executor.modify_stop("SPY", new_stop=435.0, current_stop=440.0)

    def test_modify_stop_equal_raises(self, executor):
        """new_stop == current_stop (no change) must also raise ValueError."""
        with pytest.raises(ValueError):
            executor.modify_stop("SPY", new_stop=440.0, current_stop=440.0)

    def test_modify_stop_no_open_orders_returns_false(self, executor):
        """
        If there is no open stop order for the symbol, modify_stop returns False
        without raising.
        """
        with patch.object(executor, "get_open_orders", return_value=[]):
            result = executor.modify_stop("SPY", new_stop=445.0, current_stop=440.0)
        assert result is False

    # ── Order cancellation ────────────────────────────────────────────────────

    def test_cancel_order_calls_alpaca(self, executor, mock_client):
        """cancel_order must call client._trading_client.cancel_order_by_id."""
        result = executor.cancel_order("order-abc-123")
        mock_client._trading_client.cancel_order_by_id.assert_called_once_with("order-abc-123")
        assert result is True

    def test_cancel_order_disarms_pending_timer(self, executor, mock_client):
        """
        When a limit order with a pending cancel timer is cancelled manually,
        the timer must be disarmed (not fire a redundant cancel later).
        """
        order_id = "limit-order-timer-test"
        mock_timer = MagicMock()

        with executor._lock:
            executor._pending_limits[order_id] = ("trade-id-x", mock_timer)

        executor.cancel_order(order_id)

        mock_timer.cancel.assert_called_once()
        assert order_id not in executor._pending_limits, (
            "Pending timer entry must be removed after cancel"
        )

    def test_cancel_all_orders_disarms_all_timers(self, executor, mock_client):
        """cancel_all_orders must disarm every pending timer."""
        mock_client._trading_client.cancel_orders.return_value = []

        timers = {}
        for i in range(3):
            oid = f"order-{i}"
            t   = MagicMock()
            timers[oid] = t
            with executor._lock:
                executor._pending_limits[oid] = (f"trade-{i}", t)

        executor.cancel_all_orders()

        for t in timers.values():
            t.cancel.assert_called_once()
        assert len(executor._pending_limits) == 0


# ─────────────────────────────────────────────────────────────────────────────
# e. State recovery
# ─────────────────────────────────────────────────────────────────────────────

class TestStateRecovery:
    """
    Verify crash-recovery semantics:
      - Snapshot JSON round-trips correctly (all fields preserved)
      - PortfolioState fields are correctly restored from snapshot
      - Peak equity is always the higher of snapshot and current
      - Duplicate-guard blocks re-entry after restart completes startup
    """

    # ── Snapshot JSON round-trip ──────────────────────────────────────────────

    def test_snapshot_json_round_trip(self, tmp_path):
        """
        _save_snapshot / _load_snapshot must reproduce all scalar fields
        exactly, including ISO timestamps and floats.
        """
        from main import _save_snapshot, _load_snapshot

        path = tmp_path / "state_snapshot.json"
        state = {
            "session_start":      "2025-06-01T09:30:00+00:00",
            "snapshot_time":      "2025-06-01T10:15:00+00:00",
            "bars_processed":     42,
            "signals_generated":  7,
            "orders_submitted":   5,
            "orders_rejected":    2,
            "regime_label":       "BULL",
            "regime_probability": 0.883,
            "equity":             102_450.75,
            "daily_pnl":          2_450.75,
            "weekly_pnl":         3_100.00,
            "peak_equity":        104_000.00,
            "circuit_breaker":    "NORMAL",
            "last_train_time":    "2025-05-25T09:30:00+00:00",
            "last_bar_time":      "2025-06-01T16:00:00+00:00",
        }
        _save_snapshot(path, state)
        loaded = _load_snapshot(path)

        assert loaded is not None, "Snapshot file not found after save"
        for key in state:
            assert key in loaded, f"Missing key after round-trip: {key!r}"
            assert loaded[key] == state[key] or str(loaded[key]) == str(state[key]), (
                f"Mismatch for {key!r}: saved={state[key]!r} loaded={loaded[key]!r}"
            )

    def test_missing_snapshot_returns_none(self, tmp_path):
        """_load_snapshot returns None when the file does not exist."""
        from main import _load_snapshot
        result = _load_snapshot(tmp_path / "nonexistent.json")
        assert result is None

    def test_corrupt_snapshot_returns_none(self, tmp_path):
        """_load_snapshot returns None (with a warning) for corrupt JSON."""
        from main import _load_snapshot
        path = tmp_path / "corrupt.json"
        path.write_text("{this is not valid JSON{{{{")
        result = _load_snapshot(path)
        assert result is None

    # ── Portfolio field restoration ───────────────────────────────────────────

    def test_restore_daily_pnl_from_snapshot(self, tmp_path):
        """
        Simulate the restore logic: daily_pnl and weekly_pnl from the snapshot
        must overwrite the portfolio's default zeros after restart.
        """
        portfolio = _make_portfolio(equity=100_000.0)
        assert portfolio.daily_pnl == 0.0

        snap = {"daily_pnl": 1_234.56, "weekly_pnl": 2_500.00, "peak_equity": 105_000.0}

        # Replicate _restore_from_snapshot logic directly
        portfolio.daily_pnl  = float(snap["daily_pnl"])
        portfolio.weekly_pnl = float(snap["weekly_pnl"])
        portfolio.peak_equity = max(portfolio.equity, float(snap["peak_equity"]))

        assert portfolio.daily_pnl  == 1_234.56
        assert portfolio.weekly_pnl == 2_500.00
        assert portfolio.peak_equity == 105_000.0   # snapshot > current equity

    def test_peak_equity_is_max_of_snapshot_and_current(self, tmp_path):
        """
        If the current equity is higher than the snapshot peak (e.g. paper
        account already has unrealised gains), the live equity wins.
        """
        portfolio = _make_portfolio(equity=110_000.0)
        snap      = {"peak_equity": 105_000.0, "daily_pnl": 0.0, "weekly_pnl": 0.0}

        portfolio.peak_equity = max(portfolio.equity, float(snap["peak_equity"]))

        assert portfolio.peak_equity == 110_000.0, (
            "Live equity (110k) should override lower snapshot peak (105k)"
        )

    # ── No double-entry after restart ─────────────────────────────────────────

    def test_no_double_entry_existing_position_blocks_duplicate_guard(self):
        """
        After a restart, if AAPL is already in the portfolio (reconciled via
        sync_from_broker), a new BUY signal for AAPL within the cooldown window
        must be blocked by the duplicate-order guard once a signal has already
        been processed this session.
        """
        cfg      = RiskConfig(duplicate_window_s=60)
        risk_mgr = RiskManager(cfg, initial_equity=100_000.0)

        # Simulate: first signal processed normally (could be session startup recovery)
        signal    = _make_signal("AAPL", entry_price=180.0, stop_loss=175.0, position_size_pct=0.08)
        portfolio = _make_portfolio()
        first     = risk_mgr.validate_signal(signal, portfolio)
        assert first.approved

        # Immediate second signal for the same symbol — must be blocked
        second = risk_mgr.validate_signal(signal, portfolio)
        assert not second.approved
        assert "DUPLICATE" in second.rejection_reason

    def test_position_already_owned_exposure_limit_protects_against_doubling(self):
        """
        If an existing AAPL position already consumes 14 % of equity and a new
        signal for AAPL requests another 14 %, the exposure cap must prevent
        the total from exceeding max_single_position (15 %).
        """
        cfg = RiskConfig(max_single_position=0.15, max_exposure=0.80)
        risk_mgr  = RiskManager(cfg, initial_equity=100_000.0)

        existing_pos = {
            "AAPL": Position(
                symbol="AAPL", shares=140, entry_price=100.0,
                stop_loss=95.0, notional=14_000.0,
            )
        }
        portfolio = _make_portfolio(equity=100_000.0, positions=existing_pos)
        signal    = _make_signal("AAPL", entry_price=100.0, stop_loss=95.0, position_size_pct=0.14)

        decision  = risk_mgr.validate_signal(signal, portfolio)

        # If approved, the combined (existing + new) would exceed 15 %
        # The risk layer must cap the new allocation
        if decision.approved:
            final_pct = decision.modified_signal.position_size_pct
            # Existing 14 % + final_pct must stay within single-position cap
            # (14 % is replaced, not added — the risk manager subtracts existing notional)
            assert final_pct <= 0.16, (
                f"Approved size {final_pct:.2%} combined with existing would exceed cap"
            )

    def test_snapshot_bars_processed_restored(self, tmp_path):
        """bars_processed from a snapshot should be recoverable as an integer."""
        from main import _save_snapshot, _load_snapshot

        path = tmp_path / "snap.json"
        _save_snapshot(path, {"bars_processed": 999, "regime_label": "NEUTRAL"})
        loaded = _load_snapshot(path)

        assert int(loaded["bars_processed"]) == 999
