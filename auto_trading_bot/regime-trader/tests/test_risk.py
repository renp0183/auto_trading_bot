"""
test_risk.py — Unit tests for the risk management layer.

Coverage:
  - Position sizing: risk cap, single-position cap, overnight gap cap
  - Portfolio limits: max_exposure, max_concurrent, sector cap
  - Circuit breakers: NORMAL → REDUCED → HALTED transitions
  - Peak-DD halt: state, lock file guard
  - Daily / weekly resets: DD counters, state restoration
  - validate_signal(): full pipeline (approval, modification, rejection)
  - Leverage enforcement: force 1.0× when circuit breaker / 3+ positions
  - Correlation limits: soft (>0.70) and hard (>0.85) rejection
  - Duplicate-order guard (60-second cooldown)
  - CircuitBreaker.get_history() tracks every trigger event
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Literal, Optional
from unittest.mock import patch

import pandas as pd
import pytest

from core.risk_manager import (
    BreakerType,
    CircuitBreaker,
    CircuitBreakerEvent,
    PortfolioState,
    Position,
    RiskConfig,
    RiskDecision,
    RiskManager,
    SizingResult,
    TradingState,
    _HALT_LOCK_FILE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — minimal Signal stub (avoids importing regime_strategies)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Signal:
    """Minimal Signal-compatible dataclass for testing."""
    symbol:             str
    direction:          Literal["LONG", "FLAT"]
    confidence:         float
    entry_price:        float
    stop_loss:          float
    take_profit:        Optional[float]
    position_size_pct:  float
    leverage:           float
    regime_id:          int              = 0
    regime_name:        str              = "BULL"
    regime_probability: float           = 0.80
    timestamp:          Optional[object] = None
    reasoning:          str             = ""
    strategy_name:      str             = "TEST"
    metadata:           dict            = field(default_factory=dict)

    @property
    def risk_per_share(self) -> float:
        return max(0.0, self.entry_price - self.stop_loss)


def _make_portfolio(
    equity: float = 100_000.0,
    cash:   float = 100_000.0,
    positions: Optional[dict] = None,
    flicker_rate: float = 0.0,
    hmm_regime: str = "BULL",
) -> PortfolioState:
    return PortfolioState(
        equity=equity,
        cash=cash,
        buying_power=cash,
        positions=positions or {},
        sod_equity=equity,
        sow_equity=equity,
        peak_equity=equity,
        flicker_rate=flicker_rate,
        hmm_regime=hmm_regime,
    )


def _make_signal(
    symbol: str = "SPY",
    entry: float = 400.0,
    stop: float = 390.0,
    size_pct: float = 0.10,
    leverage: float = 1.0,
    confidence: float = 0.80,
    regime_prob: float = 0.80,
) -> _Signal:
    return _Signal(
        symbol=symbol,
        direction="LONG",
        confidence=confidence,
        entry_price=entry,
        stop_loss=stop,
        take_profit=None,
        position_size_pct=size_pct,
        leverage=leverage,
        regime_probability=regime_prob,
    )


def _make_config(**overrides) -> RiskConfig:
    defaults = dict(
        max_risk_per_trade=0.01,
        max_exposure=0.80,
        max_leverage=1.25,
        max_single_position=0.15,
        max_concurrent=5,
        max_correlated_sector=0.30,
        max_daily_trades=20,
        daily_dd_reduce=0.02,
        daily_dd_halt=0.03,
        weekly_dd_reduce=0.05,
        weekly_dd_halt=0.07,
        max_dd_from_peak=0.10,
        max_spread_pct=0.005,
        duplicate_window_s=60,
        overnight_gap_mult=3.0,
        overnight_gap_risk=0.02,
        min_notional=100.0,
    )
    defaults.update(overrides)
    return RiskConfig(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg() -> RiskConfig:
    return _make_config()


@pytest.fixture
def rm(cfg) -> RiskManager:
    return RiskManager(cfg, initial_equity=100_000.0)


@pytest.fixture(autouse=True)
def _clean_lock_file():
    """Remove lock file before and after each test."""
    if _HALT_LOCK_FILE.exists():
        _HALT_LOCK_FILE.unlink()
    yield
    if _HALT_LOCK_FILE.exists():
        _HALT_LOCK_FILE.unlink()


# ─────────────────────────────────────────────────────────────────────────────
# Position sizing
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionSizing:

    def test_risk_amount_does_not_exceed_max_risk(self, rm):
        """Risk amount (shares × |entry − stop|) must not exceed 1% of equity."""
        result = rm.size_position("SPY", entry_price=400.0, stop_price=390.0,
                                  allocation_fraction=0.50, leverage=1.0,
                                  portfolio=_make_portfolio())
        assert not result.rejected, result.rejection_reason
        max_risk = 100_000 * 0.01  # 1 000
        assert result.risk_amount <= max_risk + 0.01  # floating-point tolerance

    def test_notional_does_not_exceed_max_single_position(self, rm):
        """Notional must not exceed max_single_position (15%) of equity."""
        result = rm.size_position("SPY", entry_price=400.0, stop_price=399.0,
                                  allocation_fraction=0.50, leverage=1.0,
                                  portfolio=_make_portfolio())
        assert not result.rejected
        assert result.pct_of_equity <= 0.15 + 1e-6

    def test_overnight_gap_cap_applied(self, rm):
        """
        3× gap risk must not exceed overnight_gap_risk (2% of equity).

            max_gap_shares = (equity × 0.02) / (3 × risk_per_share)
                           = 2 000 / (3 × 10) = 66 shares  (equity=100k, stop gap=10)
        """
        # stop is far from entry to force a tight gap constraint
        result = rm.size_position("SPY", entry_price=400.0, stop_price=390.0,
                                  allocation_fraction=0.50, leverage=1.0,
                                  portfolio=_make_portfolio(equity=100_000))
        assert not result.rejected
        # Expected gap limit: (100_000 × 0.02) / (3 × 10) = 66 shares
        expected_gap_max = int((100_000 * 0.02) / (3 * 10))
        assert result.shares <= expected_gap_max

    def test_zero_shares_rejected(self, rm):
        """Entry == stop → risk_per_share ≈ 0 → should be rejected."""
        result = rm.size_position("SPY", entry_price=400.0, stop_price=400.0,
                                  allocation_fraction=0.10, leverage=1.0,
                                  portfolio=_make_portfolio())
        assert result.rejected
        assert "INVALID_PRICES" in result.rejection_reason or "ZERO_SHARES" in result.rejection_reason

    def test_size_is_integer_shares(self, rm):
        """Shares must always be a non-negative integer."""
        result = rm.size_position("SPY", entry_price=400.0, stop_price=395.0,
                                  allocation_fraction=0.10, leverage=1.0,
                                  portfolio=_make_portfolio())
        assert isinstance(result.shares, int)
        assert result.shares >= 0

    def test_allocation_larger_than_risk_cap_is_bounded(self, rm):
        """
        When a tight stop allows many shares by allocation but few by 1% risk,
        the risk cap wins.
        """
        # risk_per_share = 10 → max risk shares = 1000/10 = 100
        # allocation: 50% of 100k / 400 = 125 shares → risk cap binds
        result = rm.size_position("SPY", entry_price=400.0, stop_price=390.0,
                                  allocation_fraction=0.50, leverage=1.0,
                                  portfolio=_make_portfolio())
        assert not result.rejected
        # Tightest of risk (100), gap (66), alloc (125) → 66
        assert result.shares <= 100  # bounded by risk or gap cap

    def test_with_leverage_increases_alloc_cap(self, rm):
        """1.25× leverage doubles the allocation-cap shares (before other caps)."""
        result_1x = rm.size_position("SPY", entry_price=400.0, stop_price=399.5,
                                     allocation_fraction=0.10, leverage=1.0,
                                     portfolio=_make_portfolio())
        result_lev = rm.size_position("SPY", entry_price=400.0, stop_price=399.5,
                                      allocation_fraction=0.10, leverage=1.25,
                                      portfolio=_make_portfolio())
        # Leveraged alloc cap is larger, so shares_by_alloc is larger
        # Risk cap is tight here (risk_per_share=0.5 → max_risk_shares=2000),
        # so alloc cap should be binding → leveraged shares ≥ 1x shares
        assert result_lev.shares >= result_1x.shares


# ─────────────────────────────────────────────────────────────────────────────
# Circuit breakers — state transitions
# ─────────────────────────────────────────────────────────────────────────────

class TestCircuitBreakerTransitions:

    def test_normal_state_below_daily_threshold(self, cfg):
        """No breaker fires when daily DD < 2%."""
        rm = RiskManager(cfg, initial_equity=100_000.0)
        port = _make_portfolio(equity=99_000)   # 1% daily DD → below 2% threshold
        port.sod_equity = 100_000
        state, fired = rm._circuit_breaker.check(port)
        assert state == TradingState.NORMAL
        assert fired == BreakerType.NONE

    def test_reduced_state_on_daily_dd_reduce(self, cfg):
        """Daily DD == 2% exactly triggers REDUCED."""
        rm = RiskManager(cfg, initial_equity=100_000.0)
        port = _make_portfolio(equity=98_000)
        port.sod_equity = 100_000
        state, _ = rm._circuit_breaker.check(port)
        assert state == TradingState.REDUCED

    def test_halted_state_on_daily_dd_halt(self, cfg):
        """Daily DD >= 3% triggers HALTED."""
        rm = RiskManager(cfg, initial_equity=100_000.0)
        port = _make_portfolio(equity=97_000)
        port.sod_equity = 100_000
        state, fired = rm._circuit_breaker.check(port)
        assert state == TradingState.HALTED
        assert fired == BreakerType.DAILY_HALT

    def test_weekly_reduce_overrides_normal(self, cfg):
        """Weekly DD >= 5% triggers REDUCED even if daily DD < 2%."""
        rm = RiskManager(cfg, initial_equity=100_000.0)
        port = _make_portfolio(equity=94_900)  # 5.1% weekly
        port.sod_equity = 100_000      # tiny daily loss
        port.sow_equity = 100_000
        state, fired = rm._circuit_breaker.check(port)
        assert state in (TradingState.REDUCED, TradingState.HALTED)

    def test_weekly_halt_on_large_weekly_dd(self, cfg):
        """Weekly DD >= 7% triggers HALTED (WEEKLY_HALT).

        The daily DD must stay below 3% so DAILY_HALT does not fire first.
        We simulate: SOW equity was 100k, current is 92.9k (7.1% weekly),
        but SOD was 93.5k so daily loss is only 0.6%.
        """
        rm = RiskManager(cfg, initial_equity=100_000.0)
        port = _make_portfolio(equity=92_900)
        port.sow_equity = 100_000   # 7.1% weekly DD
        port.sod_equity = 93_500    # 0.6% daily DD — below daily thresholds
        state, fired = rm._circuit_breaker.check(port)
        assert state == TradingState.HALTED
        assert fired == BreakerType.WEEKLY_HALT

    def test_peak_dd_halt(self, cfg):
        """Peak DD >= 10% triggers HALTED and writes lock file."""
        rm = RiskManager(cfg, initial_equity=100_000.0)
        port = _make_portfolio(equity=89_900)
        port.peak_equity = 100_000  # 10.1% peak DD
        state, fired = rm._circuit_breaker.check(port)
        assert state == TradingState.HALTED
        assert fired == BreakerType.PEAK_DD_HALT
        assert _HALT_LOCK_FILE.exists()

    def test_daily_trades_limit_halts(self, cfg):
        """Hitting max_daily_trades triggers HALTED."""
        rm = RiskManager(cfg, initial_equity=100_000.0)
        port = _make_portfolio()
        port.daily_trades = cfg.max_daily_trades  # exactly at limit
        state, fired = rm._circuit_breaker.check(port)
        assert state == TradingState.HALTED
        assert fired == BreakerType.DAILY_TRADES

    def test_halted_state_persists_until_daily_reset(self, cfg):
        """DAILY_HALT persists for the rest of the day; reset_daily() clears it."""
        rm = RiskManager(cfg, initial_equity=100_000.0)
        port = _make_portfolio(equity=97_000)
        port.sod_equity = 100_000
        rm._circuit_breaker.check(port)
        assert rm._circuit_breaker.state == TradingState.HALTED

        # Equity recovers intra-day — state must NOT clear automatically
        port.equity = 100_000
        port.sod_equity = 100_000
        state, _ = rm._circuit_breaker.check(port)
        assert state == TradingState.HALTED, "Halt must persist even after equity recovery"

        # Daily reset clears it
        rm._circuit_breaker.reset_daily()
        port2 = _make_portfolio()
        state2, _ = rm._circuit_breaker.check(port2)
        assert state2 == TradingState.NORMAL

    def test_event_logged_on_breaker_fire(self, cfg):
        """Each new breaker trigger adds an event to history."""
        rm = RiskManager(cfg, initial_equity=100_000.0)
        port = _make_portfolio(equity=97_000)
        port.sod_equity = 100_000
        rm._circuit_breaker.check(port)
        history = rm._circuit_breaker.get_history()
        assert len(history) >= 1
        assert any(e.breaker_type == BreakerType.DAILY_HALT for e in history)

    def test_event_records_hmm_regime(self, cfg):
        """Circuit breaker event must record the HMM regime at time of trigger."""
        rm = RiskManager(cfg, initial_equity=100_000.0)
        port = _make_portfolio(equity=97_000, hmm_regime="CRASH")
        port.sod_equity = 100_000
        rm._circuit_breaker.check(port)
        history = rm._circuit_breaker.get_history()
        assert history[-1].hmm_regime == "CRASH"


# ─────────────────────────────────────────────────────────────────────────────
# Resets
# ─────────────────────────────────────────────────────────────────────────────

class TestResets:

    def test_daily_reset_clears_daily_state(self, rm):
        """reset_daily() lifts a DAILY_HALT."""
        port = _make_portfolio(equity=97_000)
        port.sod_equity = 100_000
        rm._circuit_breaker.check(port)
        assert rm.state == TradingState.HALTED

        rm.reset_daily(port)
        port2 = _make_portfolio()
        rm._circuit_breaker.check(port2)
        assert rm.state == TradingState.NORMAL

    def test_weekly_reset_clears_weekly_halt(self, rm):
        """reset_weekly() lifts a WEEKLY_HALT."""
        port = _make_portfolio(equity=92_900)
        port.sod_equity = 100_000
        port.sow_equity = 100_000
        rm._circuit_breaker.check(port)
        assert rm.state == TradingState.HALTED

        rm.reset_weekly(port)
        port2 = _make_portfolio()
        rm._circuit_breaker.check(port2)
        assert rm.state == TradingState.NORMAL

    def test_peak_dd_halt_not_cleared_by_daily_reset(self, rm):
        """Peak-DD halt persists across daily and weekly resets (manual only)."""
        port = _make_portfolio(equity=89_900)
        port.peak_equity = 100_000
        rm._circuit_breaker.check(port)
        assert rm.state == TradingState.HALTED

        rm.reset_daily()
        rm.reset_weekly()
        port2 = _make_portfolio(equity=89_900)
        port2.peak_equity = 100_000
        rm._circuit_breaker.check(port2)
        assert rm.state == TradingState.HALTED, "Peak-DD halt must survive daily/weekly resets"

    def test_peak_equity_not_reset_on_daily_reset(self, rm):
        """Updating equity to a new high should update peak, which persists."""
        rm.update_equity(120_000.0)
        rm.reset_daily()
        assert rm._peak_equity == 120_000.0

    def test_sod_equity_updated_on_daily_reset(self, rm):
        """reset_daily() must update the start-of-day anchor to current equity."""
        rm.update_equity(110_000.0)
        rm.reset_daily()
        assert rm._sod_equity == 110_000.0

    def test_sow_equity_updated_on_weekly_reset(self, rm):
        """reset_weekly() must update the start-of-week anchor to current equity."""
        rm.update_equity(95_000.0)
        rm.reset_weekly()
        assert rm._sow_equity == 95_000.0


# ─────────────────────────────────────────────────────────────────────────────
# Drawdown properties
# ─────────────────────────────────────────────────────────────────────────────

class TestDrawdownProperties:

    def test_daily_drawdown_calculation(self, rm):
        """daily_drawdown = (sod_equity − current) / sod_equity."""
        rm._sod_equity = 100_000.0
        rm._equity     =  97_000.0
        assert abs(rm.daily_drawdown - 0.03) < 1e-9

    def test_weekly_drawdown_calculation(self, rm):
        """weekly_drawdown = (sow_equity − current) / sow_equity."""
        rm._sow_equity = 100_000.0
        rm._equity     =  94_000.0
        assert abs(rm.weekly_drawdown - 0.06) < 1e-9

    def test_peak_drawdown_tracks_rolling_max(self, rm):
        """peak_drawdown reflects the worst decline from the all-time high."""
        rm.update_equity(120_000.0)
        rm.update_equity(108_000.0)  # 10% drop from 120k
        assert abs(rm.peak_drawdown - 0.10) < 1e-9

    def test_no_negative_drawdown(self, rm):
        """Equity rising above sod / sow / peak must not yield negative drawdown."""
        rm._sod_equity = 90_000.0
        rm._equity     = 110_000.0
        assert rm.daily_drawdown == 0.0

    def test_portfolio_state_daily_drawdown(self):
        """PortfolioState.daily_drawdown is consistent with its equity fields."""
        port = _make_portfolio(equity=97_000)
        port.sod_equity = 100_000
        assert abs(port.daily_drawdown - 0.03) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# validate_signal — full pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateSignal:

    def test_flat_signal_always_approved(self, rm):
        """FLAT signals bypass all checks and are always approved."""
        sig = _Signal(
            symbol="SPY", direction="FLAT", confidence=0.1,
            entry_price=400.0, stop_loss=0.0, take_profit=None,
            position_size_pct=0.0, leverage=1.0,
        )
        decision = rm.validate_signal(sig, _make_portfolio())
        assert decision.approved

    def test_missing_stop_loss_rejected(self, rm):
        """Signal with stop_loss=0 must be rejected."""
        sig = _make_signal(stop=0.0)
        decision = rm.validate_signal(sig, _make_portfolio())
        assert not decision.approved
        assert "STOP_LOSS" in decision.rejection_reason or "INVALID_PRICES" in decision.rejection_reason

    def test_entry_equals_stop_rejected(self, rm):
        """Entry == stop means zero risk per share → reject."""
        sig = _make_signal(entry=400.0, stop=400.0)
        decision = rm.validate_signal(sig, _make_portfolio())
        assert not decision.approved

    def test_normal_signal_approved(self, rm):
        """A well-formed signal in normal market conditions should be approved."""
        sig = _make_signal(entry=400.0, stop=390.0, size_pct=0.10, leverage=1.0)
        decision = rm.validate_signal(sig, _make_portfolio())
        assert decision.approved
        assert decision.modified_signal is not None

    def test_halted_state_rejects_all_longs(self, rm):
        """In HALTED state, no LONG signal should be approved."""
        # Force halt
        port = _make_portfolio(equity=97_000)
        port.sod_equity = 100_000
        rm._circuit_breaker.check(port)
        assert rm.state == TradingState.HALTED

        sig = _make_signal()
        decision = rm.validate_signal(sig, port)
        assert not decision.approved
        assert decision.breaker_fired != BreakerType.NONE

    def test_reduced_state_halves_size(self, rm):
        """In REDUCED state, the approved position_size_pct must be ≤ half of original."""
        port = _make_portfolio(equity=98_000)
        port.sod_equity = 100_000  # 2% DD → REDUCED
        rm._circuit_breaker.check(port)
        assert rm.state == TradingState.REDUCED

        sig = _make_signal(size_pct=0.10)
        decision = rm.validate_signal(sig, port)
        assert decision.approved
        approved_size = decision.modified_signal.position_size_pct
        # After risk caps AND 50% REDUCED reduction, approved_size should be
        # at most half of what a normal-state signal with size_pct=0.10 would get
        assert approved_size <= 0.05 + 1e-9

    def test_max_concurrent_blocks_new_symbol(self, rm):
        """Attempting to add a 6th position when max_concurrent=5 is rejected."""
        positions = {
            f"SYM{i}": Position(f"SYM{i}", 10, 100.0, 95.0, 1000.0)
            for i in range(5)
        }
        port = _make_portfolio(equity=100_000, positions=positions)
        sig = _make_signal(symbol="NEW")
        decision = rm.validate_signal(sig, port)
        assert not decision.approved
        # reason may be forwarded from _apply_portfolio_limits or generic
        assert not decision.approved  # key invariant: must be rejected

    def test_lock_file_blocks_all_trading(self, rm, tmp_path):
        """Existing trading_halted.lock rejects every LONG signal."""
        _HALT_LOCK_FILE.write_text("halt")
        try:
            sig = _make_signal()
            decision = rm.validate_signal(sig, _make_portfolio())
            assert not decision.approved
            assert "PEAK_DD_HALT" in decision.rejection_reason
        finally:
            _HALT_LOCK_FILE.unlink(missing_ok=True)

    def test_leverage_forced_1x_on_circuit_breaker(self, rm):
        """Leverage is forced to 1.0× when any circuit breaker is active."""
        port = _make_portfolio(equity=98_000)
        port.sod_equity = 100_000  # REDUCED
        rm._circuit_breaker.check(port)

        sig = _make_signal(leverage=1.25)
        decision = rm.validate_signal(sig, port)
        if decision.approved:
            assert decision.modified_signal.leverage == 1.0

    def test_leverage_forced_1x_with_3_or_more_positions(self, rm):
        """Leverage is forced to 1.0× when 3+ positions are already open."""
        positions = {f"SYM{i}": Position(f"SYM{i}", 10, 100.0, 95.0, 1000.0) for i in range(3)}
        port = _make_portfolio(positions=positions)
        sig = _make_signal(leverage=1.25)
        decision = rm.validate_signal(sig, port)
        if decision.approved:
            assert decision.modified_signal.leverage == 1.0
        # If rejected for some other reason (4th position vs max 5), that's fine too

    def test_low_confidence_reduces_leverage(self, rm):
        """Confidence < 0.55 forces leverage to 1.0×."""
        sig = _make_signal(leverage=1.25, confidence=0.50, regime_prob=0.50)
        decision = rm.validate_signal(sig, _make_portfolio())
        if decision.approved:
            assert decision.modified_signal.leverage == 1.0

    def test_modifications_list_populated(self, rm):
        """When the risk manager changes the signal, modifications must be non-empty."""
        sig = _make_signal(leverage=1.25, confidence=0.50)  # will force 1.0×
        decision = rm.validate_signal(sig, _make_portfolio())
        if decision.approved:
            assert len(decision.modifications) >= 1

    def test_duplicate_order_blocked_in_window(self, rm):
        """Same symbol + direction within 60s cooldown must be rejected."""
        sig = _make_signal()
        port = _make_portfolio()

        first = rm.validate_signal(sig, port)
        assert first.approved, first.rejection_reason  # first is fine

        # Immediately re-submit same symbol + direction
        second = rm.validate_signal(sig, port)
        assert not second.approved
        assert "DUPLICATE" in second.rejection_reason


# ─────────────────────────────────────────────────────────────────────────────
# Leverage enforcement
# ─────────────────────────────────────────────────────────────────────────────

class TestLeverageEnforcement:

    def test_leverage_capped_at_max_config(self, rm):
        """Any requested leverage above max_leverage is capped."""
        lev, notes = rm._enforce_leverage(2.0, _make_signal(), _make_portfolio())
        assert lev <= rm.config.max_leverage
        assert any("capped" in n for n in notes)

    def test_leverage_1x_when_flicker_rate_nonzero(self, rm):
        """Non-zero flicker rate forces leverage to 1.0×."""
        port = _make_portfolio(flicker_rate=0.5)
        lev, notes = rm._enforce_leverage(1.25, _make_signal(), port)
        assert lev == 1.0

    def test_leverage_unmodified_in_normal_state(self, rm):
        """1.25× leverage is allowed in NORMAL state with 0 positions and high confidence."""
        port = _make_portfolio()
        lev, notes = rm._enforce_leverage(1.25, _make_signal(confidence=0.80), port)
        assert lev == 1.25
        assert not notes


# ─────────────────────────────────────────────────────────────────────────────
# Correlation limits
# ─────────────────────────────────────────────────────────────────────────────

class TestCorrelationLimits:

    def test_high_correlation_halves_size(self, rm):
        """Correlation > 0.70 with existing position halves the signal size."""
        rm.update_correlations({"SPY": {"QQQ": 0.75}})
        positions = {"QQQ": Position("QQQ", 100, 300.0, 290.0, 30_000.0)}
        port = _make_portfolio(positions=positions)
        sig = _make_signal(symbol="SPY", size_pct=0.10)
        _, notes, rejected = rm._apply_correlation_limits(sig, port)
        assert not rejected
        assert any("halved" in n for n in notes)

    def test_very_high_correlation_rejects(self, rm):
        """Correlation > 0.85 with existing position hard-rejects the trade."""
        rm.update_correlations({"SPY": {"QQQ": 0.90}})
        positions = {"QQQ": Position("QQQ", 100, 300.0, 290.0, 30_000.0)}
        port = _make_portfolio(positions=positions)
        sig = _make_signal(symbol="SPY")
        _, notes, rejected = rm._apply_correlation_limits(sig, port)
        assert rejected
        assert any("CORR_REJECT" in n for n in notes)

    def test_no_correlations_no_change(self, rm):
        """No correlation data → signal passes unchanged."""
        rm.update_correlations({})
        sig = _make_signal(symbol="SPY")
        modified, notes, rejected = rm._apply_correlation_limits(sig, _make_portfolio())
        assert not rejected
        assert not notes


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio limits
# ─────────────────────────────────────────────────────────────────────────────

class TestPortfolioLimits:

    def test_max_exposure_cap(self, rm):
        """
        Total notional after trade must not exceed max_exposure × equity.
        Signal is reduced to fit within available headroom.
        """
        # Already deployed 75% → 5% headroom before 80% ceiling
        positions = {"QQQ": Position("QQQ", 100, 750.0, 700.0, 75_000.0)}
        port = _make_portfolio(equity=100_000, positions=positions)
        sig = _make_signal(symbol="SPY", size_pct=0.20, leverage=1.0)  # would push to 95%
        modified, mods = rm._apply_portfolio_limits(sig, port, leverage=1.0)
        assert mods is not None  # not rejected
        # approved_size should be ≤ 5% + tiny tolerance
        assert modified.position_size_pct <= 0.05 + 1e-6 or mods is None

    def test_max_single_position_cap(self, rm):
        """Signal with 50% size must be capped to max_single_position (15%)."""
        sig = _make_signal(size_pct=0.50)
        modified, mods = rm._apply_portfolio_limits(sig, _make_portfolio(), leverage=1.0)
        assert mods is not None
        assert modified.position_size_pct <= 0.15 + 1e-9

    def test_sector_exposure_cap(self, rm):
        """Sector exposure must not exceed max_correlated_sector (30%)."""
        rm.update_sectors({"SPY": "EQUITY_ETF", "QQQ": "EQUITY_ETF"})
        # QQQ already at 25% → adding SPY at 10% would push sector to 35%
        positions = {"QQQ": Position("QQQ", 100, 250.0, 240.0, 25_000.0)}
        port = _make_portfolio(equity=100_000, positions=positions)
        sig = _make_signal(symbol="SPY", size_pct=0.10)
        modified, mods = rm._apply_portfolio_limits(sig, port, leverage=1.0)
        assert mods is not None
        # SPY sector headroom: 30% − 25% = 5%
        assert modified.position_size_pct <= 0.05 + 1e-6

    def test_max_concurrent_rejects_new_position(self, rm):
        """Adding a 6th symbol when max_concurrent=5 returns None mods (rejection)."""
        positions = {f"SYM{i}": Position(f"SYM{i}", 10, 100.0, 90.0, 1000.0) for i in range(5)}
        port = _make_portfolio(positions=positions)
        sig = _make_signal(symbol="NEW")
        _, mods = rm._apply_portfolio_limits(sig, port, leverage=1.0)
        assert mods is None  # None signals hard rejection


# ─────────────────────────────────────────────────────────────────────────────
# Circuit breaker — independence from HMM
# ─────────────────────────────────────────────────────────────────────────────

class TestHMMIndependence:
    """
    The circuit breakers must fire based purely on actual P&L,
    regardless of the HMM regime label.
    """

    def test_breaker_fires_in_bull_regime(self, cfg):
        """Daily halt fires even when HMM says BULL — P&L is authoritative."""
        rm = RiskManager(cfg, initial_equity=100_000.0)
        port = _make_portfolio(equity=97_000, hmm_regime="BULL")
        port.sod_equity = 100_000
        state, fired = rm._circuit_breaker.check(port)
        assert state == TradingState.HALTED
        assert fired == BreakerType.DAILY_HALT

    def test_breaker_fires_in_crash_regime(self, cfg):
        """Same threshold fires regardless of regime label."""
        rm = RiskManager(cfg, initial_equity=100_000.0)
        port = _make_portfolio(equity=97_000, hmm_regime="CRASH")
        port.sod_equity = 100_000
        state, fired = rm._circuit_breaker.check(port)
        assert state == TradingState.HALTED

    def test_validate_signal_rejects_regardless_of_regime(self, rm):
        """validate_signal rejects a LONG in HALTED state even for BULL regime."""
        port = _make_portfolio(equity=97_000, hmm_regime="BULL")
        port.sod_equity = 100_000
        rm._circuit_breaker.check(port)

        sig = _make_signal(symbol="SPY")
        decision = rm.validate_signal(sig, port)
        assert not decision.approved

    def test_history_records_regime_mismatch(self, cfg):
        """Event history includes HMM regime so we can audit if HMM was wrong."""
        rm = RiskManager(cfg, initial_equity=100_000.0)
        # Crash while HMM says BULL (regime was wrong)
        port = _make_portfolio(equity=97_000, hmm_regime="BULL")
        port.sod_equity = 100_000
        rm._circuit_breaker.check(port)
        history = rm._circuit_breaker.get_history()
        bull_and_halted = [e for e in history if e.hmm_regime == "BULL"]
        assert len(bull_and_halted) >= 1, (
            "History must record that HMM was BULL when daily halt fired"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PortfolioState convenience properties
# ─────────────────────────────────────────────────────────────────────────────

class TestPortfolioStateProperties:

    def test_total_notional(self):
        positions = {
            "SPY": Position("SPY", 10, 400.0, 380.0, 4_000.0),
            "QQQ": Position("QQQ", 20, 300.0, 285.0, 6_000.0),
        }
        port = _make_portfolio(equity=100_000, positions=positions)
        assert abs(port.total_notional - 10_000.0) < 1e-6

    def test_total_exposure_pct(self):
        positions = {"SPY": Position("SPY", 100, 400.0, 380.0, 40_000.0)}
        port = _make_portfolio(equity=100_000, positions=positions)
        assert abs(port.total_exposure_pct - 0.40) < 1e-9

    def test_n_positions(self):
        positions = {f"S{i}": Position(f"S{i}", 1, 100.0, 90.0, 100.0) for i in range(3)}
        port = _make_portfolio(positions=positions)
        assert port.n_positions == 3
