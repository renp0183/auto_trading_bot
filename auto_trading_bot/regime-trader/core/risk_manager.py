"""
risk_manager.py — Position sizing, leverage enforcement, and circuit breakers.

DESIGN PHILOSOPHY
-----------------
The risk manager operates INDEPENDENTLY of the HMM.  It holds absolute veto
power over every signal.  Even if the HMM fails completely, the circuit breakers
catch drawdowns based on actual P&L.  This is the last line of defence — it must
never assume the regime is correct.

CIRCUIT BREAKER HIERARCHY (independent of HMM regime)
------------------------------------------------------
1. Daily DD > 2%   → REDUCED  (all sizes halved for rest of day)
2. Daily DD > 3%   → HALTED   (no new entries rest of day; close all)
3. Weekly DD > 5%  → REDUCED  (all sizes halved for rest of week)
4. Weekly DD > 7%  → HALTED   (no new entries rest of week; close all)
5. Peak DD > 10%   → HALTED   (full halt; writes trading_halted.lock file)

POSITION SIZING
---------------
    max_by_risk = (equity × max_risk_per_trade) / |entry − stop|
    max_by_alloc = equity × allocation_fraction × leverage / price
    shares = min(max_by_risk, max_by_alloc)

GAP RISK
--------
Overnight positions assume a 3× gap-through on the stop.  Final overnight
size is the smaller of the normal size and the size where a 3× gap eats
at most 2% of portfolio.

    gap_risk_per_share = 3 × |entry − stop|
    max_gap_shares = (equity × 0.02) / gap_risk_per_share
    shares = min(shares, max_gap_shares)

LEVERAGE RULES
--------------
Default 1.0×.  Only low-vol regimes may use 1.25×.  Forced back to 1.0× when:
- Any circuit breaker is active (REDUCED or HALTED state)
- 3+ positions already open
- Signal is in uncertainty mode (confidence < threshold, transitioning, or flickering)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Path where the peak-DD halt writes its lock file
_HALT_LOCK_FILE = Path("trading_halted.lock")


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class TradingState(str, Enum):
    """Operational state of the risk manager."""
    NORMAL  = "NORMAL"
    REDUCED = "REDUCED"   # All new position sizes halved
    HALTED  = "HALTED"    # No new entries; pending close-all triggered


class BreakerType(str, Enum):
    """Which circuit breaker fired."""
    NONE            = "NONE"
    DAILY_REDUCE    = "DAILY_REDUCE"
    DAILY_HALT      = "DAILY_HALT"
    WEEKLY_REDUCE   = "WEEKLY_REDUCE"
    WEEKLY_HALT     = "WEEKLY_HALT"
    PEAK_DD_HALT    = "PEAK_DD_HALT"
    DAILY_TRADES    = "DAILY_TRADES"


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskConfig:
    """Configuration mirroring the [risk] section of settings.yaml."""

    # Per-trade
    max_risk_per_trade:  float = 0.01   # Max fraction of equity risked per trade
    max_single_position: float = 0.15   # Max position as fraction of equity

    # Portfolio
    max_exposure:        float = 0.80   # Max total notional / equity
    max_leverage:        float = 1.25   # Hard cap on gross leverage
    max_concurrent:      int   = 5      # Max simultaneous open positions
    max_correlated_sector: float = 0.30 # Max sector exposure (fraction of equity)

    # Activity
    max_daily_trades:    int   = 20

    # Circuit breakers
    daily_dd_reduce:     float = 0.02
    daily_dd_halt:       float = 0.03
    weekly_dd_reduce:    float = 0.05
    weekly_dd_halt:      float = 0.07
    max_dd_from_peak:    float = 0.10

    # Order validation
    max_spread_pct:      float = 0.005  # Reject if bid-ask spread > 0.5%
    duplicate_window_s:  int   = 60     # Block same symbol+direction within N seconds

    # Overnight gap risk
    overnight_gap_mult:  float = 3.0    # Assume stop can gap N× in overnight move
    overnight_gap_risk:  float = 0.02   # Max fraction of equity lost to overnight gap

    # Minimum notional per position
    min_notional:        float = 100.0

    # Per-vol-tier cap (prevents two 3× ETFs from entering simultaneously)
    max_positions_per_vol_tier: int = 2


# ─────────────────────────────────────────────────────────────────────────────
# Core dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Position:
    """Single open position in the portfolio."""
    symbol:       str
    shares:       int
    entry_price:  float
    stop_loss:    float
    notional:     float          # shares × current_price
    sector:       str = "OTHER"  # For correlated-exposure check


@dataclass
class PortfolioState:
    """
    Snapshot of current portfolio used by validate_signal() and size_position().

    The caller is responsible for keeping this up to date after every fill.
    """
    equity:         float
    cash:           float
    buying_power:   float
    positions:      dict[str, Position] = field(default_factory=dict)  # symbol → Position

    # P&L accumulators (reset by reset_daily / reset_weekly)
    daily_pnl:      float = 0.0
    weekly_pnl:     float = 0.0

    # Drawdown tracking
    peak_equity:    float = 0.0           # All-time equity peak
    sod_equity:     float = 0.0           # Start-of-day equity
    sow_equity:     float = 0.0           # Start-of-week equity

    # Activity
    daily_trades:   int   = 0
    circuit_breaker_status: BreakerType = BreakerType.NONE

    # HMM context (for logging only)
    flicker_rate:   float = 0.0
    hmm_regime:     str   = "UNKNOWN"

    @property
    def daily_drawdown(self) -> float:
        """Fraction lost since start of day (positive number)."""
        if self.sod_equity <= 0:
            return 0.0
        return max(0.0, (self.sod_equity - self.equity) / self.sod_equity)

    @property
    def weekly_drawdown(self) -> float:
        """Fraction lost since start of week (positive number)."""
        if self.sow_equity <= 0:
            return 0.0
        return max(0.0, (self.sow_equity - self.equity) / self.sow_equity)

    @property
    def peak_drawdown(self) -> float:
        """Fraction lost from rolling equity peak (positive number)."""
        peak = max(self.peak_equity, self.equity)
        if peak <= 0:
            return 0.0
        return max(0.0, (peak - self.equity) / peak)

    @property
    def total_notional(self) -> float:
        """Sum of all position notionals."""
        return sum(p.notional for p in self.positions.values())

    @property
    def total_exposure_pct(self) -> float:
        """Total notional as fraction of equity."""
        if self.equity <= 0:
            return 0.0
        return self.total_notional / self.equity

    @property
    def n_positions(self) -> int:
        return len(self.positions)


@dataclass
class RiskDecision:
    """
    Result of validate_signal().

    If ``approved`` is True, ``modified_signal`` holds the (possibly modified)
    signal the caller should submit.  If False, ``rejection_reason`` explains why.
    ``modifications`` lists every change the risk manager made to the original signal.
    """
    approved:          bool
    modified_signal:   Optional[object]   # Signal (type alias avoids circular import)
    rejection_reason:  str = ""
    modifications:     list[str] = field(default_factory=list)
    breaker_fired:     BreakerType = BreakerType.NONE


@dataclass
class SizingResult:
    """Output of the position sizing calculation for a single trade."""
    symbol:           str
    shares:           int
    notional:         float
    risk_amount:      float
    pct_of_equity:    float
    rejected:         bool  = False
    rejection_reason: str   = ""


@dataclass
class CircuitBreakerEvent:
    """Structured log entry for a circuit-breaker trigger."""
    timestamp:        pd.Timestamp
    breaker_type:     BreakerType
    actual_dd:        float
    equity:           float
    positions_closed: list[str]
    hmm_regime:       str
    notes:            str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Circuit Breaker tracker
# ─────────────────────────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    Stateful circuit breaker that monitors P&L and updates TradingState.

    Designed to be owned by RiskManager; callers interact with RiskManager
    rather than this class directly.
    """

    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self._state:       TradingState = TradingState.NORMAL
        self._active_type: BreakerType  = BreakerType.NONE
        self._history:     list[CircuitBreakerEvent] = []

        # Separate daily / weekly / peak-halt flags so they don't cancel each other
        self._daily_halted:   bool = False
        self._weekly_halted:  bool = False
        self._peak_halted:    bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def check(self, portfolio: PortfolioState) -> tuple[TradingState, BreakerType]:
        """
        Evaluate all circuit breakers against the current portfolio state.

        Returns (new_state, breaker_type_that_fired).
        Fires at most one new breaker per call (the most severe one wins).
        """
        fired = BreakerType.NONE
        new_state = TradingState.NORMAL

        # ── Peak DD (hardest: requires manual lock file deletion) ─────────────
        if portfolio.peak_drawdown >= self.config.max_dd_from_peak:
            if not self._peak_halted:
                fired = BreakerType.PEAK_DD_HALT
                self._peak_halted = True
                self._write_lock_file(portfolio)
                self._log_event(
                    BreakerType.PEAK_DD_HALT,
                    portfolio.peak_drawdown,
                    portfolio,
                    list(portfolio.positions.keys()),
                )
            new_state = TradingState.HALTED

        # ── Daily halt ────────────────────────────────────────────────────────
        elif portfolio.daily_drawdown >= self.config.daily_dd_halt or self._daily_halted:
            if not self._daily_halted and portfolio.daily_drawdown >= self.config.daily_dd_halt:
                fired = BreakerType.DAILY_HALT
                self._daily_halted = True
                self._log_event(
                    BreakerType.DAILY_HALT,
                    portfolio.daily_drawdown,
                    portfolio,
                    list(portfolio.positions.keys()),
                )
            new_state = TradingState.HALTED

        # ── Weekly halt ───────────────────────────────────────────────────────
        elif portfolio.weekly_drawdown >= self.config.weekly_dd_halt or self._weekly_halted:
            if not self._weekly_halted and portfolio.weekly_drawdown >= self.config.weekly_dd_halt:
                fired = BreakerType.WEEKLY_HALT
                self._weekly_halted = True
                self._log_event(
                    BreakerType.WEEKLY_HALT,
                    portfolio.weekly_drawdown,
                    portfolio,
                    list(portfolio.positions.keys()),
                )
            new_state = TradingState.HALTED

        # ── Weekly reduce ─────────────────────────────────────────────────────
        elif portfolio.weekly_drawdown >= self.config.weekly_dd_reduce:
            if fired == BreakerType.NONE:
                fired = BreakerType.WEEKLY_REDUCE
            new_state = TradingState.REDUCED

        # ── Daily reduce ──────────────────────────────────────────────────────
        elif portfolio.daily_drawdown >= self.config.daily_dd_reduce:
            if fired == BreakerType.NONE:
                fired = BreakerType.DAILY_REDUCE
            new_state = TradingState.REDUCED

        # ── Daily trade limit ─────────────────────────────────────────────────
        if portfolio.daily_trades >= self.config.max_daily_trades:
            if fired == BreakerType.NONE:
                fired = BreakerType.DAILY_TRADES
            new_state = TradingState.HALTED

        self._state = new_state
        if fired != BreakerType.NONE:
            self._active_type = fired

        return new_state, fired

    def update_pnl(self, portfolio: PortfolioState) -> tuple[TradingState, BreakerType]:
        """Alias for check() — named update_pnl for caller clarity."""
        return self.check(portfolio)

    def reset_daily(self) -> None:
        """
        Reset daily flags at market open.

        Weekly and peak-halt flags are intentionally NOT reset here.
        """
        self._daily_halted = False
        if not self._weekly_halted and not self._peak_halted:
            if self._state == TradingState.HALTED:
                self._state = TradingState.NORMAL
                self._active_type = BreakerType.NONE
            elif self._state == TradingState.REDUCED and \
                    self._active_type in (BreakerType.DAILY_REDUCE, BreakerType.DAILY_HALT):
                self._state = TradingState.NORMAL
                self._active_type = BreakerType.NONE

    def reset_weekly(self) -> None:
        """
        Reset weekly flags at start of trading week.

        Peak-halt flag is NOT reset here — requires manual lock file deletion.
        """
        self._weekly_halted = False
        self._daily_halted  = False
        if not self._peak_halted:
            self._state = TradingState.NORMAL
            self._active_type = BreakerType.NONE

    def get_history(self) -> list[CircuitBreakerEvent]:
        """Return the full list of circuit-breaker trigger events."""
        return list(self._history)

    @property
    def state(self) -> TradingState:
        return self._state

    @property
    def active_type(self) -> BreakerType:
        return self._active_type

    @property
    def is_halted(self) -> bool:
        return self._state == TradingState.HALTED

    @property
    def is_reduced(self) -> bool:
        return self._state == TradingState.REDUCED

    @property
    def peak_halted(self) -> bool:
        """True if the peak-DD halt is active (requires manual intervention)."""
        return self._peak_halted

    # ── Private ───────────────────────────────────────────────────────────────

    def _log_event(
        self,
        breaker_type: BreakerType,
        actual_dd: float,
        portfolio: PortfolioState,
        positions_closed: list[str],
    ) -> None:
        event = CircuitBreakerEvent(
            timestamp=pd.Timestamp.now(),
            breaker_type=breaker_type,
            actual_dd=actual_dd,
            equity=portfolio.equity,
            positions_closed=positions_closed,
            hmm_regime=portfolio.hmm_regime,
            notes=f"daily_dd={portfolio.daily_drawdown:.3%} "
                  f"weekly_dd={portfolio.weekly_drawdown:.3%} "
                  f"peak_dd={portfolio.peak_drawdown:.3%}",
        )
        self._history.append(event)
        logger.warning(
            "CIRCUIT BREAKER [%s] fired | actual_dd=%.3f%% equity=%.2f "
            "positions=%s hmm_regime=%s",
            breaker_type.value,
            actual_dd * 100,
            portfolio.equity,
            positions_closed,
            portfolio.hmm_regime,
        )

    @staticmethod
    def _write_lock_file(portfolio: PortfolioState) -> None:
        """Write trading_halted.lock file; manual deletion required to resume."""
        try:
            _HALT_LOCK_FILE.write_text(
                f"TRADING HALTED — PEAK DRAWDOWN EXCEEDED\n"
                f"Triggered: {pd.Timestamp.now().isoformat()}\n"
                f"Equity at halt: {portfolio.equity:.2f}\n"
                f"Peak drawdown: {portfolio.peak_drawdown:.3%}\n"
                f"HMM regime at halt: {portfolio.hmm_regime}\n"
                f"Open positions: {list(portfolio.positions.keys())}\n\n"
                f"Delete this file manually after reviewing the situation.\n"
            )
            logger.critical(
                "trading_halted.lock written — MANUAL DELETION REQUIRED to resume trading."
            )
        except OSError as exc:
            logger.error("Could not write lock file: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Main RiskManager
# ─────────────────────────────────────────────────────────────────────────────

class RiskManager:
    """
    Enforces all position-sizing and drawdown guardrails.

    The risk manager is the LAST gate before an order is submitted.  It has
    ABSOLUTE VETO POWER — the strategy layer's signals are suggestions only.

    Workflow::

        rm = RiskManager(config, initial_equity=100_000)
        decision = rm.validate_signal(signal, portfolio_state)
        if decision.approved:
            submit(decision.modified_signal)

    State management::

        rm.update_equity(new_equity)   # after every fill
        rm.reset_daily()               # at market open each day
        rm.reset_weekly()              # at start of each trading week
    """

    def __init__(self, config: RiskConfig, initial_equity: float = 100_000.0) -> None:
        self.config = config
        self._equity       = initial_equity
        self._peak_equity  = initial_equity
        self._sod_equity   = initial_equity   # start-of-day
        self._sow_equity   = initial_equity   # start-of-week
        self._daily_pnl    = 0.0
        self._weekly_pnl   = 0.0
        self._daily_trades = 0

        self._circuit_breaker = CircuitBreaker(config)

        # Duplicate-order guard: symbol+direction → last submission date (YYYY-MM-DD)
        # Daily bars arrive once per session; a 60s monotonic window is meaningless.
        self._last_order_dates: dict[tuple[str, str], str] = {}

        # 60-day rolling correlations: symbol → {other_sym: corr}
        # Populated externally via update_correlations()
        self._correlations: dict[str, dict[str, float]] = {}

        # Sector mappings: symbol → sector string
        # Populated externally via update_sectors()
        self._sectors: dict[str, str] = {}

        # Full portfolio state reference (updated by caller)
        self._portfolio: Optional[PortfolioState] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Primary API: validate_signal
    # ─────────────────────────────────────────────────────────────────────────

    def validate_signal(
        self,
        signal: object,    # Signal from regime_strategies — avoid circular import
        portfolio: PortfolioState,
    ) -> RiskDecision:
        """
        Run the full risk-validation pipeline on a signal.

        Checks (in order):
          1. Lock file (peak-DD halt requires manual reset)
          2. Circuit breaker state
          3. Order validation (buying power, duplicate, spread)
          4. Stop-loss present
          5. Leverage rules
          6. Position size (risk-based + allocation cap + gap risk)
          7. Portfolio-level limits (exposure, concurrent, sector)
          8. Correlation check

        Returns a RiskDecision.  If approved, ``modified_signal`` may have
        a reduced ``position_size_pct`` and/or leverage compared to the original.

        Parameters
        ----------
        signal : Signal
            Trade signal from StrategyOrchestrator.
        portfolio : PortfolioState
            Current portfolio snapshot.  Updated in-place by the caller after fills.
        """
        from dataclasses import replace as _replace  # avoid top-level circular dep

        self._portfolio = portfolio

        modifications: list[str] = []

        # ── Gate 0: FLAT signals always pass (no position to validate) ────────
        if getattr(signal, "direction", None) == "FLAT":
            return RiskDecision(approved=True, modified_signal=signal)

        sym = signal.symbol

        # ── Gate 1: Peak-DD lock file ─────────────────────────────────────────
        if _HALT_LOCK_FILE.exists():
            reason = (
                f"[PEAK_DD_HALT] trading_halted.lock file exists — "
                f"manual deletion required to resume. "
                f"peak_dd={portfolio.peak_drawdown:.3%}"
            )
            logger.error("REJECT %s: %s", sym, reason)
            return RiskDecision(
                approved=False,
                modified_signal=None,
                rejection_reason=reason,
                breaker_fired=BreakerType.PEAK_DD_HALT,
            )

        # ── Gate 2: Circuit breaker state ─────────────────────────────────────
        state, fired = self._circuit_breaker.check(portfolio)

        if state == TradingState.HALTED:
            reason = (
                f"[{self._circuit_breaker.active_type.value}] trading halted — "
                f"daily_dd={portfolio.daily_drawdown:.3%} "
                f"weekly_dd={portfolio.weekly_drawdown:.3%} "
                f"peak_dd={portfolio.peak_drawdown:.3%}"
            )
            logger.warning("REJECT %s: %s", sym, reason)
            return RiskDecision(
                approved=False,
                modified_signal=None,
                rejection_reason=reason,
                breaker_fired=self._circuit_breaker.active_type,
            )

        # ── Gate 3: Stop-loss mandatory ───────────────────────────────────────
        entry = signal.entry_price
        stop  = signal.stop_loss

        if stop is None or stop <= 0:
            reason = f"[NO_STOP_LOSS] signal for {sym} has no stop loss — order blocked"
            logger.error("REJECT %s: %s", sym, reason)
            return RiskDecision(approved=False, modified_signal=None, rejection_reason=reason)

        risk_per_share = abs(entry - stop)
        if risk_per_share < 1e-8:
            reason = f"[ZERO_RISK] entry ≈ stop for {sym} (entry={entry:.4f} stop={stop:.4f})"
            logger.error("REJECT %s: %s", sym, reason)
            return RiskDecision(approved=False, modified_signal=None, rejection_reason=reason)

        # ── Gate 4: Buying power ──────────────────────────────────────────────
        target_notional = portfolio.equity * signal.position_size_pct * signal.leverage
        if target_notional > portfolio.buying_power * 1.01:  # 1% tolerance
            # Scale down to available buying power
            max_pct = portfolio.buying_power / (portfolio.equity * signal.leverage)
            if max_pct < 0.001:
                reason = (
                    f"[BUYING_POWER] insufficient buying power: "
                    f"need {target_notional:.0f} have {portfolio.buying_power:.0f}"
                )
                logger.warning("REJECT %s: %s", sym, reason)
                return RiskDecision(approved=False, modified_signal=None, rejection_reason=reason)
            modifications.append(
                f"size reduced {signal.position_size_pct:.0%}→{max_pct:.0%} (buying power)"
            )
            signal = _replace(signal, position_size_pct=max_pct)

        # ── Gate 5: Duplicate-order guard (same-date, not 60s window) ─────────
        # Daily bars arrive once per session — intraday cooldown is meaningless.
        # Block the same symbol+direction if it already fired today.
        # Exception: intraday signals carry metadata["intraday"]=True and manage
        # their own per-day budget via _intraday_trades_today in main.py.
        _is_intraday = bool(getattr(signal, "metadata", {}).get("intraday", False))
        key = (sym, signal.direction)
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        last_date = self._last_order_dates.get(key, "")
        if last_date == today and not _is_intraday:
            reason = (
                f"[DUPLICATE] {sym} {signal.direction} already submitted today "
                f"({today}) — one entry per symbol per session"
            )
            logger.warning("REJECT %s: %s", sym, reason)
            return RiskDecision(approved=False, modified_signal=None, rejection_reason=reason)

        # ── Gate 6: Leverage enforcement ──────────────────────────────────────
        leverage = signal.leverage
        leverage, lev_notes = self._enforce_leverage(leverage, signal, portfolio)
        if lev_notes:
            modifications.extend(lev_notes)
            signal = _replace(signal, leverage=leverage)

        # ── Gate 7: Risk-based position sizing ────────────────────────────────
        sizing = self.size_position(
            symbol=sym,
            entry_price=entry,
            stop_price=stop,
            allocation_fraction=signal.position_size_pct,
            leverage=leverage,
            portfolio=portfolio,
        )

        if sizing.rejected:
            logger.warning("REJECT %s: %s", sym, sizing.rejection_reason)
            return RiskDecision(
                approved=False,
                modified_signal=None,
                rejection_reason=sizing.rejection_reason,
            )

        # Translate shares back to a position_size_pct for the modified signal
        actual_pct = sizing.pct_of_equity
        if abs(actual_pct - signal.position_size_pct) > 0.001:
            modifications.append(
                f"size {signal.position_size_pct:.1%}→{actual_pct:.1%} (risk/gap/portfolio caps)"
            )
        signal = _replace(signal, position_size_pct=actual_pct)

        # ── Gate 8: Portfolio-level hard limits ───────────────────────────────
        signal, port_mods = self._apply_portfolio_limits(signal, portfolio, leverage)
        if port_mods is None:
            # Hard rejection — specific reason already logged inside _apply_portfolio_limits
            reason = "[PORTFOLIO_LIMIT] trade rejected by portfolio constraints"
            return RiskDecision(approved=False, modified_signal=None, rejection_reason=reason)
        modifications.extend(port_mods)

        # ── Gate 9: Correlation check ─────────────────────────────────────────
        signal, corr_mods, corr_rejected = self._apply_correlation_limits(signal, portfolio)
        if corr_rejected:
            reason = corr_mods[0] if corr_mods else "correlation limit exceeded"
            logger.warning("REJECT %s: %s", sym, reason)
            return RiskDecision(approved=False, modified_signal=None, rejection_reason=reason)
        modifications.extend(corr_mods)

        # ── Gate 9.5: Risk-reward minimum (only when TP is set) ──────────────
        tp = getattr(signal, "take_profit", None)
        if tp is not None and tp > 0 and entry > 0 and stop > 0:
            rr = (tp - entry) / max(entry - stop, 1e-8)
            if rr < 1.0:
                reason = (
                    f"[RR_REJECT] {sym} RR={rr:.2f} < 1.0 "
                    f"(entry={entry:.4f} stop={stop:.4f} tp={tp:.4f})"
                )
                logger.warning("REJECT %s: %s", sym, reason)
                return RiskDecision(approved=False, modified_signal=None, rejection_reason=reason)

        # ── Gate 10: Circuit breaker size reduction ───────────────────────────
        if state == TradingState.REDUCED:
            orig_pct = signal.position_size_pct
            signal = _replace(signal, position_size_pct=orig_pct * 0.5, leverage=1.0)
            modifications.append(
                f"REDUCED state: size halved {orig_pct:.1%}→{signal.position_size_pct:.1%}, "
                f"leverage forced 1.0×"
            )

        # ── Final minimum notional check ──────────────────────────────────────
        final_notional = portfolio.equity * signal.position_size_pct * signal.leverage
        if final_notional < self.config.min_notional:
            reason = (
                f"[MIN_NOTIONAL] final position {final_notional:.2f} < "
                f"minimum {self.config.min_notional:.2f}"
            )
            logger.warning("REJECT %s: %s", sym, reason)
            return RiskDecision(approved=False, modified_signal=None, rejection_reason=reason)

        # ── Approved ─────────────────────────────────────────────────────────
        self._last_order_dates[key] = today

        if modifications:
            logger.info(
                "APPROVED %s [modified]: %s",
                sym,
                " | ".join(modifications),
            )
        else:
            logger.info("APPROVED %s [unmodified]", sym)

        return RiskDecision(
            approved=True,
            modified_signal=signal,
            modifications=modifications,
            breaker_fired=fired,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Position sizing
    # ─────────────────────────────────────────────────────────────────────────

    def size_position(
        self,
        symbol:              str,
        entry_price:         float,
        stop_price:          float,
        allocation_fraction: float,
        leverage:            float = 1.0,
        portfolio:           Optional[PortfolioState] = None,
    ) -> SizingResult:
        """
        Compute share count using the tightest of three constraints:

          1. Risk constraint:  shares = (equity × max_risk) / |entry − stop|
          2. Allocation cap:   shares = equity × alloc × leverage / entry
          3. Overnight gap:    shares = (equity × overnight_gap_risk) / (gap_mult × |entry − stop|)

        Then applies portfolio max-single-position cap (15% equity).

        Parameters
        ----------
        symbol :
            Ticker being sized.
        entry_price :
            Expected fill price.
        stop_price :
            Hard stop-loss price.
        allocation_fraction :
            Strategy-requested fraction of equity (0.0–0.95).
        leverage :
            Effective leverage after enforcement.
        portfolio :
            Current portfolio snapshot.  If None, uses last known equity.
        """
        equity = portfolio.equity if portfolio is not None else self._equity

        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share < 1e-8 or entry_price <= 0:
            return SizingResult(
                symbol=symbol, shares=0, notional=0.0, risk_amount=0.0,
                pct_of_equity=0.0, rejected=True,
                rejection_reason=f"[INVALID_PRICES] entry={entry_price} stop={stop_price}",
            )

        # ── Constraint 1: risk-based ──────────────────────────────────────────
        max_risk_dollars = equity * self.config.max_risk_per_trade
        shares_by_risk = int(max_risk_dollars / risk_per_share)

        # ── Constraint 2: allocation-based ───────────────────────────────────
        allocation_dollars = equity * allocation_fraction * leverage
        shares_by_alloc = int(allocation_dollars / entry_price)

        # ── Constraint 3: overnight gap risk ─────────────────────────────────
        gap_risk_per_share = self.config.overnight_gap_mult * risk_per_share
        max_gap_dollars = equity * self.config.overnight_gap_risk
        shares_by_gap = int(max_gap_dollars / gap_risk_per_share) if gap_risk_per_share > 0 else shares_by_alloc

        # Tightest constraint wins
        shares = min(shares_by_risk, shares_by_alloc, shares_by_gap)

        # ── Cap at max_single_position ────────────────────────────────────────
        max_position_dollars = equity * self.config.max_single_position
        shares_by_max_pos = int(max_position_dollars / entry_price)
        shares = min(shares, shares_by_max_pos)

        if shares <= 0:
            return SizingResult(
                symbol=symbol, shares=0, notional=0.0, risk_amount=0.0,
                pct_of_equity=0.0, rejected=True,
                rejection_reason=(
                    f"[ZERO_SHARES] risk={shares_by_risk} alloc={shares_by_alloc} "
                    f"gap={shares_by_gap} max_pos={shares_by_max_pos}"
                ),
            )

        notional    = shares * entry_price
        risk_amount = shares * risk_per_share
        pct_equity  = notional / equity if equity > 0 else 0.0

        return SizingResult(
            symbol=symbol,
            shares=shares,
            notional=notional,
            risk_amount=risk_amount,
            pct_of_equity=pct_equity,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Equity / state updates
    # ─────────────────────────────────────────────────────────────────────────

    def update_equity(self, new_equity: float, portfolio: Optional[PortfolioState] = None) -> TradingState:
        """
        Record latest equity, update drawdown trackers, fire circuit breakers.

        Call after every fill or at each end-of-bar equity mark.

        Parameters
        ----------
        new_equity :
            Current portfolio equity.
        portfolio :
            If supplied, updates its peak_equity / sod_equity fields and runs
            the circuit breaker against it.  If None, only internal state updated.
        """
        self._equity = new_equity
        self._peak_equity = max(self._peak_equity, new_equity)

        if portfolio is not None:
            portfolio.equity = new_equity
            portfolio.peak_equity = max(portfolio.peak_equity, new_equity)
            if portfolio.sod_equity <= 0:
                portfolio.sod_equity = new_equity
            if portfolio.sow_equity <= 0:
                portfolio.sow_equity = new_equity

            state, _ = self._circuit_breaker.check(portfolio)
            portfolio.circuit_breaker_status = self._circuit_breaker.active_type
            return state

        return self._circuit_breaker.state

    def reset_daily(self, portfolio: Optional[PortfolioState] = None) -> None:
        """
        Reset daily accumulators.  Call at market open each day.

        Updates sod_equity to current equity so daily DD calculation is correct.
        """
        self._daily_pnl    = 0.0
        self._daily_trades = 0
        self._sod_equity   = self._equity

        if portfolio is not None:
            portfolio.daily_pnl    = 0.0
            portfolio.daily_trades = 0
            portfolio.sod_equity   = portfolio.equity

        self._circuit_breaker.reset_daily()
        logger.debug("Daily risk counters reset (sod_equity=%.2f)", self._equity)

    def reset_weekly(self, portfolio: Optional[PortfolioState] = None) -> None:
        """
        Reset weekly accumulators.  Call at start of each trading week.

        Does NOT reset peak-DD halt — that requires manual lock file deletion.
        """
        self._weekly_pnl  = 0.0
        self._sow_equity  = self._equity

        if portfolio is not None:
            portfolio.weekly_pnl  = 0.0
            portfolio.sow_equity  = portfolio.equity

        self._circuit_breaker.reset_weekly()
        logger.debug("Weekly risk counters reset (sow_equity=%.2f)", self._equity)

    def register_fill(self, portfolio: Optional[PortfolioState] = None) -> None:
        """Increment the daily trade counter after a confirmed fill."""
        self._daily_trades += 1
        if portfolio is not None:
            portfolio.daily_trades += 1

    # ─────────────────────────────────────────────────────────────────────────
    # External data feeds
    # ─────────────────────────────────────────────────────────────────────────

    def update_correlations(self, correlations: dict[str, dict[str, float]]) -> None:
        """
        Supply latest 60-day rolling correlations for all tracked symbols.

        Parameters
        ----------
        correlations :
            {symbol: {other_symbol: correlation_coefficient}}
        """
        self._correlations = correlations

    def update_sectors(self, sectors: dict[str, str]) -> None:
        """
        Supply sector mappings for correlated-exposure tracking.

        Parameters
        ----------
        sectors :
            {symbol: sector_string}  e.g. {"AAPL": "TECH", "GOOGL": "TECH"}
        """
        self._sectors = sectors

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def state(self) -> TradingState:
        """Current operational state (NORMAL / REDUCED / HALTED)."""
        return self._circuit_breaker.state

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Direct access to the CircuitBreaker for history queries."""
        return self._circuit_breaker

    @property
    def equity(self) -> float:
        """Last recorded portfolio equity."""
        return self._equity

    @property
    def daily_drawdown(self) -> float:
        """Drawdown from start-of-day equity (positive fraction)."""
        if self._sod_equity <= 0:
            return 0.0
        return max(0.0, (self._sod_equity - self._equity) / self._sod_equity)

    @property
    def weekly_drawdown(self) -> float:
        """Drawdown from start-of-week equity (positive fraction)."""
        if self._sow_equity <= 0:
            return 0.0
        return max(0.0, (self._sow_equity - self._equity) / self._sow_equity)

    @property
    def peak_drawdown(self) -> float:
        """Drawdown from rolling equity peak (positive fraction)."""
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, (self._peak_equity - self._equity) / self._peak_equity)

    @property
    def is_halted(self) -> bool:
        """True if any circuit breaker has halted trading."""
        return self._circuit_breaker.is_halted

    @property
    def is_reduced(self) -> bool:
        """True if a REDUCED circuit breaker is active."""
        return self._circuit_breaker.is_reduced

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _enforce_leverage(
        self,
        requested_leverage: float,
        signal: object,
        portfolio: PortfolioState,
    ) -> tuple[float, list[str]]:
        """
        Apply leverage rules.  Returns (effective_leverage, list_of_notes).

        Leverage is forced to 1.0 when:
        - Any circuit breaker is active
        - 3+ positions already open
        - Signal is in uncertainty mode (low confidence / transitioning / flickering)
        """
        leverage = min(requested_leverage, self.config.max_leverage)
        notes: list[str] = []

        if leverage != requested_leverage:
            notes.append(
                f"leverage capped {requested_leverage:.2f}x→{leverage:.2f}x (config max)"
            )
            requested_leverage = leverage

        force_1x_reasons: list[str] = []

        if self._circuit_breaker.state != TradingState.NORMAL:
            force_1x_reasons.append(
                f"circuit_breaker={self._circuit_breaker.active_type.value}"
            )

        if portfolio.n_positions >= 3:
            force_1x_reasons.append(
                f"n_positions={portfolio.n_positions} >= 3"
            )

        if portfolio.flicker_rate > 0:
            force_1x_reasons.append(f"flicker_rate={portfolio.flicker_rate:.2f}")

        # Uncertainty mode: low confidence or in-transition
        conf = getattr(signal, "confidence", 1.0)
        regime_prob = getattr(signal, "regime_probability", 1.0)
        effective_conf = min(conf, regime_prob)
        if effective_conf < 0.55:
            force_1x_reasons.append(f"confidence={effective_conf:.3f} < 0.55")

        if force_1x_reasons and leverage > 1.0:
            notes.append(
                f"leverage {leverage:.2f}x→1.0x (force_1x: {'; '.join(force_1x_reasons)})"
            )
            leverage = 1.0

        return leverage, notes

    def _apply_portfolio_limits(
        self,
        signal: object,
        portfolio: PortfolioState,
        leverage: float,
    ) -> tuple[object, Optional[list[str]]]:
        """
        Apply portfolio-level hard and soft limits.

        Returns (possibly_modified_signal, modifications_list).
        Returns (signal, None) when a hard limit rejects the trade.
        """
        from dataclasses import replace as _replace

        modifications: list[str] = []
        sym = signal.symbol
        equity = portfolio.equity

        # ── Hard: max concurrent positions ───────────────────────────────────
        if sym not in portfolio.positions and portfolio.n_positions >= self.config.max_concurrent:
            logger.warning(
                "REJECT %s: [MAX_CONCURRENT] %d positions open (max %d)",
                sym, portfolio.n_positions, self.config.max_concurrent,
            )
            return signal, None  # None signals rejection

        # ── Hard: max single position ─────────────────────────────────────────
        if signal.position_size_pct > self.config.max_single_position:
            old_pct = signal.position_size_pct
            signal = _replace(signal, position_size_pct=self.config.max_single_position)
            modifications.append(
                f"size {old_pct:.1%}→{self.config.max_single_position:.1%} (max_single_position)"
            )

        # ── Soft: total exposure cap ──────────────────────────────────────────
        proposed_notional = equity * signal.position_size_pct * leverage
        existing_notional = portfolio.total_notional
        # Current position in this symbol (replacing it, not adding)
        existing_sym_notional = portfolio.positions[sym].notional if sym in portfolio.positions else 0.0
        new_total_notional = existing_notional - existing_sym_notional + proposed_notional

        max_notional = equity * self.config.max_exposure
        if new_total_notional > max_notional:
            available = max(0.0, max_notional - (existing_notional - existing_sym_notional))
            capped_pct = (available / (equity * leverage)) if equity * leverage > 0 else 0.0
            if capped_pct < 0.01:
                logger.warning(
                    "REJECT %s: [MAX_EXPOSURE] total exposure %.1f%% would exceed "
                    "max %.1f%% with no room",
                    sym,
                    new_total_notional / equity * 100,
                    self.config.max_exposure * 100,
                )
                return signal, None
            old_pct = signal.position_size_pct
            signal = _replace(signal, position_size_pct=capped_pct)
            modifications.append(
                f"size {old_pct:.1%}→{capped_pct:.1%} (max_exposure cap)"
            )

        # ── Soft: sector exposure cap ─────────────────────────────────────────
        sector = self._sectors.get(sym, "OTHER")
        sector_notional = sum(
            p.notional for s, p in portfolio.positions.items()
            if self._sectors.get(s, "OTHER") == sector and s != sym
        )
        proposed_sym_notional = equity * signal.position_size_pct * leverage
        new_sector_total = sector_notional + proposed_sym_notional
        max_sector_notional = equity * self.config.max_correlated_sector

        if new_sector_total > max_sector_notional:
            available_sector = max(0.0, max_sector_notional - sector_notional)
            capped_sector_pct = available_sector / (equity * leverage) if equity * leverage > 0 else 0.0
            if capped_sector_pct < signal.position_size_pct:
                old_pct = signal.position_size_pct
                signal = _replace(signal, position_size_pct=max(0.0, capped_sector_pct))
                modifications.append(
                    f"size {old_pct:.1%}→{capped_sector_pct:.1%} "
                    f"(sector '{sector}' cap {self.config.max_correlated_sector:.0%})"
                )

        return signal, modifications

    def _apply_correlation_limits(
        self,
        signal: object,
        portfolio: PortfolioState,
    ) -> tuple[object, list[str], bool]:
        """
        Apply rolling-correlation constraints.

        Returns (possibly_modified_signal, notes, rejected).
        ``rejected=True`` means correlation > 0.85 → hard reject.
        """
        from dataclasses import replace as _replace

        sym = signal.symbol
        sym_corrs = self._correlations.get(sym, {})
        notes: list[str] = []

        if not sym_corrs or not portfolio.positions:
            return signal, notes, False

        # Find highest correlation with any open position
        max_corr = 0.0
        max_corr_sym = ""
        for open_sym in portfolio.positions:
            corr = abs(sym_corrs.get(open_sym, 0.0))
            if corr > max_corr:
                max_corr = corr
                max_corr_sym = open_sym

        # Hard reject: correlation > 0.85
        if max_corr > 0.85:
            reason = (
                f"[CORR_REJECT] {sym} corr={max_corr:.3f} with {max_corr_sym} > 0.85 "
                f"— too correlated to existing position"
            )
            return signal, [reason], True

        # Soft: correlation > 0.70 → halve size
        if max_corr > 0.70:
            old_pct = signal.position_size_pct
            new_pct = old_pct * 0.5
            signal = _replace(signal, position_size_pct=new_pct)
            notes.append(
                f"size halved {old_pct:.1%}→{new_pct:.1%} "
                f"(corr={max_corr:.3f} with {max_corr_sym} > 0.70)"
            )

        return signal, notes, False

    # ─────────────────────────────────────────────────────────────────────────
    # Legacy compatibility (used by backtester stub)
    # ─────────────────────────────────────────────────────────────────────────

    def check_portfolio_limits(
        self,
        proposed: SizingResult,
        current_positions: dict[str, float],
        leverage: float,
    ) -> SizingResult:
        """
        Legacy API — apply portfolio constraints to a SizingResult.

        Retained for back-compatibility with the backtester skeleton.
        New code should use validate_signal() instead.
        """
        equity = self._equity
        total_notional = sum(current_positions.values())

        # Max concurrent
        if proposed.symbol not in current_positions and \
                len(current_positions) >= self.config.max_concurrent:
            proposed.rejected = True
            proposed.rejection_reason = (
                f"[MAX_CONCURRENT] {len(current_positions)} positions open"
            )
            return proposed

        # Max single position
        max_notional = equity * self.config.max_single_position
        if proposed.notional > max_notional:
            scale = max_notional / max(proposed.notional, 1e-9)
            proposed.shares   = int(proposed.shares * scale)
            proposed.notional = proposed.shares * (proposed.notional / max(proposed.shares / scale, 1))
            proposed.pct_of_equity = proposed.notional / equity if equity > 0 else 0.0

        # Max total exposure
        new_total = total_notional + proposed.notional
        max_total = equity * self.config.max_exposure
        if new_total > max_total:
            available = max(0.0, max_total - total_notional)
            price_est = proposed.notional / max(proposed.shares, 1)
            proposed.shares   = int(available / price_est) if price_est > 0 else 0
            proposed.notional = proposed.shares * price_est
            proposed.pct_of_equity = proposed.notional / equity if equity > 0 else 0.0
            if proposed.shares <= 0:
                proposed.rejected = True
                proposed.rejection_reason = (
                    f"[MAX_EXPOSURE] total exposure {new_total:.0f} > max {max_total:.0f}"
                )

        return proposed

    def _evaluate_trading_state(self) -> TradingState:
        """Re-evaluate state from current internal drawdowns (no portfolio needed)."""
        if self.peak_drawdown >= self.config.max_dd_from_peak:
            return TradingState.HALTED
        if self.daily_drawdown >= self.config.daily_dd_halt:
            return TradingState.HALTED
        if self.weekly_drawdown >= self.config.weekly_dd_halt:
            return TradingState.HALTED
        if self.daily_drawdown >= self.config.daily_dd_reduce:
            return TradingState.REDUCED
        if self.weekly_drawdown >= self.config.weekly_dd_reduce:
            return TradingState.REDUCED
        return TradingState.NORMAL

    def _apply_size_reduction(self, shares: int) -> int:
        """Halve share count when in REDUCED state."""
        if self._circuit_breaker.state == TradingState.REDUCED:
            return max(0, shares // 2)
        return shares
