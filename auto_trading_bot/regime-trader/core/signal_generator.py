"""
signal_generator.py — Combines HMM regime state with strategy and risk layers
to produce actionable trade signals.

Responsibilities:
  - Orchestrate HMMEngine → RegimeStrategy → RiskManager pipeline.
  - Produce BUY / SELL / HOLD / REBALANCE signals for each symbol.
  - Attach stop-loss, take-profit, and position-size metadata to each signal.
  - Filter out signals that are blocked by the current TradingState.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd

from core.hmm_engine import HMMEngine, RegimeState
from core.regime_strategies import RegimeStrategy, AllocationTarget
from core.risk_manager import RiskManager, SizingResult, TradingState


class SignalType(str, Enum):
    """Actionable signal types produced by the generator."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    REBALANCE = "REBALANCE"
    BLOCKED = "BLOCKED"       # Risk manager or state machine suppressed this signal


@dataclass
class TradeSignal:
    """Complete trade signal, ready for the order executor."""

    symbol: str
    signal_type: SignalType
    entry_price: float
    stop_price: float
    target_price: Optional[float]         # None when no explicit take-profit
    shares: int
    notional: float
    regime: str                            # Regime label at signal time
    confidence: float                      # HMM posterior confidence
    allocation_fraction: float             # Strategy allocation fraction
    leverage: float                        # Strategy leverage multiplier
    trading_state: TradingState            # Risk manager state at signal time
    blocked_reason: str = ""               # Populated when signal_type == BLOCKED
    metadata: dict = field(default_factory=dict)


class SignalGenerator:
    """
    Combines the HMM engine, regime strategy, and risk manager into a single
    pipeline that produces a list of TradeSignal objects for each bar.

    Usage::

        generator = SignalGenerator(hmm_engine, strategy, risk_manager)
        signals = generator.generate(features_df, prices_df, positions)
    """

    def __init__(
        self,
        hmm_engine: HMMEngine,
        strategy: RegimeStrategy,
        risk_manager: RiskManager,
    ) -> None:
        """
        Parameters
        ----------
        hmm_engine:
            Fitted HMMEngine instance.
        strategy:
            RegimeStrategy instance.
        risk_manager:
            RiskManager instance (carries live equity state).
        """
        ...

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        features: pd.DataFrame,
        prices: pd.DataFrame,
        current_positions: dict[str, float],
    ) -> list[TradeSignal]:
        """
        Run the full pipeline and return signals for all tracked symbols.

        Parameters
        ----------
        features:
            Feature matrix passed to HMMEngine.predict_current().
        prices:
            OHLCV DataFrame with a column per symbol (close prices at minimum).
        current_positions:
            Symbol → current notional value of open positions.

        Returns
        -------
        List of TradeSignal, one per symbol.  Signals may be HOLD or BLOCKED
        when no action is warranted.
        """
        ...

    def get_current_regime(self, features: pd.DataFrame) -> RegimeState:
        """
        Convenience method — return the current RegimeState without generating
        full signals.
        """
        ...

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_stop(
        self, symbol: str, entry_price: float, prices: pd.DataFrame
    ) -> float:
        """
        Derive a stop-loss price using ATR-based volatility sizing.
        Returns entry_price * (1 - atr_multiplier * atr_pct).
        """
        ...

    def _compute_target(
        self, entry_price: float, stop_price: float, reward_ratio: float = 2.0
    ) -> float:
        """Compute take-profit as entry + reward_ratio × (entry - stop)."""
        ...

    def _is_entry_allowed(
        self, symbol: str, current_positions: dict[str, float]
    ) -> tuple[bool, str]:
        """
        Return (allowed, reason_if_blocked) based on TradingState and
        max_concurrent position count.
        """
        ...
