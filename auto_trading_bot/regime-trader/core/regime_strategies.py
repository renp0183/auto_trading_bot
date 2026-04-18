"""
regime_strategies.py — Volatility-regime-based capital allocation strategies.

DESIGN PHILOSOPHY
-----------------
The HMM detects VOLATILITY ENVIRONMENTS, not market direction.
Stocks trend upward roughly 70 % of the time in low-volatility periods.
The worst drawdowns cluster in high-volatility spikes.

The strategy is always LONG — never short.  Shorting was tested extensively
in walk-forward backtesting and consistently destroyed returns:
  1. Markets have long-term upward drift.
  2. V-shaped recoveries happen fast; the HMM is 2-3 days late detecting them.
  3. Short positions during rebounds wipe out crash gains.

The correct response to high volatility is REDUCING ALLOCATION, not
reversing direction.

THREE STRATEGY TIERS (by volatility rank of the current HMM regime):
  ─────────────────────────────────────────────────────────────────────
  LowVolBull       vol_rank ≤ 0.33  95% allocation, 1.25× leverage
  MidVolCautious   0.33 < rank < 0.67  95% if price > EMA50, else 60%
  HighVolDefensive vol_rank ≥ 0.67  60% allocation, 1.0× leverage
  ─────────────────────────────────────────────────────────────────────

Volatility rank is computed INDEPENDENTLY of the return-based label sort.
"BULL" label does NOT mean low vol.  The orchestrator ignores labels.

UNCERTAINTY MODE  (triggers when confidence < threshold OR is_flickering)
  - Halve all position sizes
  - Force leverage to 1.0×
  - Append "[UNCERTAINTY - size halved]" to reasoning

REBALANCING
  - Only rebalance when target weight drifts > 10 % from current weight
  - Prevents churn from minor probability fluctuations
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange

from core.hmm_engine import Regime, RegimeInfo, RegimeState

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    """
    Configuration for all three strategy tiers.

    Maps 1-to-1 to the ``[strategy]`` section of ``settings.yaml``, with
    additional tuning parameters for stops and EMA windows.
    """

    # ── Allocation fractions ──────────────────────────────────────────────────
    low_vol_allocation:          float = 0.95
    mid_vol_allocation_trend:    float = 0.95   # price > EMA50
    mid_vol_allocation_no_trend: float = 0.60   # price < EMA50
    high_vol_allocation:         float = 0.60

    # ── Leverage multipliers ──────────────────────────────────────────────────
    low_vol_leverage:            float = 1.25   # only tier with leverage
    # mid and high vol: always 1.0×

    # ── Rebalancing ───────────────────────────────────────────────────────────
    rebalance_threshold:         float = 0.10   # drift > 10 % triggers rebalance

    # ── Uncertainty mode ──────────────────────────────────────────────────────
    uncertainty_size_mult:       float = 0.50   # fraction applied to position_size_pct
    min_confidence:              float = 0.55   # below this → uncertainty mode

    # ── Stop-loss parameters ─────────────────────────────────────────────────
    atr_window:                  int   = 14
    ema_window:                  int   = 50
    trailing_stop_atr_mult:      float = 3.0    # LowVol: max_price − N×ATR
    low_vol_ema_atr_mult:        float = 0.5    # LowVol: EMA50 − N×ATR
    mid_vol_ema_atr_mult:        float = 0.5    # MidVol: EMA50 − N×ATR
    high_vol_ema_atr_mult:       float = 1.0    # HighVol: EMA50 − N×ATR (wider)
    trailing_stop_lookback:      int   = 20     # bars for rolling-max trailing stop


# ─────────────────────────────────────────────────────────────────────────────
# Signal dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Signal:
    """
    A fully resolved trade signal produced by a strategy tier.

    ``direction`` is always ``"LONG"`` or ``"FLAT"`` — shorting is never used.
    ``position_size_pct`` is the fraction of equity to deploy (0.0 – 0.95).
    The risk manager may further reduce this based on drawdown state.
    """

    symbol:             str
    direction:          Literal["LONG", "FLAT"]
    confidence:         float              # regime posterior probability
    entry_price:        float              # last close (fills at next open in practice)
    stop_loss:          float              # absolute stop price
    take_profit:        Optional[float]    # None → set by risk manager
    position_size_pct:  float              # 0.0 – 0.95 fraction of equity
    leverage:           float              # 1.0 or 1.25
    regime_id:          int
    regime_name:        str
    regime_probability: float
    timestamp:          Optional[pd.Timestamp]
    reasoning:          str
    strategy_name:      str
    metadata:           dict = field(default_factory=dict)

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def is_long(self) -> bool:
        return self.direction == "LONG"

    @property
    def risk_per_share(self) -> float:
        """Absolute dollar risk per share (entry − stop)."""
        return max(0.0, self.entry_price - self.stop_loss)

    def __repr__(self) -> str:
        return (
            f"Signal({self.symbol} {self.direction} "
            f"size={self.position_size_pct:.0%} lev={self.leverage}× "
            f"stop={self.stop_loss:.2f} [{self.strategy_name}] "
            f"regime={self.regime_name} p={self.regime_probability:.3f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Phase-1 backward-compatible allocation target (kept as a thin wrapper)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AllocationTarget:
    """
    Desired portfolio-level allocation target (Phase-1 interface).

    For new code use ``Signal`` directly.
    """

    allocation_fraction: float
    leverage:            float
    regime:              Regime
    confidence:          float
    trend_confirmed:     bool
    notes:               str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Shared price indicator helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ema(close: pd.Series, span: int) -> float:
    """Return the current (last) EMA value."""
    return float(close.ewm(span=span, adjust=False, min_periods=span // 2).mean().iloc[-1])


def _atr(bars: pd.DataFrame, window: int) -> float:
    """Return the current (last) ATR value."""
    atr_series = AverageTrueRange(
        high=bars["high"],
        low=bars["low"],
        close=bars["close"],
        window=window,
        fillna=True,
    ).average_true_range()
    return float(atr_series.iloc[-1])


def _rolling_max(close: pd.Series, lookback: int) -> float:
    """Return the rolling high-water mark over the last ``lookback`` bars."""
    return float(close.rolling(lookback, min_periods=1).max().iloc[-1])


def _is_above_ema(close: pd.Series, span: int) -> bool:
    """Return True if the last close is above its EMA."""
    ema_val = _ema(close, span)
    return float(close.iloc[-1]) > ema_val


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class BaseStrategy(ABC):
    """
    Abstract base class for all regime strategy tiers.

    Subclasses implement ``generate_signal()`` to produce a ``Signal`` for
    a single symbol given its OHLCV bars and the current HMM regime state.
    """

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy tier name."""

    @abstractmethod
    def _base_position_size(self, bars: pd.DataFrame) -> tuple[float, float]:
        """
        Return (position_size_pct, leverage) for normal (non-uncertainty) conditions.
        """

    def generate_signal(
        self,
        symbol: str,
        bars: pd.DataFrame,
        regime_state: RegimeState,
    ) -> Optional[Signal]:
        """
        Produce a trade signal for ``symbol``.

        Parameters
        ----------
        symbol : str
            Ticker being evaluated.
        bars : DataFrame
            OHLCV history.  Must contain columns open, high, low, close, volume.
            Needs at least ``config.ema_window`` rows for a valid EMA.
        regime_state : RegimeState
            Current confirmed regime from HMMEngine.

        Returns
        -------
        Signal or None if bars are insufficient for indicator computation.
        """
        bars = self._normalise_cols(bars)
        if len(bars) < max(self.config.ema_window, self.config.atr_window * 2):
            logger.warning(
                "%s.generate_signal(%s): only %d bars — need ≥ %d; returning None",
                self.name, symbol, len(bars),
                max(self.config.ema_window, self.config.atr_window * 2),
            )
            return None

        close       = bars["close"]
        entry_price = float(close.iloc[-1])
        atr_val     = _atr(bars, self.config.atr_window)
        ema50       = _ema(close, self.config.ema_window)
        stop_loss   = self._compute_stop(bars, entry_price, atr_val, ema50)

        # Guard: stop must be strictly below entry
        stop_loss = min(stop_loss, entry_price * 0.99)

        r_value     = max(0.0, entry_price - stop_loss)
        take_profit = self._compute_take_profit(entry_price, r_value)

        pos_pct, leverage = self._base_position_size(bars)
        reasoning = self._build_reasoning(bars, entry_price, atr_val, ema50, regime_state)

        signal = Signal(
            symbol=symbol,
            direction="LONG",
            confidence=regime_state.probability,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=pos_pct,
            leverage=leverage,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            timestamp=regime_state.timestamp,
            reasoning=reasoning,
            strategy_name=self.name,
            metadata={
                "atr": atr_val,
                "ema50": ema50,
                "r_value": r_value,          # per-share risk; used by BE trigger
                "in_transition": regime_state.in_transition,
                "consecutive_bars": regime_state.consecutive_bars,
            },
        )
        return signal

    def _compute_take_profit(self, entry_price: float, r_value: float) -> Optional[float]:
        """
        Return the take-profit price for TP1 (first partial exit).

        LowVol override: returns None (regime change is the exit signal).
        MidVol / HighVol overrides: returns entry + RR_mult × R.
        """
        return None  # base: no fixed take-profit

    # ── Stop computation (overridden per tier) ────────────────────────────────

    @abstractmethod
    def _compute_stop(
        self,
        bars: pd.DataFrame,
        entry_price: float,
        atr_val: float,
        ema50: float,
    ) -> float:
        """Compute the absolute stop-loss price."""

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _build_reasoning(
        self,
        bars: pd.DataFrame,
        entry_price: float,
        atr_val: float,
        ema50: float,
        regime_state: RegimeState,
    ) -> str:
        above = entry_price > ema50
        return (
            f"{self.name} | regime={regime_state.label} "
            f"(p={regime_state.probability:.3f}) | "
            f"price={'above' if above else 'below'} EMA{self.config.ema_window} | "
            f"ATR={atr_val:.4f}"
        )

    @staticmethod
    def _normalise_cols(bars: pd.DataFrame) -> pd.DataFrame:
        """Lower-case column names."""
        df = bars.copy()
        df.columns = [c.lower() for c in df.columns]
        return df


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1: Low-volatility bull strategy
# ─────────────────────────────────────────────────────────────────────────────

class LowVolBullStrategy(BaseStrategy):
    """
    Deployed when the HMM regime's vol rank ≤ 0.33 (calmest third).

    Calm markets trend upward most of the time.  Apply modest leverage to
    compound returns during benign periods.

    Allocation : 95 % of equity
    Leverage   : 1.25×
    Stop       : tighter of (max_price_20bar − 3×ATR) or (EMA50 − 0.5×ATR)
    """

    @property
    def name(self) -> str:
        return "LowVolBull"

    def _base_position_size(self, bars: pd.DataFrame) -> tuple[float, float]:
        return self.config.low_vol_allocation, self.config.low_vol_leverage

    def _compute_stop(
        self,
        bars: pd.DataFrame,
        entry_price: float,
        atr_val: float,
        ema50: float,
    ) -> float:
        close = bars["close"]
        # Trailing stop from rolling high-water mark
        max_price       = _rolling_max(close, self.config.trailing_stop_lookback)
        trailing_stop   = max_price - self.config.trailing_stop_atr_mult * atr_val
        # EMA-anchored stop
        ema_stop        = ema50 - self.config.low_vol_ema_atr_mult * atr_val
        # Use the tighter (higher) of the two stops
        return max(trailing_stop, ema_stop)

    def _build_reasoning(self, bars, entry_price, atr_val, ema50, regime_state) -> str:
        close     = bars["close"]
        max_price = _rolling_max(close, self.config.trailing_stop_lookback)
        t_stop    = max_price - self.config.trailing_stop_atr_mult * atr_val
        e_stop    = ema50 - self.config.low_vol_ema_atr_mult * atr_val
        chosen    = "trailing" if t_stop >= e_stop else "EMA-anchor"
        return (
            f"{self.name} | regime={regime_state.label} "
            f"(p={regime_state.probability:.3f}) | "
            f"95% alloc, 1.25× leverage | "
            f"stop={max(t_stop, e_stop):.4f} ({chosen}) | "
            f"ATR={atr_val:.4f} EMA50={ema50:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2: Mid-volatility cautious strategy
# ─────────────────────────────────────────────────────────────────────────────

class MidVolCautiousStrategy(BaseStrategy):
    """
    Deployed when the HMM regime's vol rank is in (0.33, 0.67) (middle third).

    In elevated but not extreme volatility, stay invested if the trend is
    intact (price above EMA50) but reduce exposure if it has broken.

    Allocation : 95 % if price > EMA50, else 60 %
    Leverage   : 1.0× (no leverage)
    Stop       : EMA50 − 0.5×ATR
    """

    @property
    def name(self) -> str:
        return "MidVolCautious"

    def _base_position_size(self, bars: pd.DataFrame) -> tuple[float, float]:
        close = self._normalise_cols(bars)["close"]
        trend = _is_above_ema(close, self.config.ema_window)
        allocation = (
            self.config.mid_vol_allocation_trend
            if trend
            else self.config.mid_vol_allocation_no_trend
        )
        return allocation, 1.0

    def _compute_stop(
        self,
        bars: pd.DataFrame,
        entry_price: float,
        atr_val: float,
        ema50: float,
    ) -> float:
        return ema50 - self.config.mid_vol_ema_atr_mult * atr_val

    def _compute_take_profit(self, entry_price: float, r_value: float) -> Optional[float]:
        """TP1 at 2R — close 50% of position at this level, then trail remainder."""
        if r_value <= 0:
            return None
        return round(entry_price + 2.0 * r_value, 4)

    def _build_reasoning(self, bars, entry_price, atr_val, ema50, regime_state) -> str:
        close     = self._normalise_cols(bars)["close"]
        trend     = _is_above_ema(close, self.config.ema_window)
        alloc     = self.config.mid_vol_allocation_trend if trend else self.config.mid_vol_allocation_no_trend
        trend_str = f"price {'>' if trend else '<'} EMA{self.config.ema_window} → {alloc:.0%} alloc"
        return (
            f"{self.name} | regime={regime_state.label} "
            f"(p={regime_state.probability:.3f}) | "
            f"{trend_str}, 1.0× leverage | "
            f"stop=EMA50-{self.config.mid_vol_ema_atr_mult}×ATR={self._compute_stop(bars, entry_price, atr_val, ema50):.4f} | "
            f"ATR={atr_val:.4f} EMA50={ema50:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3: High-volatility defensive strategy
# ─────────────────────────────────────────────────────────────────────────────

class HighVolDefensiveStrategy(BaseStrategy):
    """
    Deployed when the HMM regime's vol rank ≥ 0.67 (most volatile third).

    Stay 60 % invested to capture sharp V-shaped rebounds.  Do NOT short —
    the HMM lag means short positions are always entered after the worst of
    the move and reversed too late.

    Allocation : 60 % of equity
    Leverage   : 1.0× (no leverage)
    Stop       : EMA50 − 1.0×ATR (wider stop for elevated vol)
    """

    @property
    def name(self) -> str:
        return "HighVolDefensive"

    def _base_position_size(self, bars: pd.DataFrame) -> tuple[float, float]:
        return self.config.high_vol_allocation, 1.0

    def _compute_stop(
        self,
        bars: pd.DataFrame,
        entry_price: float,
        atr_val: float,
        ema50: float,
    ) -> float:
        return ema50 - self.config.high_vol_ema_atr_mult * atr_val

    def _compute_take_profit(self, entry_price: float, r_value: float) -> Optional[float]:
        """TP1 at 1.5R — capture fast mean-reversion spike; close 50%, trail rest."""
        if r_value <= 0:
            return None
        return round(entry_price + 1.5 * r_value, 4)

    def _build_reasoning(self, bars, entry_price, atr_val, ema50, regime_state) -> str:
        stop = self._compute_stop(bars, entry_price, atr_val, ema50)
        return (
            f"{self.name} | regime={regime_state.label} "
            f"(p={regime_state.probability:.3f}) | "
            f"60% alloc, 1.0× leverage (defensive, no short) | "
            f"stop=EMA50-{self.config.high_vol_ema_atr_mult}×ATR={stop:.4f} | "
            f"ATR={atr_val:.4f} EMA50={ema50:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Strategy Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class StrategyOrchestrator:
    """
    Routes each HMM regime to the correct strategy tier by volatility rank.

    Volatility rank is computed by sorting ``regime_infos`` on
    ``expected_volatility`` (ascending) and normalising to [0, 1]:

        vol_rank = position / (n_regimes − 1)

    The mapping is INDEPENDENT of the return-based label sort.
    "BULL" does not mean low vol — only the expected_volatility field matters.

    Usage
    -----
    ::

        orchestrator = StrategyOrchestrator(config, hmm_engine.get_regime_infos())
        signals = orchestrator.generate_signals(symbols, bars_dict, regime_state, is_flickering)

        # After HMM retrain:
        orchestrator.update_regime_infos(new_regime_infos)
    """

    # Vol rank boundaries
    LOW_VOL_BOUNDARY  = 0.33
    HIGH_VOL_BOUNDARY = 0.67

    def __init__(
        self,
        config: StrategyConfig,
        regime_infos: dict[str, RegimeInfo],
    ) -> None:
        """
        Parameters
        ----------
        config : StrategyConfig
        regime_infos : dict[str, RegimeInfo]
            Label → RegimeInfo mapping from a fitted HMMEngine.
        """
        self.config = config
        self._low_vol_strat  = LowVolBullStrategy(config)
        self._mid_vol_strat  = MidVolCautiousStrategy(config)
        self._high_vol_strat = HighVolDefensiveStrategy(config)

        # regime_id (int) → vol_rank (float 0…1)
        self._vol_rank_map:     dict[int, float] = {}
        # regime_id (int) → strategy instance
        self._strategy_map:     dict[int, BaseStrategy] = {}

        self.update_regime_infos(regime_infos)

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate_signals(
        self,
        symbols: list[str],
        bars: dict[str, pd.DataFrame],
        regime_state: RegimeState,
        is_flickering: bool = False,
    ) -> list[Signal]:
        """
        Produce one Signal per symbol for the current regime.

        Parameters
        ----------
        symbols : list[str]
            Tickers to generate signals for.
        bars : dict[str, DataFrame]
            Symbol → OHLCV DataFrame.
        regime_state : RegimeState
            Current confirmed HMM state.
        is_flickering : bool
            True if the HMM flicker rate exceeds the threshold (from
            ``HMMEngine.is_flickering()``).  Triggers uncertainty mode.

        Returns
        -------
        list[Signal]  — one entry per symbol in ``symbols``.
        """
        strategy = self._get_strategy(regime_state.state_id)
        uncertainty = self._is_uncertain(regime_state, is_flickering)

        if uncertainty:
            logger.debug(
                "Orchestrator: uncertainty mode (p=%.3f, flickering=%s, transition=%s)",
                regime_state.probability, is_flickering, regime_state.in_transition,
            )

        signals: list[Signal] = []
        for sym in symbols:
            sym_bars = bars.get(sym)
            if sym_bars is None or len(sym_bars) == 0:
                logger.warning("generate_signals: no bars for %s, skipping", sym)
                continue

            sig = strategy.generate_signal(sym, sym_bars, regime_state)
            if sig is None:
                continue

            if uncertainty:
                sig = self._apply_uncertainty(sig)

            signals.append(sig)
            logger.debug(sig)

        return signals

    def update_regime_infos(self, regime_infos: dict[str, RegimeInfo]) -> None:
        """
        Rebuild the vol-rank and strategy maps from a new set of RegimeInfo objects.

        Call this after every HMM retrain so the orchestrator reflects the
        new model's volatility ordering.

        Parameters
        ----------
        regime_infos : dict[str, RegimeInfo]
            Label → RegimeInfo mapping (from HMMEngine._regime_infos).
        """
        if not regime_infos:
            logger.warning("update_regime_infos: empty regime_infos, maps unchanged")
            return

        self._vol_rank_map  = self._compute_vol_ranks(regime_infos)
        self._strategy_map  = self._assign_strategies(self._vol_rank_map)

        logger.info(
            "StrategyOrchestrator: vol-rank map updated  %s",
            {
                info.regime_name: f"{self._vol_rank_map.get(info.regime_id, -1):.2f}"
                for info in regime_infos.values()
            },
        )

    def needs_rebalance(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> bool:
        """
        Return True if any symbol has drifted > ``rebalance_threshold`` from
        its target weight.  Prevents churn from small probability fluctuations.

        Parameters
        ----------
        current_weights : dict[str, float]
            Symbol → current portfolio weight (fraction of equity).
        target_weights : dict[str, float]
            Symbol → desired portfolio weight.
        """
        all_syms = set(current_weights) | set(target_weights)
        for sym in all_syms:
            cur = current_weights.get(sym, 0.0)
            tgt = target_weights.get(sym, 0.0)
            if abs(cur - tgt) > self.config.rebalance_threshold:
                logger.debug(
                    "Rebalance triggered: %s current=%.3f target=%.3f drift=%.3f",
                    sym, cur, tgt, abs(cur - tgt),
                )
                return True
        return False

    def get_vol_rank(self, regime_id: int) -> float:
        """Return the vol rank (0.0–1.0) for a given HMM state index."""
        return self._vol_rank_map.get(regime_id, 0.5)

    def get_strategy_for_regime(self, regime_id: int) -> BaseStrategy:
        """Return the strategy tier assigned to a specific HMM state index."""
        return self._strategy_map.get(regime_id, self._mid_vol_strat)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_strategy(self, regime_id: int) -> BaseStrategy:
        return self._strategy_map.get(regime_id, self._mid_vol_strat)

    @staticmethod
    def _compute_vol_ranks(regime_infos: dict[str, RegimeInfo]) -> dict[int, float]:
        """
        Sort regimes by expected_volatility (ascending) and compute vol ranks.

            vol_rank_i = i / (n_regimes − 1)   for i = 0, 1, …, n−1

        A model with a single state gets vol_rank = 0.5 (mid-vol by default).
        """
        infos = list(regime_infos.values())
        infos.sort(key=lambda r: r.expected_volatility)
        n = len(infos)
        if n == 1:
            return {infos[0].regime_id: 0.5}
        return {r.regime_id: i / (n - 1) for i, r in enumerate(infos)}

    def _assign_strategies(
        self, vol_rank_map: dict[int, float]
    ) -> dict[int, BaseStrategy]:
        """Map each regime_id to a strategy tier based on its vol rank."""
        mapping: dict[int, BaseStrategy] = {}
        for regime_id, rank in vol_rank_map.items():
            if rank <= self.LOW_VOL_BOUNDARY:
                mapping[regime_id] = self._low_vol_strat
            elif rank >= self.HIGH_VOL_BOUNDARY:
                mapping[regime_id] = self._high_vol_strat
            else:
                mapping[regime_id] = self._mid_vol_strat
        return mapping

    def _is_uncertain(self, regime_state: RegimeState, is_flickering: bool) -> bool:
        """
        Return True when uncertainty mode should be activated.

        Triggers on any of:
          - HMM posterior probability below min_confidence
          - Regime is mid-transition (not yet confirmed)
          - Flicker rate exceeds threshold
        """
        return (
            regime_state.probability < self.config.min_confidence
            or regime_state.in_transition
            or is_flickering
        )

    def _apply_uncertainty(self, signal: Signal) -> Signal:
        """
        Halve position size and force 1.0× leverage.  Append rationale note.
        """
        from dataclasses import replace
        return replace(
            signal,
            position_size_pct=signal.position_size_pct * self.config.uncertainty_size_mult,
            leverage=1.0,
            reasoning=signal.reasoning + " [UNCERTAINTY - size halved]",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Phase-1 backward-compatible RegimeStrategy wrapper
# ─────────────────────────────────────────────────────────────────────────────

class RegimeStrategy:
    """
    Phase-1 compatibility wrapper.

    Delegates to the appropriate strategy tier and wraps the result in the
    Phase-1 ``AllocationTarget`` dataclass.  New code should use
    ``StrategyOrchestrator`` and ``Signal`` directly.
    """

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        # Minimal orchestrator with empty regime_infos (updated on first use)
        self._orchestrator = StrategyOrchestrator(config, {})

    def get_target_allocation(
        self,
        regime_state: RegimeState,
        price_series: pd.Series,
        min_confidence: float = 0.55,
    ) -> AllocationTarget:
        """
        Return an AllocationTarget for the given regime state.

        Constructs a minimal bars DataFrame from ``price_series`` (close only)
        to drive the strategy, then wraps the result.
        """
        bars = pd.DataFrame(
            {
                "open":   price_series,
                "high":   price_series,
                "low":    price_series,
                "close":  price_series,
                "volume": pd.Series(1.0, index=price_series.index),
            }
        )
        sym = "GENERIC"
        strategy = self._orchestrator.get_strategy_for_regime(regime_state.state_id)
        sig = strategy.generate_signal(sym, bars, regime_state)

        if sig is None:
            # Fallback: use mid-vol defaults
            return AllocationTarget(
                allocation_fraction=self.config.mid_vol_allocation_no_trend,
                leverage=1.0,
                regime=Regime(regime_state.label) if regime_state.label in Regime._value2member_map_ else Regime.UNKNOWN,
                confidence=regime_state.probability,
                trend_confirmed=False,
                notes="Insufficient bars for indicator computation",
            )

        uncertain = self._orchestrator._is_uncertain(regime_state, regime_state.flicker_rate > 0.2)
        if uncertain:
            sig = self._orchestrator._apply_uncertainty(sig)

        trend_confirmed = _is_above_ema(price_series, self.config.ema_window)
        try:
            regime_enum = Regime(regime_state.label)
        except ValueError:
            regime_enum = Regime.UNKNOWN

        return AllocationTarget(
            allocation_fraction=sig.position_size_pct,
            leverage=sig.leverage,
            regime=regime_enum,
            confidence=regime_state.probability,
            trend_confirmed=trend_confirmed,
            notes=sig.reasoning,
        )

    def needs_rebalance(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> bool:
        """Delegate to orchestrator."""
        return self._orchestrator.needs_rebalance(current_weights, target_weights)


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible class aliases
# ─────────────────────────────────────────────────────────────────────────────

#: Alias — CRASH and BEAR regimes both route to the defensive tier
CrashDefensiveStrategy = HighVolDefensiveStrategy
BearTrendStrategy      = HighVolDefensiveStrategy

#: Alias — mean-reversion logic lives in the mid-vol tier (trend-conditional)
MeanReversionStrategy  = MidVolCautiousStrategy

#: Alias — calm bull regime uses the low-vol tier
BullTrendStrategy      = LowVolBullStrategy

#: Alias — euphoria is calm in vol terms (late-cycle but low realised vol)
EuphoriaCautiousStrategy = LowVolBullStrategy

#: Alias — weak bear still routes to defensive
WeakBearStrategy       = HighVolDefensiveStrategy

#: Alias — neutral sits in the mid-vol tier
NeutralStrategy        = MidVolCautiousStrategy

#: Alias — weak bull still uses mid-vol caution
WeakBullStrategy       = MidVolCautiousStrategy

#: Alias — strong bull uses low-vol aggression
StrongBullStrategy     = LowVolBullStrategy

#: Alias — strong bear → defensive
StrongBearStrategy     = HighVolDefensiveStrategy

# ─────────────────────────────────────────────────────────────────────────────
# Label-to-strategy convenience dict
# Provides a sensible default mapping when vol_rank data is unavailable.
# The orchestrator's vol-rank based mapping supersedes this at runtime.
# ─────────────────────────────────────────────────────────────────────────────

LABEL_TO_STRATEGY: dict[str, type[BaseStrategy]] = {
    # High-vol / defensive labels
    "CRASH":        HighVolDefensiveStrategy,
    "STRONG_BEAR":  HighVolDefensiveStrategy,
    "BEAR":         HighVolDefensiveStrategy,
    # Mid-vol / cautious labels
    "WEAK_BEAR":    MidVolCautiousStrategy,
    "NEUTRAL":      MidVolCautiousStrategy,
    "WEAK_BULL":    MidVolCautiousStrategy,
    # Low-vol / bull labels
    "BULL":         LowVolBullStrategy,
    "STRONG_BULL":  LowVolBullStrategy,
    "EUPHORIA":     LowVolBullStrategy,
    # Fallback
    "UNKNOWN":      MidVolCautiousStrategy,
}
