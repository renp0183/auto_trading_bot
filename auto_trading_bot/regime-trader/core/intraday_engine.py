"""
intraday_engine.py — Intraday signal generation for the hybrid regime trader.

Architecture role
-----------------
This module is the EXECUTION layer in the hybrid design:

    Daily HMM (1D bars)  →  regime_state per asset  →  regime filter
                                                              ↓
    Intraday bars (5m)   →  IntradayEngine.generate_signal() ↓
                                                              ↓
                                          Signal → RiskManager → OrderExecutor

The daily HMM never generates trade entries directly.  Every order originates
here from intraday price action that is FILTERED and SIZED by the daily regime.

Signal layers (priority order — first match wins)
--------------------------------------------------
    1. INTRADAY_TREND          Core momentum: ADX≥22, MA-aligned, MACD+
    2. INTRADAY_BREAKOUT       Above 20-bar swing high with volume+compression
    3. INTRADAY_MOMENTUM       MACD histogram crossover (neg→pos) + above EMA55
    4. INTRADAY_PULLBACK       Retracement to EMA20/21 zone in uptrend
    5. INTRADAY_EARLY_TREND    Pre-trend: ADX 14–20, MAs converging (< 3%)
    6. INTRADAY_MEAN_REVERSION Oversold bounce: RSI≤40, Stoch≤30, near swing low

Regime size weights (from daily HMM vol_rank)
----------------------------------------------
    vol_rank ≤ 0.33  (LowVol)   → all signals, 1.00× base weight
    0.33 < rank < 0.67 (MidVol) → all signals, 0.75× base weight
    vol_rank ≥ 0.67  (HighVol)  → PULLBACK + MR only, 0.35× base weight

Cooldown
--------
    Per-symbol cooldown of N intraday bars between entries (default 3 bars = 15 min
    on 5m timeframe).  Prevents rapid re-entry after a stop-out.

Market-hours gate
-----------------
    No new entries in the last 15 minutes of the session (3:45 PM ET).
    Positions opened intraday are managed by the existing trailing-stop /
    TP1 / BE infrastructure in main.py — no change needed there.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import timezone
from typing import Optional

import numpy as np
import pandas as pd

# Reuse the Signal dataclass from the daily layer so the signal flows into
# the existing RiskManager → OrderExecutor pipeline unchanged.
from core.regime_strategies import Signal
from core.hmm_engine import RegimeState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Signal type labels (stored in Signal.strategy_name)
SIG_TREND    = "IntradayTrend"
SIG_BREAKOUT = "IntradayBreakout"
SIG_MOMENTUM = "IntradayMomentum"
SIG_PULLBACK = "IntradayPullback"
SIG_EARLY    = "IntradayEarlyTrend"
SIG_MR       = "IntradayMeanReversion"

# ── User-specified execution constants (DO NOT CHANGE) ────────────────────────
PARTIAL_PROFIT_R1        = 1.0    # TP1 = entry + 1×R
PARTIAL_PROFIT_R2        = 3.0    # TP2 = entry + 3×R
PARTIAL_PROFIT_R3        = 4.0    # TP3 = entry + 4×R (conditional on confidence)
PARTIAL_CLOSE_FRAC_1     = 0.30   # close 30% at TP1
PARTIAL_CLOSE_FRAC_2     = 0.40   # close 40% at TP2

TRAIL_ATR_MULTIPLIER     = 3.5    # trailing stop before TP1 hit
TRAIL_ATR_MULTIPLIER_2   = 3.0    # trailing stop after TP1 hit (tighter)

SL_ATR_MULT_TIGHT        = 1.2    # SL multiplier: PULLBACK / MR / EARLY / MOMENTUM
SL_ATR_MULT_WIDE         = 2.0    # SL multiplier: TREND / BREAKOUT

TP3_CONFIDENCE_THRESHOLD = 5.5    # raw confidence (0–10) required for TP3 to exist

BASE_RISK_PCT            = 0.01   # 1% of equity base risk per trade
HARD_MAX_RISK_PCT        = 0.02   # 2% hard cap (fraction of equity)
HARD_MAX_RISK_USD        = 2000.0 # $2,000 hard cap (absolute)

# ── Adaptive regime routing (replaces fixed vol-tier routing) ─────────────────
_ALLOWED_BY_REGIME: dict[str, set[str]] = {
    "TRENDING": {SIG_TREND, SIG_BREAKOUT, SIG_MOMENTUM, SIG_PULLBACK, SIG_EARLY, SIG_MR},
    "NEUTRAL":  {SIG_TREND, SIG_BREAKOUT, SIG_MOMENTUM, SIG_PULLBACK, SIG_EARLY, SIG_MR},
    "CHOPPY":   {SIG_PULLBACK, SIG_MR},
}

REGIME_SIZE_WEIGHT: dict[str, float] = {
    "TRENDING": 1.00,
    "NEUTRAL":  0.65,
    "CHOPPY":   0.35,
}

# Signal-specific size multipliers (unchanged)
_SIG_SIZE: dict[str, float] = {
    SIG_TREND:    1.00,
    SIG_BREAKOUT: 1.00,
    SIG_MOMENTUM: 1.00,
    SIG_PULLBACK: 0.85,
    SIG_EARLY:    0.55,
    SIG_MR:       0.35,
}

# Technical thresholds
_ADX_TRENDING   = 22.0
_ADX_CHOPPY     = 14.0
_MIN_SL_PCT     = 0.004  # 0.4% floor (intraday)
_MAX_SL_PCT     = 0.018  # 1.8% ceiling


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> float:
    """Last value of an EMA series; returns NaN when insufficient data."""
    if len(series) < span // 2:
        return float("nan")
    return float(series.ewm(span=span, adjust=False).mean().iloc[-1])


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window // 2).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> float:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    val = tr.ewm(span=window, adjust=False).mean().iloc[-1]
    return float(val) if not pd.isna(val) else 0.0


def _rsi(close: pd.Series, window: int = 14) -> float:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(span=window, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=window, adjust=False).mean()
    rs    = gain / (loss + 1e-10)
    val   = (100 - 100 / (1 + rs)).iloc[-1]
    return float(val) if not pd.isna(val) else 50.0


def _stoch(high: pd.Series, low: pd.Series, close: pd.Series,
           k_window: int = 14, d_window: int = 3) -> tuple[float, float]:
    lowest  = low.rolling(k_window, min_periods=k_window // 2).min()
    highest = high.rolling(k_window, min_periods=k_window // 2).max()
    k = 100 * (close - lowest) / (highest - lowest + 1e-10)
    d = k.rolling(d_window).mean()
    return (
        float(k.iloc[-1]) if not pd.isna(k.iloc[-1]) else 50.0,
        float(d.iloc[-1]) if not pd.isna(d.iloc[-1]) else 50.0,
    )


def _macd(close: pd.Series,
          fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float, float, float]:
    """Returns (macd_line, signal_line, histogram) at the last bar."""
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    sig_line  = macd_line.ewm(span=signal, adjust=False).mean()
    hist      = macd_line - sig_line
    def _last(s):
        v = s.iloc[-1]
        return float(v) if not pd.isna(v) else 0.0
    return _last(macd_line), _last(sig_line), _last(hist)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> tuple[float, float, float]:
    """Returns (adx, +DI, -DI) at last bar."""
    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_vals   = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_raw = pd.Series(tr_vals.values, index=close.index).ewm(span=window, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm,  index=close.index).ewm(span=window, adjust=False).mean() / (atr_raw + 1e-10)
    mdi = 100 * pd.Series(minus_dm, index=close.index).ewm(span=window, adjust=False).mean() / (atr_raw + 1e-10)
    dx  = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)
    adx_series = dx.ewm(span=window, adjust=False).mean()
    def _last(s):
        v = s.iloc[-1]
        return float(v) if not pd.isna(v) else 0.0
    return _last(adx_series), _last(pdi), _last(mdi)


# ---------------------------------------------------------------------------
# Per-symbol adaptive regime state
# ---------------------------------------------------------------------------

@dataclass
class SymbolState:
    """
    Rolling history for adaptive percentile-based regime classification.

    Keeps the last 100 bars of ATR, absolute 1-bar return, and ADX.
    ``classify()`` returns a (label, vol_pct) tuple used for signal routing
    and SL/TP scaling — no per-asset parameter tuning required.
    """
    _WINDOW: int = field(default=100, init=False, repr=False)

    atr_history:  deque = field(default_factory=lambda: deque(maxlen=100))
    vol_history:  deque = field(default_factory=lambda: deque(maxlen=100))   # abs 1-bar return
    adx_history:  deque = field(default_factory=lambda: deque(maxlen=100))
    regime_score: float = 0.5     # EMA-smoothed score in [0, 1]
    regime_label: str   = "NEUTRAL"  # "TRENDING" | "NEUTRAL" | "CHOPPY"

    def update(self, atr: float, abs_ret: float, adx: float) -> None:
        self.atr_history.append(atr)
        self.vol_history.append(abs_ret)
        self.adx_history.append(adx)

    @staticmethod
    def _percentile(history: deque, current: float) -> float:
        if len(history) < 10:
            return 0.5
        arr = np.array(history)
        return float(np.mean(arr <= current))

    def classify(self, adx: float, atr: float, abs_ret: float) -> tuple[str, float]:
        """
        Returns (regime_label, vol_pct).

        Applies EMA smoothing: regime_score = 0.8×prev + 0.2×current.
        Thresholds: >0.60 → TRENDING, <0.35 → CHOPPY, else NEUTRAL.
        """
        adx_pct = self._percentile(self.adx_history, adx)
        vol_pct = self._percentile(self.vol_history, abs_ret)

        # High ADX + low vol → trending; high vol → choppy
        raw_score = 0.6 * adx_pct + 0.4 * (1.0 - vol_pct)
        self.regime_score = 0.8 * self.regime_score + 0.2 * raw_score

        if self.regime_score > 0.60:
            self.regime_label = "TRENDING"
        elif self.regime_score < 0.35:
            self.regime_label = "CHOPPY"
        else:
            self.regime_label = "NEUTRAL"

        return self.regime_label, vol_pct


# ---------------------------------------------------------------------------
# Standalone SL/TP and position-sizing functions
# ---------------------------------------------------------------------------

def compute_sl_tp(
    signal_type: str,
    entry: float,
    atr: float,
    confidence: float,   # 0–10 raw score
    vol_pct: float,      # 0–1 volatility percentile for SL scaling
) -> tuple[float, float, float, Optional[float], float]:
    """
    Compute stop-loss, TP1, TP2, TP3 (optional), and sl_dist.

    SL distance adapts to volatility percentile:
        vol_scale = 0.8 (low vol) → 1.2 (high vol)

    Returns
    -------
    (stop_loss, tp1, tp2, tp3_or_None, sl_dist)
    """
    tight_signals = {SIG_PULLBACK, SIG_MR, SIG_EARLY, SIG_MOMENTUM}
    atr_mult  = SL_ATR_MULT_TIGHT if signal_type in tight_signals else SL_ATR_MULT_WIDE
    vol_scale = 0.8 + vol_pct * 0.4   # [0.8, 1.2]

    sl_dist = max(
        entry * _MIN_SL_PCT,
        min(entry * _MAX_SL_PCT, atr * atr_mult * vol_scale),
    )

    sl  = entry - sl_dist
    tp1 = entry + sl_dist * PARTIAL_PROFIT_R1
    tp2 = entry + sl_dist * PARTIAL_PROFIT_R2
    tp3 = (entry + sl_dist * PARTIAL_PROFIT_R3) if confidence >= TP3_CONFIDENCE_THRESHOLD else None

    return sl, tp1, tp2, tp3, sl_dist


def _hmm_macro_scale(hmm_label: str) -> float:
    """Map HMM regime label → macro size scale [0.3, 1.0]."""
    bull = {"BULL", "STRONG_BULL", "EUPHORIA", "WEAK_BULL"}
    bear = {"CRASH", "STRONG_BEAR", "BEAR", "WEAK_BEAR"}
    label = hmm_label.upper()
    if label in bull:
        return 1.0
    if label in bear:
        return 0.3
    return 0.7


def compute_size(
    equity: float,
    sl_dist: float,
    entry: float,
    conf_mult: float,         # 0–1 normalised confidence
    hmm_label: str,           # daily HMM label → macro scale
    regime: str,              # adaptive regime: "TRENDING"|"NEUTRAL"|"CHOPPY"
    signal_size_mult: float,  # from _SIG_SIZE[signal_type]
    guard_active: bool = False,
) -> float:
    """
    Compute fractional position size as a fraction of equity (0–1).

    Risk is capped at both HARD_MAX_RISK_PCT × equity and HARD_MAX_RISK_USD.
    """
    _EQUITY_GUARD_SCALE = 0.5

    if sl_dist <= 0 or entry <= 0 or equity <= 0:
        return 0.0

    regime_weight = REGIME_SIZE_WEIGHT.get(regime, 0.35)
    macro_scale   = _hmm_macro_scale(hmm_label)

    risk = equity * BASE_RISK_PCT * conf_mult * macro_scale
    risk *= regime_weight
    risk *= signal_size_mult
    if guard_active:
        risk *= _EQUITY_GUARD_SCALE

    risk = min(risk, min(equity * HARD_MAX_RISK_PCT, HARD_MAX_RISK_USD))

    shares = risk / sl_dist
    return shares * entry / equity   # → pos_pct (0–1)


# ---------------------------------------------------------------------------
# Signal result container
# ---------------------------------------------------------------------------

@dataclass
class _IntradayCandidate:
    """Internal candidate before conversion to Signal."""
    signal_type:  str
    size_mult:    float   # combined tier × signal multiplier
    confidence:   float   # 0–10 score


# ---------------------------------------------------------------------------
# Main engine class
# ---------------------------------------------------------------------------

class IntradayEngine:
    """
    Generates intraday trade signals from short-timeframe OHLCV bars, gated
    and sized by the daily HMM regime.

    Parameters
    ----------
    min_bars : int
        Minimum intraday bars required for indicator warm-up (default 60).
    cooldown_bars : int
        Bars that must elapse after the last entry before a new one is
        considered.  Prevents rapid re-entry (default 3 = 15 min on 5m).
    no_entry_minutes_before_close : int
        Cut-off before market close after which no new entries are opened
        (default 15 minutes).
    """

    def __init__(
        self,
        min_bars: int = 60,
        cooldown_bars: int = 3,
        no_entry_minutes_before_close: int = 15,
    ) -> None:
        self._min_bars    = min_bars
        self._cooldown    = cooldown_bars
        self._no_entry_mins = no_entry_minutes_before_close
        # symbol → bars_since_last_entry
        self._bars_since_entry: dict[str, int] = {}
        # symbol → SymbolState (rolling windows for adaptive regime)
        self._symbol_states: dict[str, SymbolState] = {}

    # ── Public interface ──────────────────────────────────────────────────────

    def tick(self, symbol: str) -> None:
        """Increment the per-symbol bar counter (call on every intraday bar)."""
        self._bars_since_entry[symbol] = self._bars_since_entry.get(symbol, 999) + 1

    def record_entry(self, symbol: str) -> None:
        """Reset the cooldown counter after a successful entry."""
        self._bars_since_entry[symbol] = 0

    def generate_signal(
        self,
        symbol: str,
        bars: pd.DataFrame,
        daily_regime: RegimeState,
        market_close_utc: Optional[pd.Timestamp] = None,
    ) -> Optional[Signal]:
        """
        Produce an intraday trade signal for ``symbol``.

        Parameters
        ----------
        symbol : str
            Ticker.
        bars : DataFrame
            Intraday OHLCV bars (5m or 15m).  Must have columns
            open/high/low/close/volume and a UTC DatetimeIndex.
        daily_regime : RegimeState
            The daily HMM regime state for this symbol.
        market_close_utc : Timestamp, optional
            Today's market close time in UTC.  Used for the no-entry gate.

        Returns
        -------
        Signal or None.
        """
        bars = self._normalise(bars)
        if len(bars) < self._min_bars:
            logger.debug("[%s] IntradayEngine: only %d bars (need %d)", symbol, len(bars), self._min_bars)
            return None

        # ── Market-hours gate ─────────────────────────────────────────────────
        if market_close_utc is not None:
            bar_ts = bars.index[-1]
            if bar_ts.tzinfo is None:
                bar_ts = bar_ts.tz_localize("UTC")
            minutes_to_close = (market_close_utc - bar_ts).total_seconds() / 60.0
            if minutes_to_close < self._no_entry_mins:
                logger.debug("[%s] IntradayEngine: %dm to close — no new entries", symbol, minutes_to_close)
                return None

        # ── Cooldown gate ─────────────────────────────────────────────────────
        bars_elapsed = self._bars_since_entry.get(symbol, 999)
        if bars_elapsed < self._cooldown:
            logger.debug("[%s] IntradayEngine: cooldown (%d/%d bars)", symbol, bars_elapsed, self._cooldown)
            return None

        # ── Compute indicators (all causal — last bar only reads t−1 and prior) ─
        close  = bars["close"]
        high   = bars["high"]
        low    = bars["low"]

        ma20   = float(_sma(close, 20).iloc[-1])
        ma50   = float(_sma(close, 50).iloc[-1])
        ma200  = float(_sma(close, 200).iloc[-1]) if len(bars) >= 100 else float("nan")
        ema9   = _ema(close, 9)
        ema21  = _ema(close, 21)
        ema55  = _ema(close, 55)
        entry  = float(close.iloc[-1])

        atr14  = _atr(high, low, close, 14)
        rsi14  = _rsi(close, 14)
        stoch_k, stoch_d = _stoch(high, low, close)
        _, _, macd_h   = _macd(close)
        _, _, macd_h_prev = _macd(close.iloc[:-1]) if len(close) > 30 else (0.0, 0.0, macd_h)

        adx_val, adx_pos, adx_neg = _adx(high, low, close)

        swing_high20 = float(high.rolling(20).max().iloc[-2]) if len(bars) >= 21 else float("nan")
        swing_low10  = float(low.rolling(10).min().iloc[-2])  if len(bars) >= 11 else float("nan")

        dist_ma20 = (entry - ma20) / ma20 * 100 if ma20 > 0 else 0.0
        dist_ma200 = (entry - ma200) / ma200 * 100 if (ma200 and ma200 > 0) else 0.0
        ma_converging = (
            not pd.isna(ma50) and not pd.isna(ma200)
            and ma50 > ma200
            and abs(ma50 - ma200) / ma200 * 100 < 3.0
        )

        # ── Adaptive regime (per-symbol SymbolState) ──────────────────────────
        abs_ret   = abs(float(close.pct_change().iloc[-1])) if len(close) > 1 else 0.0
        sym_state = self._get_symbol_state(symbol)
        sym_state.update(atr14, abs_ret, adx_val)
        regime, vol_pct = sym_state.classify(adx_val, atr14, abs_ret)
        allowed = _ALLOWED_BY_REGIME[regime]

        # ── Signal selection (priority order) ─────────────────────────────────
        candidate = self._select_signal(
            entry, ma20, ma50, ma200, ema9, ema21, ema55,
            atr14, rsi14, stoch_k, stoch_d,
            macd_h, macd_h_prev,
            adx_val, adx_pos, adx_neg,
            swing_high20, swing_low10,
            dist_ma20, dist_ma200, ma_converging,
            allowed,
        )
        if candidate is None:
            return None

        # ── Confidence gate ───────────────────────────────────────────────────
        if candidate.confidence < 3.0:
            logger.debug("[%s] IntradayEngine: confidence %.1f < 3.0 — skip", symbol, candidate.confidence)
            return None

        # ── Exact SL/TP computation ───────────────────────────────────────────
        stop_loss, tp1, tp2, tp3, sl_dist = compute_sl_tp(
            signal_type = candidate.signal_type,
            entry       = entry,
            atr         = atr14,
            confidence  = candidate.confidence,
            vol_pct     = vol_pct,
        )

        # ── Exact position sizing ─────────────────────────────────────────────
        pos_pct = compute_size(
            equity           = 1.0,            # normalised; risk_manager scales to real equity
            sl_dist          = sl_dist,
            entry            = entry,
            conf_mult        = candidate.confidence / 10.0,
            hmm_label        = daily_regime.label,
            regime           = regime,
            signal_size_mult = _SIG_SIZE[candidate.signal_type],
            guard_active     = False,
        )

        # ── Build Signal (plugs into existing RiskManager unchanged) ──────────
        sig = Signal(
            symbol             = symbol,
            direction          = "LONG",
            confidence         = candidate.confidence / 10.0,  # normalise to 0–1
            entry_price        = entry,
            stop_loss          = stop_loss,
            take_profit        = round(tp1, 4),                # TP1 = primary bracket target
            position_size_pct  = pos_pct,
            leverage           = 1.0,
            regime_id          = daily_regime.state_id,
            regime_name        = daily_regime.label,
            regime_probability = daily_regime.probability,
            timestamp          = bars.index[-1],
            reasoning          = self._build_reasoning(
                candidate.signal_type, regime, daily_regime.label,
                adx_val, rsi14, macd_h, dist_ma20,
            ),
            strategy_name = candidate.signal_type,
            metadata={
                "atr":            atr14,
                "r_value":        sl_dist,
                "intraday":       True,
                "regime":         regime,
                "vol_pct":        round(vol_pct, 3),
                "tp2_level":      round(tp2, 4),
                "tp3_level":      round(tp3, 4) if tp3 else None,
                "sig_size":       candidate.size_mult,
                "confidence_raw": candidate.confidence,
            },
        )
        return sig

    def _get_symbol_state(self, symbol: str) -> SymbolState:
        """Return the SymbolState for a symbol, creating it if missing."""
        if symbol not in self._symbol_states:
            self._symbol_states[symbol] = SymbolState()
        return self._symbol_states[symbol]

    # ── Signal selection logic ────────────────────────────────────────────────

    def _select_signal(
        self,
        entry: float,
        ma20: float, ma50: float, ma200: float,
        ema9: float, ema21: float, ema55: float,
        atr14: float,
        rsi14: float, stoch_k: float, stoch_d: float,
        macd_h: float, macd_h_prev: float,
        adx_val: float, adx_pos: float, adx_neg: float,
        swing_high20: float, swing_low10: float,
        dist_ma20: float, dist_ma200: float,
        ma_converging: bool,
        allowed: set[str],
    ) -> Optional[_IntradayCandidate]:
        """
        First-match wins across 6 layers.  Returns None if no condition met.
        Each layer checks `allowed` before firing.
        """
        # Derived booleans
        above_ma200   = (not pd.isna(ma200)) and entry > ma200 * 0.97
        ma_uptrend    = (not pd.isna(ma50)) and (not pd.isna(ma200)) and ma50 > ma200
        above_ema21   = (not pd.isna(ema21)) and entry > ema21 * 0.995
        above_ema55   = (not pd.isna(ema55)) and entry > ema55
        adx_trending  = adx_val >= _ADX_TRENDING
        adx_building  = _ADX_CHOPPY <= adx_val < _ADX_TRENDING
        adx_dir_bull  = adx_pos > adx_neg
        macd_positive = macd_h > 0
        macd_cross    = macd_h > 0 and macd_h_prev <= 0

        # ── Layer 1: TREND ────────────────────────────────────────────────────
        if SIG_TREND in allowed:
            if (
                ma_uptrend and above_ema21 and adx_trending and adx_dir_bull
                and macd_positive and 38 <= rsi14 <= 78 and dist_ma20 < 12.0
                and above_ma200
            ):
                conf = self._confidence(adx_val, rsi14, macd_h, dist_ma20,
                                        ma_uptrend, above_ma200, base=5.5)
                return _IntradayCandidate(SIG_TREND, _SIG_SIZE[SIG_TREND], conf)

        # ── Layer 2: BREAKOUT ─────────────────────────────────────────────────
        if SIG_BREAKOUT in allowed and not pd.isna(swing_high20) and swing_high20 > 0:
            if (
                entry > swing_high20 and adx_val >= 18.0 and macd_positive
                and rsi14 <= 80 and above_ma200
            ):
                conf = self._confidence(adx_val, rsi14, macd_h, dist_ma20,
                                        ma_uptrend, above_ma200, base=5.0)
                return _IntradayCandidate(SIG_BREAKOUT, _SIG_SIZE[SIG_BREAKOUT], conf)

        # ── Layer 3: MOMENTUM (MACD crossover) ───────────────────────────────
        if SIG_MOMENTUM in allowed:
            if (
                macd_cross and above_ema55 and adx_val >= 16.0
                and ma_uptrend and 42 <= rsi14 <= 72
            ):
                conf = self._confidence(adx_val, rsi14, macd_h, dist_ma20,
                                        ma_uptrend, above_ma200, base=4.5)
                return _IntradayCandidate(SIG_MOMENTUM, _SIG_SIZE[SIG_MOMENTUM], conf)

        # ── Layer 4: PULLBACK ─────────────────────────────────────────────────
        if SIG_PULLBACK in allowed:
            at_ema21 = (not pd.isna(ema21)) and abs(entry - ema21) / ema21 * 100 < 3.5
            at_ma20z = -6.0 <= dist_ma20 <= 5.5
            above_ma50_leeway = (not pd.isna(ma50)) and entry >= ma50 * 0.985
            rsi_reset  = rsi14 <= 68
            stoch_reset = stoch_k <= 70
            trend_ok   = ma_uptrend
            if (
                trend_ok and above_ma50_leeway and (at_ema21 or at_ma20z)
                and rsi_reset and stoch_reset and adx_val >= _ADX_CHOPPY
            ):
                conf = self._confidence(adx_val, rsi14, macd_h, dist_ma20,
                                        ma_uptrend, above_ma200, base=5.0)
                return _IntradayCandidate(SIG_PULLBACK, _SIG_SIZE[SIG_PULLBACK], conf)

        # ── Layer 5: EARLY TREND ──────────────────────────────────────────────
        if SIG_EARLY in allowed:
            if (
                ma_converging and adx_building and entry > (ma50 * 0.99 if not pd.isna(ma50) else entry)
                and 35 <= rsi14 <= 72 and macd_positive and adx_pos > adx_neg * 0.9
            ):
                conf = self._confidence(adx_val, rsi14, macd_h, dist_ma20,
                                        ma_uptrend, above_ma200, base=3.5)
                return _IntradayCandidate(SIG_EARLY, _SIG_SIZE[SIG_EARLY], conf)

        # ── Layer 6: MEAN REVERSION ───────────────────────────────────────────
        if SIG_MR in allowed:
            near_low = (not pd.isna(swing_low10) and swing_low10 > 0
                        and entry <= swing_low10 * 1.02)
            rsi_os   = rsi14 <= 40
            stoch_os = stoch_k <= 30
            macd_imp = macd_h > macd_h_prev   # improving (not necessarily positive)
            not_crash = dist_ma200 > -15.0
            not_far_below = above_ma200
            if (
                rsi_os and stoch_os and macd_imp
                and (near_low or dist_ma20 < -3.0)
                and not_crash and not_far_below
            ):
                conf = self._confidence(adx_val, rsi14, macd_h, dist_ma20,
                                        ma_uptrend, above_ma200, base=3.0)
                return _IntradayCandidate(SIG_MR, _SIG_SIZE[SIG_MR], conf)

        return None

    def _confidence(
        self,
        adx: float, rsi: float, macd_h: float, dist_ma20: float,
        ma_uptrend: bool, above_ma200: bool,
        base: float = 5.0,
    ) -> float:
        """Additive confidence scorer (0–10 scale, base sets starting point)."""
        score = base
        # ADX strength bonus
        if adx >= 35:        score += 1.5
        elif adx >= 28:      score += 1.0
        elif adx >= _ADX_TRENDING: score += 0.5
        # RSI quality
        if 45 <= rsi <= 68:  score += 1.0
        elif 38 <= rsi <= 78: score += 0.5
        elif rsi > 80 or rsi < 28: score -= 0.5
        # MACD
        score += 0.75 if macd_h > 0 else -0.25
        # Distance from MA20 (extended = penalty)
        if dist_ma20 > 8.0:  score -= 1.0
        elif dist_ma20 > 5.0: score -= 0.5
        # Trend alignment bonus
        if ma_uptrend:   score += 0.5
        if above_ma200:  score += 0.5
        return max(0.0, min(10.0, score))

    @staticmethod
    def _normalise(bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        df.columns = [c.lower() for c in df.columns]
        return df

    @staticmethod
    def _build_reasoning(
        sig_type: str, regime: str, regime_label: str,
        adx: float, rsi: float, macd_h: float, dist_ma20: float,
    ) -> str:
        return (
            f"{sig_type} | regime={regime} | dailyRegime={regime_label} | "
            f"ADX={adx:.1f} RSI={rsi:.1f} MACDh={macd_h:.4f} DistMA20={dist_ma20:.1f}%"
        )
