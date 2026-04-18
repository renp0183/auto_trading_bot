"""
backtester.py — Walk-forward allocation backtester.

Design: Allocation-based walk-forward backtester. NO individual trade stops.
Sets target portfolio allocation each bar based on HMM vol-regime, rebalances
when allocation changes >10%.

Walk-forward windows: train=252 bars, test=126 bars, step=126 bars.

Per-fold algorithm:
  1. Fit HMMEngine on IS (train) features
  2. Build StrategyOrchestrator from engine._regime_infos
  3. Warm up forward algo: call engine.predict_filtered_next() for each bar
     in train_features (to get initial alpha state)
  4. Walk OOS bar by bar:
     - Call engine.predict_filtered_next(obs, timestamp) → regime_state
     - Execute PENDING orders from previous bar (1-bar fill delay)
     - Get signals from orchestrator.generate_signals()
     - Compute target allocation per symbol
     - If drift > rebalance_threshold: queue as pending for next bar
     - Record equity = cash + shares*price
  5. Return equity_series, trades_df, fold_metrics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from core.hmm_engine import HMMEngine, HMMConfig, RegimeState
from core.regime_strategies import StrategyOrchestrator, StrategyConfig, Signal
from core.risk_manager import RiskConfig
from data.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    """Configuration mirroring the [backtest] section of settings.yaml."""

    slippage_pct: float = 0.0005
    initial_capital: float = 100_000.0
    train_window: int = 252
    test_window: int = 126
    step_size: int = 126
    risk_free_rate: float = 0.045
    commission: float = 0.0


@dataclass
class BacktestResult:
    """Aggregated results from a full walk-forward backtest run."""

    equity_curve: pd.Series                # DatetimeIndex → portfolio equity
    trade_ledger: pd.DataFrame             # All simulated trades
    regime_history: pd.DataFrame           # date, regime, confidence, is_confirmed
    fold_results: list[dict]               # Per-fold metrics
    config: BacktestConfig
    symbols: list[str]
    start_date: str
    end_date: str
    summary: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_fold_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    risk_free_rate: float,
) -> dict:
    """Compute basic metrics for a single backtest fold."""
    if len(equity) < 2:
        return {
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "n_trades": 0,
            "win_rate": 0.0,
        }

    returns = equity.pct_change().dropna()
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1.0

    rf_daily = risk_free_rate / 252.0
    excess = returns - rf_daily
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = float(drawdown.min())

    n_trades = len(trades)
    if n_trades > 0 and "realised_pnl" in trades.columns:
        wins = (trades["realised_pnl"] > 0).sum()
        win_rate = float(wins / n_trades)
    else:
        win_rate = 0.0

    return {
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "n_trades": n_trades,
        "win_rate": win_rate,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main backtester
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardBacktester:
    """
    Walk-forward allocation backtester for the regime-based trading strategy.

    Each iteration:
      1. Train HMM on bars [t : t + train_window].
      2. Build StrategyOrchestrator from the fitted engine.
      3. Warm up forward algo on IS bars.
      4. Walk OOS bar by bar — 1-bar fill delay, allocation-based sizing.
      5. Step forward by step_size bars.

    Usage::

        backtester = WalkForwardBacktester(config, hmm_cfg, strategy_cfg, risk_cfg)
        result = backtester.run(ohlcv_dict, feature_df)
        print(result.equity_curve.tail())
    """

    def __init__(
        self,
        config: BacktestConfig,
        hmm_config: HMMConfig,
        strategy_config: StrategyConfig,
        risk_config: RiskConfig,
    ) -> None:
        self.config = config
        self.hmm_config = hmm_config
        self.strategy_config = strategy_config
        self.risk_config = risk_config

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        ohlcv: dict[str, pd.DataFrame],
        features: pd.DataFrame,
    ) -> BacktestResult:
        """
        Execute the full walk-forward backtest.

        Parameters
        ----------
        ohlcv:
            Dict of symbol → OHLCV DataFrame.  All DataFrames must share a
            common DatetimeIndex aligned with features.
        features:
            Pre-computed feature matrix from FeatureEngineer.build_features().
            Rows already have NaN dropped; index is a DatetimeIndex.

        Returns
        -------
        BacktestResult with equity curve, trade ledger, regime history, fold metrics.
        """
        try:
            from hmmlearn import hmm as _hmm_check  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "hmmlearn is required for backtesting. "
                "Install it: pip install hmmlearn  (requires Python ≤ 3.12 or C++ build tools)"
            )

        train_w = self.config.train_window
        test_w = self.config.test_window
        step = self.config.step_size

        if len(features) < train_w + test_w:
            raise ValueError(
                f"Not enough data: need ≥ {train_w + test_w} bars, "
                f"got {len(features)}."
            )

        symbols = list(ohlcv.keys())
        all_equity: list[pd.Series] = []
        all_trades: list[pd.DataFrame] = []
        all_regime_rows: list[dict] = []
        fold_results: list[dict] = []

        # Walk-forward folds
        fold_idx = 0
        start_pos = 0
        equity_at_fold_start = self.config.initial_capital

        while start_pos + train_w + test_w <= len(features):
            train_end = start_pos + train_w
            test_end = train_end + test_w

            train_features = features.iloc[start_pos:train_end]
            test_features = features.iloc[train_end:test_end]

            # Slice OHLCV for the test window
            test_dates = test_features.index
            test_ohlcv: dict[str, pd.DataFrame] = {}
            for sym, df in ohlcv.items():
                # Select rows whose index falls in test_dates
                mask = df.index.isin(test_dates)
                test_ohlcv[sym] = df.loc[mask].copy()

            logger.info(
                "Fold %d: train=[%s..%s] test=[%s..%s]",
                fold_idx,
                train_features.index[0].date() if len(train_features) > 0 else "?",
                train_features.index[-1].date() if len(train_features) > 0 else "?",
                test_features.index[0].date() if len(test_features) > 0 else "?",
                test_features.index[-1].date() if len(test_features) > 0 else "?",
            )

            try:
                eq_series, trades_df, metrics, regime_rows = self._run_fold(
                    fold_idx=fold_idx,
                    train_features=train_features,
                    test_features=test_features,
                    test_ohlcv=test_ohlcv,
                    equity_at_fold_start=equity_at_fold_start,
                    symbols=symbols,
                )
            except Exception as exc:
                logger.error("Fold %d failed: %s", fold_idx, exc, exc_info=True)
                # Move forward — fill with flat equity
                eq_series = pd.Series(
                    equity_at_fold_start,
                    index=test_features.index,
                )
                trades_df = pd.DataFrame()
                metrics = {"error": str(exc)}
                regime_rows = []

            all_equity.append(eq_series)
            all_trades.append(trades_df)
            all_regime_rows.extend(regime_rows)
            metrics["fold_idx"] = fold_idx
            fold_results.append(metrics)

            # Equity at start of next fold = last value of this fold
            equity_at_fold_start = float(eq_series.iloc[-1]) if len(eq_series) > 0 else equity_at_fold_start

            start_pos += step
            fold_idx += 1

        # Combine
        equity_curve = self._build_equity_curve(all_equity)
        trade_ledger = pd.concat([t for t in all_trades if len(t) > 0], ignore_index=True) if any(len(t) > 0 for t in all_trades) else pd.DataFrame()
        regime_history = pd.DataFrame(all_regime_rows) if all_regime_rows else pd.DataFrame(
            columns=["date", "regime", "confidence", "is_confirmed"]
        )

        start_date = str(features.index[0].date()) if len(features) > 0 else ""
        end_date = str(features.index[-1].date()) if len(features) > 0 else ""

        result = BacktestResult(
            equity_curve=equity_curve,
            trade_ledger=trade_ledger,
            regime_history=regime_history,
            fold_results=fold_results,
            config=self.config,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            summary={},
        )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Private: fold execution
    # ─────────────────────────────────────────────────────────────────────────

    def _run_fold(
        self,
        fold_idx: int,
        train_features: pd.DataFrame,
        test_features: pd.DataFrame,
        test_ohlcv: dict[str, pd.DataFrame],
        equity_at_fold_start: float,
        symbols: list[str],
    ) -> tuple[pd.Series, pd.DataFrame, dict, list[dict]]:
        """
        Run a single train/test fold.

        Returns (equity_series, trades_df, fold_metrics_dict, regime_rows).
        """
        # ── Step 1: Fit HMM on training data ─────────────────────────────────
        engine = HMMEngine(self.hmm_config)
        engine.fit(train_features)

        # ── Step 2: Build StrategyOrchestrator ───────────────────────────────
        orchestrator = StrategyOrchestrator(
            config=self.strategy_config,
            regime_infos=engine._regime_infos,
        )

        # ── Step 3: Warm up forward algorithm on IS data ─────────────────────
        engine.reset_live_state()
        X_train = train_features.values.astype(float)
        train_timestamps = train_features.index if isinstance(train_features.index, pd.DatetimeIndex) else None
        for t in range(len(X_train)):
            ts = train_timestamps[t] if train_timestamps is not None else None
            engine.predict_filtered_next(X_train[t], timestamp=ts)

        # ── Step 4: Walk OOS bar by bar ───────────────────────────────────────
        cash = equity_at_fold_start
        shares: dict[str, int] = {sym: 0 for sym in symbols}
        cost_basis: dict[str, float] = {sym: 0.0 for sym in symbols}  # weighted avg cost

        # Pending orders: list of dicts with keys: symbol, delta_shares, fill_price_close, side
        pending_orders: list[dict] = []

        equity_index: list[pd.Timestamp] = []
        equity_values: list[float] = []
        trade_rows: list[dict] = []
        regime_rows: list[dict] = []

        X_test = test_features.values.astype(float)
        test_timestamps = test_features.index if isinstance(test_features.index, pd.DatetimeIndex) else None
        n_test = len(X_test)

        # Keep a rolling history of regime states for flicker detection
        regime_state_history: list[RegimeState] = []

        for t in range(n_test):
            ts = test_timestamps[t] if test_timestamps is not None else None

            # Get current prices for all symbols
            current_prices: dict[str, float] = {}
            for sym in symbols:
                sym_df = test_ohlcv.get(sym)
                if sym_df is not None and len(sym_df) > t:
                    current_prices[sym] = float(sym_df["close"].iloc[t])
                elif sym_df is not None and len(sym_df) > 0:
                    # Use last available price
                    current_prices[sym] = float(sym_df["close"].iloc[-1])
                else:
                    current_prices[sym] = 1.0  # fallback

            # ── Compute equity before fills ───────────────────────────────────
            equity_before = cash + sum(
                shares[sym] * current_prices.get(sym, 1.0)
                for sym in symbols
            )

            # ── Execute PENDING orders from previous bar ──────────────────────
            for order in pending_orders:
                sym = order["symbol"]
                delta = order["delta_shares"]
                side = order["side"]
                close_price = current_prices.get(sym, 1.0)
                fill_price = self._simulate_fill(sym, side, abs(delta), close_price)

                old_sh = shares[sym]
                new_sh = old_sh + delta
                shares[sym] = new_sh

                # Commission
                commission_cost = abs(delta) * fill_price * self.config.commission
                cash -= delta * fill_price + commission_cost

                # Realised P&L
                if side == "SELL" and old_sh > 0:
                    sold_qty = min(abs(delta), old_sh)
                    realised_pnl = sold_qty * (fill_price - cost_basis.get(sym, fill_price))
                    realised_pnl_pct = (fill_price / cost_basis.get(sym, fill_price) - 1.0) if cost_basis.get(sym, fill_price) > 0 else 0.0
                    entry_px = cost_basis.get(sym, fill_price)
                    exit_px = fill_price
                else:
                    realised_pnl = 0.0
                    realised_pnl_pct = 0.0
                    entry_px = fill_price
                    exit_px = 0.0

                # Update cost basis (weighted average for buys)
                if side == "BUY" and delta > 0:
                    old_cost_total = cost_basis.get(sym, 0.0) * old_sh
                    new_cost_total = old_cost_total + delta * fill_price
                    cost_basis[sym] = new_cost_total / new_sh if new_sh > 0 else fill_price
                elif side == "SELL" and new_sh == 0:
                    cost_basis[sym] = 0.0

                equity_after = cash + sum(
                    shares[s] * current_prices.get(s, 1.0)
                    for s in symbols
                )

                trade_rows.append({
                    "timestamp": ts,
                    "symbol": sym,
                    "side": side,
                    "shares": abs(delta),
                    "fill_price": fill_price,
                    "old_shares": old_sh,
                    "new_shares": new_sh,
                    "equity_before": equity_before,
                    "equity_after": equity_after,
                    "regime_label": order.get("regime_label", ""),
                    "regime_confidence": order.get("regime_confidence", 0.0),
                    "target_allocation": order.get("target_allocation", 0.0),
                    "leverage": order.get("leverage", 1.0),
                    "realised_pnl": realised_pnl,
                    "realised_pnl_pct": realised_pnl_pct,
                    "entry_price": entry_px,
                    "exit_price": exit_px,
                    "qty": abs(delta),
                    "regime_at_entry": order.get("regime_label", ""),
                })

            pending_orders = []

            # ── Recompute equity after fills ──────────────────────────────────
            equity_now = cash + sum(
                shares[sym] * current_prices.get(sym, 1.0)
                for sym in symbols
            )

            # ── HMM: advance forward algorithm ───────────────────────────────
            regime_state = engine.predict_filtered_next(X_test[t], timestamp=ts)
            regime_state_history.append(regime_state)

            is_flickering = engine.is_flickering(regime_state_history)

            # Record regime history
            regime_rows.append({
                "date": ts,
                "regime": regime_state.label,
                "confidence": regime_state.probability,
                "is_confirmed": regime_state.is_confirmed,
                "in_transition": regime_state.in_transition,
                "fold": fold_idx,
            })

            # ── Build bars windows for each symbol ────────────────────────────
            # Use available OOS bars up to and including current bar t
            bars_dict: dict[str, pd.DataFrame] = {}
            for sym in symbols:
                sym_df = test_ohlcv.get(sym)
                if sym_df is not None and len(sym_df) > 0:
                    # Also prepend last N bars of training data for indicator warmup
                    train_sym_df = test_ohlcv.get(sym)
                    end_idx = min(t + 1, len(sym_df))
                    bars_dict[sym] = sym_df.iloc[:end_idx].copy()
                else:
                    bars_dict[sym] = pd.DataFrame(
                        columns=["open", "high", "low", "close", "volume"]
                    )

            # ── Generate signals from orchestrator ────────────────────────────
            signals: list[Signal] = orchestrator.generate_signals(
                symbols=symbols,
                bars=bars_dict,
                regime_state=regime_state,
                is_flickering=is_flickering,
            )

            # ── Compute target allocations ────────────────────────────────────
            n_syms = max(len(symbols), 1)
            current_weights: dict[str, float] = {}
            target_weights: dict[str, float] = {}

            for sym in symbols:
                price = current_prices.get(sym, 1.0)
                sym_value = shares[sym] * price
                current_weights[sym] = sym_value / equity_now if equity_now > 0 else 0.0

            # Map signal by symbol
            signal_by_sym: dict[str, Signal] = {s.symbol: s for s in signals}

            for sym in symbols:
                sig = signal_by_sym.get(sym)
                if sig is not None and sig.direction == "LONG":
                    alloc_per_sym = sig.position_size_pct * sig.leverage / n_syms
                else:
                    alloc_per_sym = 0.0
                target_weights[sym] = alloc_per_sym

            # ── Check if rebalance needed ─────────────────────────────────────
            needs_rebal = orchestrator.needs_rebalance(current_weights, target_weights)

            if needs_rebal:
                # Queue new orders as PENDING (1-bar fill delay)
                for sym in symbols:
                    sig = signal_by_sym.get(sym)
                    price = current_prices.get(sym, 1.0)
                    alloc = target_weights[sym]

                    # target_shares = int(equity * alloc / price)
                    target_sh = int(equity_now * alloc / price) if price > 0 else 0
                    delta = target_sh - shares[sym]

                    if delta == 0:
                        continue

                    side = "BUY" if delta > 0 else "SELL"
                    pending_orders.append({
                        "symbol": sym,
                        "delta_shares": delta,
                        "side": side,
                        "regime_label": regime_state.label,
                        "regime_confidence": regime_state.probability,
                        "target_allocation": alloc,
                        "leverage": sig.leverage if sig else 1.0,
                    })

            # ── Record equity ─────────────────────────────────────────────────
            equity_index.append(ts if ts is not None else pd.Timestamp(f"2000-01-01") + pd.Timedelta(days=t))
            equity_values.append(equity_now)

        # Build equity series
        if equity_index:
            eq_series = pd.Series(equity_values, index=pd.DatetimeIndex(equity_index))
        else:
            eq_series = pd.Series(dtype=float)

        # Build trade ledger
        trades_df = pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame(
            columns=[
                "timestamp", "symbol", "side", "shares", "fill_price",
                "old_shares", "new_shares", "equity_before", "equity_after",
                "regime_label", "regime_confidence", "target_allocation", "leverage",
                "realised_pnl", "realised_pnl_pct", "entry_price", "exit_price",
                "qty", "regime_at_entry",
            ]
        )

        fold_metrics = _compute_fold_metrics(
            eq_series, trades_df, self.config.risk_free_rate
        )

        return eq_series, trades_df, fold_metrics, regime_rows

    # ─────────────────────────────────────────────────────────────────────────
    # Private: utilities
    # ─────────────────────────────────────────────────────────────────────────

    def _simulate_fill(
        self,
        symbol: str,
        side: str,
        shares: int,
        bar_close: float,
    ) -> float:
        """
        Simulate order fill with slippage.

        Returns fill_price = bar_close * (1 + slippage) for BUY,
                              bar_close * (1 - slippage) for SELL.
        """
        if side == "BUY":
            return bar_close * (1.0 + self.config.slippage_pct)
        else:
            return bar_close * (1.0 - self.config.slippage_pct)

    def _build_equity_curve(
        self,
        fold_curves: list[pd.Series],
    ) -> pd.Series:
        """Concatenate per-fold equity series into a single continuous curve."""
        valid = [s for s in fold_curves if len(s) > 0]
        if not valid:
            return pd.Series(
                self.config.initial_capital,
                index=pd.DatetimeIndex([]),
                dtype=float,
            )
        combined = pd.concat(valid)
        # Remove duplicate index entries (overlap at fold boundaries)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        return combined


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_buyhold_benchmark(
    ohlcv: dict[str, pd.DataFrame],
    initial_capital: float = 100_000.0,
) -> pd.Series:
    """
    Simulate equal-weight buy-and-hold benchmark.

    Buys equal shares of each symbol at open of first bar,
    holds until end.  Returns equity curve aligned to the
    common date index.

    Parameters
    ----------
    ohlcv : dict[str, pd.DataFrame]
        Symbol → OHLCV DataFrame.  Expects column "close".
    initial_capital : float

    Returns
    -------
    pd.Series — DatetimeIndex → equity.
    """
    symbols = list(ohlcv.keys())
    n = len(symbols)
    if n == 0:
        return pd.Series(dtype=float)

    capital_per_sym = initial_capital / n
    holdings: dict[str, tuple[int, float]] = {}  # sym → (shares, cost_basis)

    # Use first symbol to get common index
    first_df = ohlcv[symbols[0]]
    common_index = first_df.index

    for sym in symbols:
        df = ohlcv[sym]
        entry_price = float(df["close"].iloc[0])
        n_shares = int(capital_per_sym / entry_price) if entry_price > 0 else 0
        holdings[sym] = (n_shares, entry_price)

    equity_values = []
    for ts in common_index:
        total = 0.0
        cash_remaining = initial_capital - sum(
            n_sh * ep for n_sh, ep in holdings.values()
        )
        for sym in symbols:
            n_sh, _ = holdings[sym]
            df = ohlcv[sym]
            if ts in df.index:
                price = float(df.loc[ts, "close"])
            elif len(df) > 0:
                price = float(df["close"].iloc[-1])
            else:
                price = 0.0
            total += n_sh * price
        equity_values.append(total + cash_remaining)

    return pd.Series(equity_values, index=common_index, name="buyhold")


def run_sma200_benchmark(
    ohlcv: dict[str, pd.DataFrame],
    initial_capital: float = 100_000.0,
) -> pd.Series:
    """
    Simulate SMA-200 trend-following benchmark on first symbol.

    Holds when price > SMA-200, moves to cash otherwise.
    Uses close prices; fills next bar (1-bar delay).

    Parameters
    ----------
    ohlcv : dict[str, pd.DataFrame]
        Symbol → OHLCV DataFrame.  Uses first symbol as market proxy.
    initial_capital : float

    Returns
    -------
    pd.Series — DatetimeIndex → equity.
    """
    symbols = list(ohlcv.keys())
    if not symbols:
        return pd.Series(dtype=float)

    proxy_sym = symbols[0]
    df = ohlcv[proxy_sym].copy()
    df.columns = [c.lower() for c in df.columns]

    close = df["close"]
    sma200 = close.rolling(200, min_periods=1).mean()
    in_market = close > sma200  # True = long, False = cash

    cash = initial_capital
    n_shares = 0
    cost_basis = 0.0
    equity_values = []

    for i, ts in enumerate(df.index):
        price = float(close.iloc[i])
        above = bool(in_market.iloc[i])

        # 1-bar delay: act on signal from previous bar
        if i > 0:
            prev_above = bool(in_market.iloc[i - 1])
            if prev_above and n_shares == 0:
                # Buy
                n_shares = int(cash / price) if price > 0 else 0
                cost_basis = price
                cash -= n_shares * price
            elif not prev_above and n_shares > 0:
                # Sell
                cash += n_shares * price
                n_shares = 0
                cost_basis = 0.0

        equity = cash + n_shares * price
        equity_values.append(equity)

    return pd.Series(equity_values, index=df.index, name="sma200")
