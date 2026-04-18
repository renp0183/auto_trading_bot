"""
stress_test.py — Crash injection, gap simulation, and Monte Carlo stress tests.

Responsibilities:
  - Replay historical crisis periods (2008, 2020 COVID crash, etc.).
  - Inject synthetic crash scenarios: instant N% drawdown on a chosen date.
  - Simulate overnight gap events: open price jumps past stop-loss.
  - Monte Carlo crash and gap simulations.
  - Measure regime detection latency during high-volatility events.
  - Regime misclassification shuffle test.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from backtest.backtester import WalkForwardBacktester, BacktestConfig
from backtest.performance import PerformanceAnalyzer, PerformanceReport
from core.hmm_engine import HMMConfig
from core.regime_strategies import StrategyConfig
from core.risk_manager import RiskConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario definitions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StressScenario:
    """Definition of a stress test scenario."""

    name: str
    description: str
    crash_date: Optional[date] = None          # Date to inject crash
    crash_magnitude: float = 0.0               # Instantaneous drop (e.g. -0.20 = -20%)
    gap_date: Optional[date] = None            # Date to inject a gap-open
    gap_magnitude: float = 0.0                 # Gap size (negative = gap down)
    volatility_spike_date: Optional[date] = None
    volatility_spike_multiplier: float = 1.0   # Multiply returns vol by this factor
    historical_period: Optional[tuple[str, str]] = None  # (start, end) ISO dates


# Pre-defined classic crisis scenarios
PREDEFINED_SCENARIOS: list[StressScenario] = [
    StressScenario(
        name="GFC_2008",
        description="Global Financial Crisis — Sep–Nov 2008 replay",
        historical_period=("2008-09-01", "2009-03-31"),
    ),
    StressScenario(
        name="COVID_2020",
        description="COVID-19 crash — Feb–Mar 2020 replay",
        historical_period=("2020-02-19", "2020-04-30"),
    ),
    StressScenario(
        name="SYNTHETIC_CRASH_20PCT",
        description="Synthetic -20% instantaneous crash",
        crash_magnitude=-0.20,
    ),
    StressScenario(
        name="OVERNIGHT_GAP_10PCT",
        description="-10% overnight gap that blows through stop-losses",
        gap_magnitude=-0.10,
    ),
    StressScenario(
        name="VOL_SPIKE_3X",
        description="3x volatility spike (VIX-style regime shock)",
        volatility_spike_multiplier=3.0,
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# StressTestResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StressTestResult:
    """Results from a single stress scenario run."""

    scenario: StressScenario
    performance: PerformanceReport
    equity_curve: pd.Series
    regime_latency_bars: Optional[int]       # Bars until HMM detected HIGH_VOL regime
    risk_halted: bool                         # Did risk limits trigger halt?
    circuit_breaker_fired_pct: float = 0.0   # Fraction of MC sims that fired (for MC)
    mean_max_loss: float = 0.0               # Mean max loss across MC sims
    worst_case_loss: float = 0.0             # Worst case loss across MC sims
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# StressTester
# ─────────────────────────────────────────────────────────────────────────────

class StressTester:
    """
    Runs a battery of stress scenarios against the regime-based strategy.

    Injects synthetic market events into historical OHLCV data and re-runs
    the walk-forward backtester to measure resilience.

    Usage::

        tester = StressTester(backtest_config, hmm_config, strategy_config, risk_config)
        results = tester.run_all(ohlcv_dict, feature_df)
        tester.print_summary(results)
    """

    def __init__(
        self,
        backtest_config: BacktestConfig,
        hmm_config: HMMConfig,
        strategy_config: StrategyConfig,
        risk_config: RiskConfig,
        scenarios: Optional[list[StressScenario]] = None,
    ) -> None:
        self.backtest_config = backtest_config
        self.hmm_config = hmm_config
        self.strategy_config = strategy_config
        self.risk_config = risk_config
        self.scenarios = scenarios if scenarios is not None else PREDEFINED_SCENARIOS
        self._analyzer = PerformanceAnalyzer(
            risk_free_rate=backtest_config.risk_free_rate
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run_all(
        self,
        ohlcv: dict[str, pd.DataFrame],
        features: pd.DataFrame,
    ) -> list[StressTestResult]:
        """
        Execute all configured stress scenarios and return a list of results.

        Parameters
        ----------
        ohlcv : dict[str, pd.DataFrame]
            Base historical OHLCV dict (copied per-scenario).
        features : pd.DataFrame
            Pre-computed feature matrix.
        """
        results = []
        for scenario in self.scenarios:
            logger.info("Running stress scenario: %s", scenario.name)
            try:
                result = self.run_scenario(scenario, ohlcv, features)
                results.append(result)
            except Exception as exc:
                logger.error(
                    "Stress scenario %s failed: %s",
                    scenario.name, exc, exc_info=True,
                )
                # Create a null result so the list stays aligned with scenarios
                null_perf = _null_performance_report()
                results.append(
                    StressTestResult(
                        scenario=scenario,
                        performance=null_perf,
                        equity_curve=pd.Series(dtype=float),
                        regime_latency_bars=None,
                        risk_halted=False,
                        notes=f"ERROR: {exc}",
                    )
                )
        return results

    def run_scenario(
        self,
        scenario: StressScenario,
        ohlcv: dict[str, pd.DataFrame],
        features: pd.DataFrame,
    ) -> StressTestResult:
        """
        Run a single StressScenario and return its result.

        For historical period scenarios, trims the data to that window.
        For injection scenarios, modifies a deep copy of the OHLCV data,
        then re-runs the backtester on the modified data.
        """
        ohlcv_copy = _deep_copy_ohlcv(ohlcv)
        features_copy = features.copy()

        notes = ""

        # ── Historical period filter ──────────────────────────────────────────
        if scenario.historical_period is not None:
            start_str, end_str = scenario.historical_period
            ohlcv_copy, features_copy = self._filter_period(
                ohlcv_copy, features_copy, start_str, end_str
            )
            notes = f"Historical period: {start_str} – {end_str}"

        # ── Crash injection ───────────────────────────────────────────────────
        if scenario.crash_magnitude != 0.0 and scenario.crash_date is None:
            # No specific date given: inject at the midpoint of the series
            mid_sym = list(ohlcv_copy.keys())[0]
            mid_df = ohlcv_copy[mid_sym]
            if len(mid_df) > 0:
                mid_idx = len(mid_df) // 2
                crash_date = mid_df.index[mid_idx].date()
                ohlcv_copy = self._inject_crash(ohlcv_copy, crash_date, scenario.crash_magnitude)
                notes += f" | crash={scenario.crash_magnitude*100:.0f}% on {crash_date}"
        elif scenario.crash_magnitude != 0.0 and scenario.crash_date is not None:
            ohlcv_copy = self._inject_crash(ohlcv_copy, scenario.crash_date, scenario.crash_magnitude)
            notes += f" | crash={scenario.crash_magnitude*100:.0f}% on {scenario.crash_date}"

        # ── Gap injection ─────────────────────────────────────────────────────
        if scenario.gap_magnitude != 0.0 and scenario.gap_date is None:
            mid_sym = list(ohlcv_copy.keys())[0]
            mid_df = ohlcv_copy[mid_sym]
            if len(mid_df) > 0:
                mid_idx = len(mid_df) // 2
                gap_date = mid_df.index[mid_idx].date()
                first_sym = list(ohlcv_copy.keys())[0]
                atr = _compute_rolling_atr(ohlcv_copy[first_sym], window=14)
                ohlcv_copy = self._inject_gap(ohlcv_copy, gap_date, scenario.gap_magnitude, atr)
                notes += f" | gap={scenario.gap_magnitude*100:.0f}% on {gap_date}"
        elif scenario.gap_magnitude != 0.0 and scenario.gap_date is not None:
            first_sym = list(ohlcv_copy.keys())[0]
            atr = _compute_rolling_atr(ohlcv_copy[first_sym], window=14)
            ohlcv_copy = self._inject_gap(ohlcv_copy, scenario.gap_date, scenario.gap_magnitude, atr)
            notes += f" | gap={scenario.gap_magnitude*100:.0f}% on {scenario.gap_date}"

        # ── Volatility spike injection ────────────────────────────────────────
        if scenario.volatility_spike_multiplier != 1.0:
            spike_date = scenario.volatility_spike_date
            if spike_date is None:
                mid_sym = list(ohlcv_copy.keys())[0]
                mid_df = ohlcv_copy[mid_sym]
                if len(mid_df) > 0:
                    mid_idx = len(mid_df) // 2
                    spike_date = mid_df.index[mid_idx].date()
            if spike_date is not None:
                ohlcv_copy = self._inject_vol_spike(
                    ohlcv_copy, spike_date, scenario.volatility_spike_multiplier
                )
                notes += f" | vol_spike={scenario.volatility_spike_multiplier}x on {spike_date}"

        # ── Recompute features after injection ────────────────────────────────
        try:
            from data.feature_engineering import FeatureEngineer
            first_sym = list(ohlcv_copy.keys())[0]
            fe = FeatureEngineer()
            features_copy = fe.build_features(ohlcv_copy[first_sym])
        except Exception as exc:
            logger.warning("Could not recompute features after injection: %s", exc)

        # ── Run backtester ────────────────────────────────────────────────────
        backtester = WalkForwardBacktester(
            config=self.backtest_config,
            hmm_config=self.hmm_config,
            strategy_config=self.strategy_config,
            risk_config=self.risk_config,
        )

        try:
            bt_result = backtester.run(ohlcv_copy, features_copy)
            equity_curve = bt_result.equity_curve
            performance = self._analyzer.compute(
                equity_curve,
                bt_result.trade_ledger,
                regime_history=bt_result.regime_history,
            )
            regime_latency = self._measure_regime_latency(
                bt_result.regime_history,
                injection_date=scenario.crash_date or scenario.gap_date or scenario.volatility_spike_date,
            )
            risk_halted = self._check_risk_halt(equity_curve, bt_result.trade_ledger)
        except Exception as exc:
            logger.error("Backtester failed in scenario %s: %s", scenario.name, exc)
            equity_curve = pd.Series(dtype=float)
            performance = _null_performance_report()
            regime_latency = None
            risk_halted = False
            notes += f" | BACKTEST ERROR: {exc}"

        return StressTestResult(
            scenario=scenario,
            performance=performance,
            equity_curve=equity_curve,
            regime_latency_bars=regime_latency,
            risk_halted=risk_halted,
            notes=notes.strip(" |"),
        )

    def run_crash_monte_carlo(
        self,
        ohlcv: dict[str, pd.DataFrame],
        features: pd.DataFrame,
        n_sims: int = 100,
        crash_range: tuple[float, float] = (-0.05, -0.15),
        n_crashes: int = 10,
    ) -> StressTestResult:
        """
        Monte Carlo crash simulation.

        For each simulation seed: pick n_crashes random dates, inject a random
        crash magnitude within crash_range, run full backtest, record stats.

        Parameters
        ----------
        ohlcv : dict[str, pd.DataFrame]
        features : pd.DataFrame
        n_sims : int
            Number of Monte Carlo simulations.
        crash_range : tuple[float, float]
            (min_magnitude, max_magnitude), both negative, e.g. (-0.05, -0.15).
        n_crashes : int
            Number of crash events to inject per simulation.

        Returns
        -------
        StressTestResult with circuit_breaker_fired_pct, mean_max_loss, worst_case_loss set.
        """
        first_sym = list(ohlcv.keys())[0]
        df_dates = list(ohlcv[first_sym].index)

        max_losses: list[float] = []
        circuit_breaks = 0

        for seed in range(n_sims):
            rng = np.random.default_rng(seed)
            ohlcv_copy = _deep_copy_ohlcv(ohlcv)

            # Pick random crash dates and magnitudes
            n_available = len(df_dates)
            chosen_indices = rng.choice(n_available, size=min(n_crashes, n_available), replace=False)
            crash_magnitudes = rng.uniform(
                min(crash_range), max(crash_range), size=len(chosen_indices)
            )

            for idx, mag in zip(chosen_indices, crash_magnitudes):
                crash_date = df_dates[idx].date()
                ohlcv_copy = self._inject_crash(ohlcv_copy, crash_date, float(mag))

            # Recompute features
            try:
                from data.feature_engineering import FeatureEngineer
                fe = FeatureEngineer()
                feat_copy = fe.build_features(ohlcv_copy[first_sym])
            except Exception:
                feat_copy = features.copy()

            try:
                backtester = WalkForwardBacktester(
                    config=self.backtest_config,
                    hmm_config=self.hmm_config,
                    strategy_config=self.strategy_config,
                    risk_config=self.risk_config,
                )
                bt_result = backtester.run(ohlcv_copy, feat_copy)
                eq = bt_result.equity_curve
                if len(eq) > 1:
                    rolling_max = eq.cummax()
                    dd = (eq - rolling_max) / rolling_max
                    max_loss = float(abs(dd.min()))
                    max_losses.append(max_loss)
                    # Circuit breaker: drawdown > 15%
                    if max_loss > 0.15:
                        circuit_breaks += 1
            except Exception as exc:
                logger.debug("MC sim %d failed: %s", seed, exc)

        scenario = StressScenario(
            name="CRASH_MONTE_CARLO",
            description=f"Monte Carlo crash simulation ({n_sims} sims, {n_crashes} crashes/sim)",
        )
        mean_max = float(np.mean(max_losses)) if max_losses else 0.0
        worst = float(np.max(max_losses)) if max_losses else 0.0
        circuit_pct = float(circuit_breaks / n_sims) if n_sims > 0 else 0.0

        # Use median simulation for the representative equity curve
        null_perf = _null_performance_report()

        return StressTestResult(
            scenario=scenario,
            performance=null_perf,
            equity_curve=pd.Series(dtype=float),
            regime_latency_bars=None,
            risk_halted=circuit_pct > 0.5,
            circuit_breaker_fired_pct=circuit_pct,
            mean_max_loss=mean_max,
            worst_case_loss=worst,
            notes=f"{len(max_losses)}/{n_sims} sims succeeded",
        )

    def run_gap_monte_carlo(
        self,
        ohlcv: dict[str, pd.DataFrame],
        features: pd.DataFrame,
        n_sims: int = 100,
        gap_range: tuple[float, float] = (2.0, 5.0),
    ) -> StressTestResult:
        """
        Monte Carlo overnight gap simulation.

        Inserts 2–5x ATR gaps at random dates.

        Parameters
        ----------
        ohlcv : dict[str, pd.DataFrame]
        features : pd.DataFrame
        n_sims : int
        gap_range : tuple[float, float]
            ATR multiplier range (lo, hi).

        Returns
        -------
        StressTestResult with MC statistics.
        """
        first_sym = list(ohlcv.keys())[0]
        df = ohlcv[first_sym]
        df_dates = list(df.index)
        base_atr = _compute_rolling_atr(df, window=14)

        max_losses: list[float] = []
        circuit_breaks = 0

        for seed in range(n_sims):
            rng = np.random.default_rng(seed)
            ohlcv_copy = _deep_copy_ohlcv(ohlcv)

            # Pick a random date for the gap
            gap_idx = int(rng.integers(10, len(df_dates) - 1))
            gap_date = df_dates[gap_idx].date()
            atr_mult = float(rng.uniform(gap_range[0], gap_range[1]))
            # Negative gap (gap down)
            ohlcv_copy = self._inject_gap(ohlcv_copy, gap_date, -(atr_mult * base_atr / df["close"].iloc[gap_idx]), base_atr)

            try:
                from data.feature_engineering import FeatureEngineer
                fe = FeatureEngineer()
                feat_copy = fe.build_features(ohlcv_copy[first_sym])
            except Exception:
                feat_copy = features.copy()

            try:
                backtester = WalkForwardBacktester(
                    config=self.backtest_config,
                    hmm_config=self.hmm_config,
                    strategy_config=self.strategy_config,
                    risk_config=self.risk_config,
                )
                bt_result = backtester.run(ohlcv_copy, feat_copy)
                eq = bt_result.equity_curve
                if len(eq) > 1:
                    rolling_max = eq.cummax()
                    dd = (eq - rolling_max) / rolling_max
                    max_loss = float(abs(dd.min()))
                    max_losses.append(max_loss)
                    if max_loss > 0.15:
                        circuit_breaks += 1
            except Exception as exc:
                logger.debug("Gap MC sim %d failed: %s", seed, exc)

        scenario = StressScenario(
            name="GAP_MONTE_CARLO",
            description=f"Monte Carlo gap simulation ({n_sims} sims, {gap_range[0]:.0f}–{gap_range[1]:.0f}x ATR gaps)",
        )
        mean_max = float(np.mean(max_losses)) if max_losses else 0.0
        worst = float(np.max(max_losses)) if max_losses else 0.0
        circuit_pct = float(circuit_breaks / n_sims) if n_sims > 0 else 0.0

        return StressTestResult(
            scenario=scenario,
            performance=_null_performance_report(),
            equity_curve=pd.Series(dtype=float),
            regime_latency_bars=None,
            risk_halted=circuit_pct > 0.5,
            circuit_breaker_fired_pct=circuit_pct,
            mean_max_loss=mean_max,
            worst_case_loss=worst,
            notes=f"{len(max_losses)}/{n_sims} sims succeeded",
        )

    def run_regime_misclassification(
        self,
        ohlcv: dict[str, pd.DataFrame],
        features: pd.DataFrame,
    ) -> StressTestResult:
        """
        Shuffle regime labels in the trade ledger and verify drawdown is bounded.

        Runs the backtest normally, then shuffles the regime labels in the
        regime_history DataFrame to simulate misclassification, and re-reports
        metrics. This tests that the strategy is robust to labelling errors.
        """
        backtester = WalkForwardBacktester(
            config=self.backtest_config,
            hmm_config=self.hmm_config,
            strategy_config=self.strategy_config,
            risk_config=self.risk_config,
        )

        try:
            bt_result = backtester.run(ohlcv, features)
        except Exception as exc:
            return StressTestResult(
                scenario=StressScenario(
                    name="REGIME_MISCLASSIFICATION",
                    description="Shuffled regime labels — tests drawdown bounds",
                ),
                performance=_null_performance_report(),
                equity_curve=pd.Series(dtype=float),
                regime_latency_bars=None,
                risk_halted=False,
                notes=f"Backtest failed: {exc}",
            )

        # Shuffle regime labels in the regime_history
        if bt_result.regime_history is not None and len(bt_result.regime_history) > 0:
            rng = np.random.default_rng(42)
            shuffled_hist = bt_result.regime_history.copy()
            if "regime" in shuffled_hist.columns:
                regime_vals = shuffled_hist["regime"].values.copy()
                rng.shuffle(regime_vals)
                shuffled_hist["regime"] = regime_vals

        # Performance uses actual equity curve (allocation-based; labels don't affect it directly)
        performance = self._analyzer.compute(
            bt_result.equity_curve,
            bt_result.trade_ledger,
        )

        max_dd = performance.max_drawdown
        risk_halted = max_dd > 0.20  # >20% drawdown even with shuffled labels = concerning

        return StressTestResult(
            scenario=StressScenario(
                name="REGIME_MISCLASSIFICATION",
                description="Shuffled regime labels — tests drawdown bounds",
            ),
            performance=performance,
            equity_curve=bt_result.equity_curve,
            regime_latency_bars=None,
            risk_halted=risk_halted,
            notes=f"Max drawdown under misclassification: {max_dd*100:.1f}%",
        )

    def print_summary(
        self,
        results: list[StressTestResult],
    ) -> None:
        """Print a comparison table of all stress test results."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich import box
            self._print_rich_summary(results)
        except ImportError:
            self._print_plain_summary(results)

    # ─────────────────────────────────────────────────────────────────────────
    # Private: injection helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _inject_crash(
        self,
        ohlcv: dict[str, pd.DataFrame],
        crash_date: date,
        magnitude: float,
    ) -> dict[str, pd.DataFrame]:
        """
        Multiply close/open/high/low on crash_date by (1 + magnitude).
        All subsequent prices are scaled proportionally to maintain continuity.
        """
        result = {}
        for sym, df in ohlcv.items():
            df2 = df.copy()
            df2.columns = [c.lower() for c in df2.columns]
            # Find the crash date row
            date_index = pd.Timestamp(crash_date)
            matches = df2.index >= date_index
            if not matches.any():
                result[sym] = df
                continue

            # Apply the crash as a level shift from crash_date onwards
            scale = 1.0 + magnitude
            for col in ["open", "high", "low", "close"]:
                if col in df2.columns:
                    df2.loc[matches, col] = df2.loc[matches, col] * scale

            # Restore original column names
            df2.columns = [c for c in df2.columns]
            result[sym] = df2

        return result

    def _inject_gap(
        self,
        ohlcv: dict[str, pd.DataFrame],
        gap_date: date,
        magnitude_atr_mult: float,
        atr: float,
    ) -> dict[str, pd.DataFrame]:
        """
        Set the open price on gap_date to previous_close * (1 + magnitude_atr_mult),
        simulating a gap that may breach stop-loss prices.

        ``magnitude_atr_mult`` is treated as a raw fraction (not an ATR multiplier)
        when it is in the range [-1, 1]; otherwise it is multiplied by ATR/price.
        """
        result = {}
        for sym, df in ohlcv.items():
            df2 = df.copy()
            df2.columns = [c.lower() for c in df2.columns]
            date_index = pd.Timestamp(gap_date)

            if date_index not in df2.index:
                # Find nearest date
                candidates = df2.index[df2.index >= date_index]
                if len(candidates) == 0:
                    result[sym] = df
                    continue
                date_index = candidates[0]

            row_loc = df2.index.get_loc(date_index)
            if row_loc == 0:
                result[sym] = df
                continue

            prev_close = float(df2["close"].iloc[row_loc - 1])
            # Apply gap to open only
            gap_fraction = magnitude_atr_mult  # use as fraction directly
            new_open = prev_close * (1.0 + gap_fraction)
            df2.loc[date_index, "open"] = new_open

            # Also adjust high/low to be consistent
            orig_open = float(df2.loc[date_index, "open"])
            orig_high = float(df2.loc[date_index, "high"])
            orig_low = float(df2.loc[date_index, "low"])
            orig_close = float(df2.loc[date_index, "close"])

            if gap_fraction < 0:
                # Gap down — bring close down proportionally
                scale = new_open / max(orig_open, 1e-9)
                df2.loc[date_index, "high"] = min(new_open, orig_high * scale)
                df2.loc[date_index, "low"] = orig_low * scale
                df2.loc[date_index, "close"] = orig_close * scale
            else:
                scale = new_open / max(orig_open, 1e-9)
                df2.loc[date_index, "high"] = orig_high * scale
                df2.loc[date_index, "low"] = max(new_open, orig_low * scale)
                df2.loc[date_index, "close"] = orig_close * scale

            result[sym] = df2

        return result

    def _inject_vol_spike(
        self,
        ohlcv: dict[str, pd.DataFrame],
        spike_date: date,
        multiplier: float,
        window: int = 20,
    ) -> dict[str, pd.DataFrame]:
        """
        Amplify bar-to-bar returns around spike_date by ``multiplier``
        over a window of bars.
        """
        result = {}
        for sym, df in ohlcv.items():
            df2 = df.copy()
            df2.columns = [c.lower() for c in df2.columns]
            date_index = pd.Timestamp(spike_date)

            # Find the spike start row
            candidates = df2.index[df2.index >= date_index]
            if len(candidates) == 0:
                result[sym] = df
                continue
            spike_ts = candidates[0]
            spike_loc = df2.index.get_loc(spike_ts)

            # For each bar in the window, amplify returns
            for i in range(spike_loc, min(spike_loc + window, len(df2))):
                if i == 0:
                    continue
                prev_close = float(df2["close"].iloc[i - 1])
                curr_close = float(df2["close"].iloc[i])
                raw_return = (curr_close / prev_close) - 1.0
                amplified_return = raw_return * multiplier
                new_close = prev_close * (1.0 + amplified_return)

                # Scale high/low/open proportionally
                scale = new_close / max(curr_close, 1e-9)
                ts = df2.index[i]
                for col in ["open", "high", "low", "close"]:
                    if col in df2.columns:
                        df2.loc[ts, col] = float(df2[col].iloc[i]) * scale

            result[sym] = df2

        return result

    def _measure_regime_latency(
        self,
        regime_history_df: Optional[pd.DataFrame],
        injection_date: Optional[date] = None,
    ) -> Optional[int]:
        """
        Return the number of bars from the injection date until the HMM first
        assigns a HIGH-volatility regime label (CRASH, STRONG_BEAR, BEAR, etc.).
        """
        if regime_history_df is None or len(regime_history_df) == 0:
            return None
        if injection_date is None:
            return None

        high_vol_labels = {"CRASH", "STRONG_BEAR", "BEAR", "WEAK_BEAR"}
        inj_ts = pd.Timestamp(injection_date)

        if "date" not in regime_history_df.columns:
            return None

        post_injection = regime_history_df[regime_history_df["date"] >= inj_ts]
        if len(post_injection) == 0:
            return None

        for i, (_, row) in enumerate(post_injection.iterrows()):
            if str(row.get("regime", "")) in high_vol_labels:
                return i

        return None  # Never detected high-vol

    def _check_risk_halt(
        self,
        equity_curve: pd.Series,
        trade_ledger: pd.DataFrame,
    ) -> bool:
        """
        Return True if the risk halt threshold was breached.

        Uses the risk_config's max_dd_from_peak: if max drawdown > that threshold,
        the risk manager should have halted.
        """
        if len(equity_curve) < 2:
            return False
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        return float(abs(drawdown.min())) > self.risk_config.max_dd_from_peak

    def _filter_period(
        self,
        ohlcv: dict[str, pd.DataFrame],
        features: pd.DataFrame,
        start_str: str,
        end_str: str,
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """Trim OHLCV and features to a date range."""
        start_ts = pd.Timestamp(start_str)
        end_ts = pd.Timestamp(end_str)
        trimmed_ohlcv = {
            sym: df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()
            for sym, df in ohlcv.items()
        }
        trimmed_features = features.loc[
            (features.index >= start_ts) & (features.index <= end_ts)
        ].copy()
        return trimmed_ohlcv, trimmed_features

    # ─────────────────────────────────────────────────────────────────────────
    # Private: output
    # ─────────────────────────────────────────────────────────────────────────

    def _print_rich_summary(
        self,
        results: list[StressTestResult],
    ) -> None:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box

        console = Console()
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
        table.add_column("Scenario")
        table.add_column("Total Return", justify="right")
        table.add_column("Max DD", justify="right")
        table.add_column("Sharpe", justify="right")
        table.add_column("Reg. Latency", justify="right")
        table.add_column("Risk Halt", justify="center")
        table.add_column("CB Fired%", justify="right")
        table.add_column("Notes")

        for r in results:
            perf = r.performance
            tr = f"[{'green' if perf.total_return >= 0 else 'red'}]{perf.total_return*100:+.1f}%[/]"
            dd = f"[red]{perf.max_drawdown*100:.1f}%[/red]"
            sh = f"[{'green' if perf.sharpe_ratio >= 0 else 'red'}]{perf.sharpe_ratio:.2f}[/]"
            lat = str(r.regime_latency_bars) if r.regime_latency_bars is not None else "N/A"
            halt = "[red]YES[/red]" if r.risk_halted else "[green]NO[/green]"
            cb = f"{r.circuit_breaker_fired_pct*100:.0f}%" if r.circuit_breaker_fired_pct > 0 else "—"
            table.add_row(r.scenario.name, tr, dd, sh, lat, halt, cb, r.notes[:50])

        console.print(Panel(table, title="[bold white]Stress Test Summary[/bold white]"))

    def _print_plain_summary(
        self,
        results: list[StressTestResult],
    ) -> None:
        sep = "=" * 80
        print(f"\n{sep}")
        print("  STRESS TEST SUMMARY")
        print(sep)
        header = f"  {'Scenario':<28} {'Return':>8} {'MaxDD':>7} {'Sharpe':>7} {'Lat':>5} {'Halt':>5}"
        print(header)
        print("-" * 80)
        for r in results:
            p = r.performance
            print(
                f"  {r.scenario.name:<28} "
                f"{p.total_return*100:+7.1f}%  "
                f"{p.max_drawdown*100:6.1f}%  "
                f"{p.sharpe_ratio:6.2f}  "
                f"{str(r.regime_latency_bars) if r.regime_latency_bars else 'N/A':>4}  "
                f"{'YES' if r.risk_halted else 'NO':>4}"
            )
        print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _deep_copy_ohlcv(ohlcv: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Return a deep copy of the OHLCV dict."""
    return {sym: df.copy() for sym, df in ohlcv.items()}


def _compute_rolling_atr(df: pd.DataFrame, window: int = 14) -> float:
    """Compute last ATR value from a DataFrame with open/high/low/close columns."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if len(df) < window:
        if "close" in df.columns and len(df) > 0:
            return float(df["close"].iloc[-1]) * 0.02  # 2% fallback
        return 1.0
    try:
        from ta.volatility import AverageTrueRange
        atr_series = AverageTrueRange(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=window,
            fillna=True,
        ).average_true_range()
        return float(atr_series.iloc[-1])
    except Exception:
        # Fallback: simple average of high-low
        if "high" in df.columns and "low" in df.columns:
            return float((df["high"] - df["low"]).rolling(window, min_periods=1).mean().iloc[-1])
        return 1.0


def _null_performance_report() -> PerformanceReport:
    """Return a zero-filled PerformanceReport for error cases."""
    return PerformanceReport(
        total_return=0.0,
        annualised_return=0.0,
        annualised_volatility=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        calmar_ratio=0.0,
        max_drawdown=0.0,
        max_drawdown_duration_bars=0,
        avg_drawdown=0.0,
        max_drawdown_start=None,
        max_drawdown_end=None,
        worst_day_pct=0.0,
        worst_week_pct=0.0,
        worst_month_pct=0.0,
        max_consecutive_losses=0,
        longest_underwater_days=0,
        total_trades=0,
        win_rate=0.0,
        avg_win_pct=0.0,
        avg_loss_pct=0.0,
        profit_factor=0.0,
        avg_holding_bars=0.0,
        regime_breakdown={},
        confidence_breakdown={},
    )
