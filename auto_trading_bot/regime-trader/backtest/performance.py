"""
performance.py — Backtest performance analytics.

Responsibilities:
  - Compute Sharpe, Sortino, Calmar ratios from an equity curve.
  - Calculate maximum drawdown, drawdown duration, and recovery time.
  - Break down returns and win rates by HMM regime label.
  - Compare strategy returns against a buy-and-hold benchmark.
  - Generate a Rich-formatted performance report (tables + panels).
  - Save CSV outputs: equity_curve.csv, trade_log.csv, regime_history.csv.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PerformanceReport dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PerformanceReport:
    """Aggregated performance metrics for a backtest run."""

    # Returns
    total_return: float
    annualised_return: float
    annualised_volatility: float

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Drawdown
    max_drawdown: float
    max_drawdown_duration_bars: int
    avg_drawdown: float
    max_drawdown_start: Optional[pd.Timestamp]
    max_drawdown_end: Optional[pd.Timestamp]

    # Worst cases
    worst_day_pct: float
    worst_week_pct: float
    worst_month_pct: float
    max_consecutive_losses: int
    longest_underwater_days: int

    # Trade stats
    total_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    avg_holding_bars: float

    # Regime breakdown: label → {pct_time, return_contribution, avg_trade_pnl, win_rate, sharpe}
    regime_breakdown: dict[str, dict]

    # Confidence buckets: "<50%","50-60%","60-70%","70%+" → {trades, sharpe, win_rate, avg_pnl}
    confidence_breakdown: dict[str, dict]

    # Benchmark
    benchmark_total_return: Optional[float] = None
    benchmark_sharpe: Optional[float] = None
    alpha: Optional[float] = None

    # Random baseline
    random_mean_return: Optional[float] = None
    random_std_return: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# PerformanceAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class PerformanceAnalyzer:
    """
    Computes performance metrics from a backtest equity curve and trade ledger.

    Usage::

        analyzer = PerformanceAnalyzer(risk_free_rate=0.045)
        report = analyzer.compute(equity_curve, trade_ledger)
        analyzer.print_report(report, title="Walk-Forward Backtest")
    """

    def __init__(self, risk_free_rate: float = 0.045) -> None:
        self.risk_free_rate = risk_free_rate

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def compute(
        self,
        equity_curve: pd.Series,
        trade_ledger: pd.DataFrame,
        benchmark: Optional[pd.Series] = None,
        regime_history: Optional[pd.DataFrame] = None,
    ) -> PerformanceReport:
        """
        Compute all metrics and return a PerformanceReport.

        Parameters
        ----------
        equity_curve : pd.Series
            DatetimeIndex → portfolio equity values.
        trade_ledger : pd.DataFrame
            DataFrame with columns including: realised_pnl, realised_pnl_pct,
            regime_at_entry, regime_confidence, timestamp.
        benchmark : pd.Series, optional
            DatetimeIndex → price series for benchmark comparison.
        regime_history : pd.DataFrame, optional
            Columns: date, regime, confidence, is_confirmed.
        """
        returns = equity_curve.pct_change().dropna()

        # Returns
        total_return = float((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0) if len(equity_curve) > 1 else 0.0
        n_bars = len(equity_curve)
        years = n_bars / 252.0
        ann_return = float((1.0 + total_return) ** (1.0 / max(years, 1e-9)) - 1.0)
        ann_vol = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0.0

        # Risk-adjusted
        sharpe = self.sharpe_ratio(returns)
        sortino = self.sortino_ratio(returns)
        calmar = self.calmar_ratio(returns)

        # Drawdown
        max_dd, max_dd_dur = self.max_drawdown(equity_curve)
        avg_dd = self._avg_drawdown(equity_curve)
        dd_start, dd_end = self._drawdown_dates(equity_curve)

        # Worst cases
        worst_day = float(returns.min()) if len(returns) > 0 else 0.0

        weekly_returns = equity_curve.resample("W").last().pct_change().dropna()
        worst_week = float(weekly_returns.min()) if len(weekly_returns) > 0 else 0.0

        monthly_returns = equity_curve.resample("ME").last().pct_change().dropna()
        worst_month = float(monthly_returns.min()) if len(monthly_returns) > 0 else 0.0

        max_consec_losses = self._max_consecutive_losses(returns)
        longest_underwater = self._longest_underwater(equity_curve)

        # Trade stats
        total_trades, win_rate, avg_win, avg_loss, profit_factor, avg_holding = (
            self._trade_stats(trade_ledger)
        )

        # Regime breakdown
        reg_breakdown = self.regime_breakdown(trade_ledger)

        # Confidence breakdown
        conf_breakdown = self.confidence_breakdown(trade_ledger)

        # Benchmark
        bm_total, bm_sharpe, alpha = None, None, None
        if benchmark is not None:
            bm_total, bm_sharpe, alpha = self.benchmark_metrics(equity_curve, benchmark)

        report = PerformanceReport(
            total_return=total_return,
            annualised_return=ann_return,
            annualised_volatility=ann_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_duration_bars=max_dd_dur,
            avg_drawdown=avg_dd,
            max_drawdown_start=dd_start,
            max_drawdown_end=dd_end,
            worst_day_pct=worst_day,
            worst_week_pct=worst_week,
            worst_month_pct=worst_month,
            max_consecutive_losses=max_consec_losses,
            longest_underwater_days=longest_underwater,
            total_trades=total_trades,
            win_rate=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            profit_factor=profit_factor,
            avg_holding_bars=avg_holding,
            regime_breakdown=reg_breakdown,
            confidence_breakdown=conf_breakdown,
            benchmark_total_return=bm_total,
            benchmark_sharpe=bm_sharpe,
            alpha=alpha,
        )
        return report

    def print_report(
        self,
        report: PerformanceReport,
        title: str = "Backtest Results",
    ) -> None:
        """
        Print a formatted performance report to stdout using Rich.

        Falls back to plain text if Rich is not installed.
        """
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel
            from rich import box
            self._print_rich(report, title)
        except ImportError:
            self._print_plain(report, title)

    def save_csv(self, result: "BacktestResult", output_dir: Path) -> None:
        """
        Save equity curve, trade ledger, and regime history to CSV.

        Parameters
        ----------
        result : BacktestResult
        output_dir : Path
            Directory to save files into (created if it doesn't exist).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        equity_path = output_dir / "equity_curve.csv"
        result.equity_curve.to_csv(equity_path, header=["equity"])
        logger.info("Saved equity curve to %s", equity_path)

        if len(result.trade_ledger) > 0:
            trade_path = output_dir / "trade_log.csv"
            result.trade_ledger.to_csv(trade_path, index=False)
            logger.info("Saved trade ledger to %s", trade_path)

        if hasattr(result, "regime_history") and result.regime_history is not None and len(result.regime_history) > 0:
            regime_path = output_dir / "regime_history.csv"
            result.regime_history.to_csv(regime_path, index=False)
            logger.info("Saved regime history to %s", regime_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Individual metric methods
    # ─────────────────────────────────────────────────────────────────────────

    def sharpe_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 252,
    ) -> float:
        """
        Annualised Sharpe ratio.

        sharpe = (mean(returns) - rf_daily) / std(returns) * sqrt(ppy)
        """
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        rf_daily = self.risk_free_rate / periods_per_year
        excess = returns - rf_daily
        return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))

    def sortino_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 252,
    ) -> float:
        """
        Annualised Sortino ratio.

        Downside deviation = std of returns that are negative (< 0).
        """
        if len(returns) < 2:
            return 0.0
        rf_daily = self.risk_free_rate / periods_per_year
        excess = returns - rf_daily
        downside = returns[returns < 0]
        if len(downside) < 2:
            return float(excess.mean() * periods_per_year) if excess.mean() > 0 else 0.0
        downside_std = float(downside.std())
        if downside_std == 0:
            return 0.0
        return float(excess.mean() / downside_std * np.sqrt(periods_per_year))

    def calmar_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 252,
    ) -> float:
        """
        Annualised return divided by maximum drawdown.

        calmar = CAGR / |max_drawdown|
        """
        if len(returns) < 2:
            return 0.0
        # Reconstruct equity for drawdown
        equity = (1 + returns).cumprod()
        equity = equity * 100  # scale doesn't matter
        max_dd, _ = self.max_drawdown(equity)
        if max_dd == 0.0:
            return 0.0
        n_years = len(returns) / periods_per_year
        cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / max(n_years, 1e-9)) - 1.0)
        return float(cagr / abs(max_dd))

    def max_drawdown(
        self,
        equity: pd.Series,
    ) -> tuple[float, int]:
        """
        Return (max_drawdown_fraction, max_drawdown_duration_in_bars).

        Drawdown fraction is expressed as a positive number (e.g. 0.15 = 15%).
        """
        if len(equity) < 2:
            return 0.0, 0

        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max

        max_dd = float(abs(drawdown.min()))

        # Duration: measure bars from peak to trough of the worst drawdown
        trough_idx = int(drawdown.idxmin()) if hasattr(drawdown.idxmin(), '__int__') else drawdown.index.get_loc(drawdown.idxmin())
        # Find the preceding peak
        try:
            trough_loc = drawdown.index.get_loc(drawdown.idxmin())
        except Exception:
            trough_loc = 0

        # Peak is where cummax was last equal to cummax at the trough
        peak_loc = 0
        peak_val = float(rolling_max.iloc[trough_loc])
        for i in range(trough_loc, -1, -1):
            if abs(float(equity.iloc[i]) - peak_val) < 1e-6:
                peak_loc = i
                break

        duration = trough_loc - peak_loc

        return max_dd, max(duration, 0)

    def regime_breakdown(
        self,
        trade_ledger: pd.DataFrame,
    ) -> dict[str, dict]:
        """
        Compute regime-level stats grouped by regime_at_entry.

        Returns dict: label → {pct_time, return_contribution, avg_trade_pnl, win_rate, sharpe}
        """
        if trade_ledger is None or len(trade_ledger) == 0:
            return {}
        if "regime_at_entry" not in trade_ledger.columns:
            return {}

        result = {}
        total_trades = len(trade_ledger)
        total_pnl = trade_ledger.get("realised_pnl", pd.Series(dtype=float)).sum()

        for label, group in trade_ledger.groupby("regime_at_entry"):
            n = len(group)
            pnl_col = group.get("realised_pnl", pd.Series(dtype=float)) if "realised_pnl" in group.columns else pd.Series(0.0, index=group.index)
            pnl_pct_col = group.get("realised_pnl_pct", pd.Series(dtype=float)) if "realised_pnl_pct" in group.columns else pd.Series(0.0, index=group.index)

            wins = (pnl_col > 0).sum()
            wr = float(wins / n) if n > 0 else 0.0
            avg_pnl = float(pnl_col.mean()) if n > 0 else 0.0
            pct_time = float(n / total_trades) if total_trades > 0 else 0.0
            return_contrib = float(pnl_col.sum() / total_pnl) if total_pnl != 0 else 0.0

            # Sharpe on pnl_pct returns
            if len(pnl_pct_col) > 1 and pnl_pct_col.std() > 0:
                sharpe = float(pnl_pct_col.mean() / pnl_pct_col.std() * np.sqrt(252))
            else:
                sharpe = 0.0

            result[str(label)] = {
                "pct_time": pct_time,
                "return_contribution": return_contrib,
                "avg_trade_pnl": avg_pnl,
                "win_rate": wr,
                "sharpe": sharpe,
                "n_trades": n,
            }
        return result

    def confidence_breakdown(
        self,
        trade_ledger: pd.DataFrame,
    ) -> dict[str, dict]:
        """
        Bucket trades by regime_confidence into: "<50%","50-60%","60-70%","70%+".

        Returns dict: bucket → {trades, sharpe, win_rate, avg_pnl}
        """
        buckets = {
            "<50%": (0.0, 0.5),
            "50-60%": (0.5, 0.6),
            "60-70%": (0.6, 0.7),
            "70%+": (0.7, 1.01),
        }

        if trade_ledger is None or len(trade_ledger) == 0:
            return {b: {"trades": 0, "sharpe": 0.0, "win_rate": 0.0, "avg_pnl": 0.0} for b in buckets}

        if "regime_confidence" not in trade_ledger.columns:
            return {b: {"trades": 0, "sharpe": 0.0, "win_rate": 0.0, "avg_pnl": 0.0} for b in buckets}

        result = {}
        conf = trade_ledger["regime_confidence"]
        pnl_col = trade_ledger.get("realised_pnl", pd.Series(0.0, index=trade_ledger.index)) if "realised_pnl" in trade_ledger.columns else pd.Series(0.0, index=trade_ledger.index)
        pnl_pct_col = trade_ledger.get("realised_pnl_pct", pd.Series(0.0, index=trade_ledger.index)) if "realised_pnl_pct" in trade_ledger.columns else pd.Series(0.0, index=trade_ledger.index)

        for bucket, (lo, hi) in buckets.items():
            mask = (conf >= lo) & (conf < hi)
            grp_pnl = pnl_col[mask]
            grp_pnl_pct = pnl_pct_col[mask]
            n = mask.sum()
            if n == 0:
                result[bucket] = {"trades": 0, "sharpe": 0.0, "win_rate": 0.0, "avg_pnl": 0.0}
            else:
                wins = (grp_pnl > 0).sum()
                wr = float(wins / n)
                avg_pnl = float(grp_pnl.mean())
                if len(grp_pnl_pct) > 1 and grp_pnl_pct.std() > 0:
                    sharpe = float(grp_pnl_pct.mean() / grp_pnl_pct.std() * np.sqrt(252))
                else:
                    sharpe = 0.0
                result[bucket] = {
                    "trades": int(n),
                    "sharpe": sharpe,
                    "win_rate": wr,
                    "avg_pnl": avg_pnl,
                }

        return result

    def benchmark_metrics(
        self,
        strategy_equity: pd.Series,
        benchmark_prices: pd.Series,
    ) -> tuple[float, float, float]:
        """
        Return (benchmark_total_return, benchmark_sharpe, alpha).

        Alpha = strategy annualised return − benchmark annualised return.
        """
        if benchmark_prices is None or len(benchmark_prices) < 2:
            return 0.0, 0.0, 0.0

        bm_returns = benchmark_prices.pct_change().dropna()
        bm_total = float((benchmark_prices.iloc[-1] / benchmark_prices.iloc[0]) - 1.0)
        bm_sharpe = self.sharpe_ratio(bm_returns)

        # Strategy annualised return
        if len(strategy_equity) > 1:
            n_years = len(strategy_equity) / 252.0
            strat_total = float((strategy_equity.iloc[-1] / strategy_equity.iloc[0]) - 1.0)
            strat_ann = float((1.0 + strat_total) ** (1.0 / max(n_years, 1e-9)) - 1.0)
        else:
            strat_ann = 0.0

        n_years_bm = len(bm_returns) / 252.0
        bm_ann = float((1.0 + bm_total) ** (1.0 / max(n_years_bm, 1e-9)) - 1.0)

        alpha = strat_ann - bm_ann
        return float(bm_total), float(bm_sharpe), float(alpha)

    def run_random_baseline(
        self,
        equity_curve: pd.Series,
        trade_ledger: pd.DataFrame,
        n_seeds: int = 100,
    ) -> tuple[float, float]:
        """
        Reshuffle trade order n_seeds times; compute distribution of total returns.

        Returns (mean_return, std_return) across simulations.
        """
        if trade_ledger is None or len(trade_ledger) == 0:
            return 0.0, 0.0
        if "realised_pnl" not in trade_ledger.columns:
            return 0.0, 0.0

        initial_equity = float(equity_curve.iloc[0]) if len(equity_curve) > 0 else 100_000.0
        pnls = trade_ledger["realised_pnl"].values.copy()

        rng_returns = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            shuffled = rng.permutation(pnls)
            final_equity = initial_equity + shuffled.sum()
            rng_returns.append((final_equity / initial_equity) - 1.0)

        return float(np.mean(rng_returns)), float(np.std(rng_returns))

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _avg_drawdown(self, equity: pd.Series) -> float:
        """Average drawdown (mean of all drawdown values)."""
        if len(equity) < 2:
            return 0.0
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        # Only count drawdown bars (where drawdown < 0)
        dd_bars = drawdown[drawdown < 0]
        return float(abs(dd_bars.mean())) if len(dd_bars) > 0 else 0.0

    def _drawdown_dates(
        self,
        equity: pd.Series,
    ) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Return (start, end) timestamps of the worst drawdown."""
        if len(equity) < 2:
            return None, None
        try:
            rolling_max = equity.cummax()
            drawdown = (equity - rolling_max) / rolling_max
            trough_ts = drawdown.idxmin()
            # Find peak: last point before trough where equity == rolling_max at trough
            trough_val = rolling_max.loc[trough_ts]
            pre_trough = equity.loc[:trough_ts]
            peaks = pre_trough[abs(pre_trough - trough_val) < 1e-6]
            peak_ts = peaks.index[-1] if len(peaks) > 0 else equity.index[0]
            return peak_ts, trough_ts
        except Exception:
            return None, None

    def _max_consecutive_losses(self, returns: pd.Series) -> int:
        """Return the maximum number of consecutive negative return bars."""
        if len(returns) == 0:
            return 0
        max_run = 0
        current_run = 0
        for r in returns:
            if r < 0:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        return max_run

    def _longest_underwater(self, equity: pd.Series) -> int:
        """
        Return the longest period (in bars) the equity was below a previous peak.
        """
        if len(equity) < 2:
            return 0
        rolling_max = equity.cummax()
        underwater = equity < rolling_max
        max_run = 0
        current_run = 0
        for uw in underwater:
            if uw:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        return max_run

    def _trade_stats(
        self,
        trade_ledger: pd.DataFrame,
    ) -> tuple[int, float, float, float, float, float]:
        """
        Return (total_trades, win_rate, avg_win_pct, avg_loss_pct, profit_factor, avg_holding_bars).
        """
        if trade_ledger is None or len(trade_ledger) == 0:
            return 0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Only count SELL trades (completed trades) or all if no side column
        if "side" in trade_ledger.columns:
            completed = trade_ledger[trade_ledger["side"] == "SELL"]
        else:
            completed = trade_ledger

        n = len(completed)
        if n == 0:
            return 0, 0.0, 0.0, 0.0, 0.0, 0.0

        pnl_col = completed["realised_pnl"] if "realised_pnl" in completed.columns else pd.Series(0.0, index=completed.index)
        pnl_pct_col = completed["realised_pnl_pct"] if "realised_pnl_pct" in completed.columns else pd.Series(0.0, index=completed.index)

        wins = pnl_col[pnl_col > 0]
        losses = pnl_col[pnl_col < 0]

        win_rate = float(len(wins) / n)
        avg_win = float(pnl_pct_col[pnl_col > 0].mean()) if len(wins) > 0 else 0.0
        avg_loss = float(pnl_pct_col[pnl_col < 0].mean()) if len(losses) > 0 else 0.0
        gross_profit = float(wins.sum())
        gross_loss = float(abs(losses.sum()))
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        avg_holding = 0.0  # Allocation-based backtest doesn't track per-trade holding periods

        return n, win_rate, avg_win, avg_loss, profit_factor, avg_holding

    # ─────────────────────────────────────────────────────────────────────────
    # Rich printing
    # ─────────────────────────────────────────────────────────────────────────

    def _print_rich(
        self,
        report: PerformanceReport,
        title: str,
    ) -> None:
        """Print report using Rich tables."""
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box

        console = Console()

        # ── Summary panel ─────────────────────────────────────────────────────
        summary_table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
        summary_table.add_column("Metric", style="bold")
        summary_table.add_column("Value", justify="right")

        def _pct(v: float) -> str:
            return f"[green]{v*100:.2f}%[/green]" if v >= 0 else f"[red]{v*100:.2f}%[/red]"

        def _num(v: float, decimals: int = 2) -> str:
            return f"[green]{v:.{decimals}f}[/green]" if v >= 0 else f"[red]{v:.{decimals}f}[/red]"

        summary_table.add_row("Total Return", _pct(report.total_return))
        summary_table.add_row("Annualised Return", _pct(report.annualised_return))
        summary_table.add_row("Annualised Volatility", _pct(report.annualised_volatility))
        summary_table.add_row("Sharpe Ratio", _num(report.sharpe_ratio))
        summary_table.add_row("Sortino Ratio", _num(report.sortino_ratio))
        summary_table.add_row("Calmar Ratio", _num(report.calmar_ratio))
        summary_table.add_row("Max Drawdown", f"[red]{report.max_drawdown*100:.2f}%[/red]")
        summary_table.add_row("Max DD Duration", f"{report.max_drawdown_duration_bars} bars")
        summary_table.add_row("Avg Drawdown", f"[red]{report.avg_drawdown*100:.2f}%[/red]")

        console.print(Panel(summary_table, title=f"[bold white]{title} — Summary[/bold white]"))

        # ── Trade stats table ─────────────────────────────────────────────────
        trade_table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
        trade_table.add_column("Metric", style="bold")
        trade_table.add_column("Value", justify="right")
        trade_table.add_row("Total Trades", str(report.total_trades))
        trade_table.add_row("Win Rate", _pct(report.win_rate))
        trade_table.add_row("Avg Win", _pct(report.avg_win_pct))
        trade_table.add_row("Avg Loss", _pct(report.avg_loss_pct))
        trade_table.add_row("Profit Factor", _num(report.profit_factor))
        trade_table.add_row("Avg Holding Bars", f"{report.avg_holding_bars:.1f}")
        console.print(Panel(trade_table, title="[bold white]Trade Statistics[/bold white]"))

        # ── Worst-case table ──────────────────────────────────────────────────
        wc_table = Table(box=box.SIMPLE, show_header=True, header_style="bold red")
        wc_table.add_column("Metric", style="bold")
        wc_table.add_column("Value", justify="right")
        wc_table.add_row("Worst Day", f"[red]{report.worst_day_pct*100:.2f}%[/red]")
        wc_table.add_row("Worst Week", f"[red]{report.worst_week_pct*100:.2f}%[/red]")
        wc_table.add_row("Worst Month", f"[red]{report.worst_month_pct*100:.2f}%[/red]")
        wc_table.add_row("Max Consec. Losses", str(report.max_consecutive_losses))
        wc_table.add_row("Longest Underwater", f"{report.longest_underwater_days} bars")
        console.print(Panel(wc_table, title="[bold white]Worst-Case Scenarios[/bold white]"))

        # ── Regime breakdown table ────────────────────────────────────────────
        if report.regime_breakdown:
            reg_table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
            reg_table.add_column("Regime")
            reg_table.add_column("% Time", justify="right")
            reg_table.add_column("Win Rate", justify="right")
            reg_table.add_column("Avg PnL $", justify="right")
            reg_table.add_column("Ret. Contrib", justify="right")
            reg_table.add_column("Sharpe", justify="right")
            for label, stats in sorted(report.regime_breakdown.items()):
                reg_table.add_row(
                    label,
                    f"{stats.get('pct_time', 0)*100:.1f}%",
                    f"{stats.get('win_rate', 0)*100:.1f}%",
                    _num(stats.get("avg_trade_pnl", 0.0)),
                    f"{stats.get('return_contribution', 0)*100:.1f}%",
                    _num(stats.get("sharpe", 0.0)),
                )
            console.print(Panel(reg_table, title="[bold white]Regime Breakdown[/bold white]"))

        # ── Confidence breakdown table ─────────────────────────────────────────
        if report.confidence_breakdown:
            conf_table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
            conf_table.add_column("Confidence Bucket")
            conf_table.add_column("Trades", justify="right")
            conf_table.add_column("Win Rate", justify="right")
            conf_table.add_column("Avg PnL $", justify="right")
            conf_table.add_column("Sharpe", justify="right")
            for bucket, stats in report.confidence_breakdown.items():
                conf_table.add_row(
                    bucket,
                    str(stats.get("trades", 0)),
                    f"{stats.get('win_rate', 0)*100:.1f}%",
                    _num(stats.get("avg_pnl", 0.0)),
                    _num(stats.get("sharpe", 0.0)),
                )
            console.print(Panel(conf_table, title="[bold white]Confidence Breakdown[/bold white]"))

        # ── Benchmark table ───────────────────────────────────────────────────
        if report.benchmark_total_return is not None:
            bm_table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
            bm_table.add_column("Metric", style="bold")
            bm_table.add_column("Value", justify="right")
            bm_table.add_row("Benchmark Total Return", _pct(report.benchmark_total_return))
            bm_table.add_row("Benchmark Sharpe", _num(report.benchmark_sharpe or 0.0))
            bm_table.add_row("Alpha vs Benchmark", _pct(report.alpha or 0.0))
            if report.random_mean_return is not None:
                bm_table.add_row("Random Baseline Mean", _pct(report.random_mean_return))
                bm_table.add_row("Random Baseline Std", _pct(report.random_std_return or 0.0))
            console.print(Panel(bm_table, title="[bold white]Benchmark Comparison[/bold white]"))

    def _print_plain(
        self,
        report: PerformanceReport,
        title: str,
    ) -> None:
        """Plain-text fallback when Rich is not installed."""
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  {title}")
        print(sep)
        print(f"  Total Return:          {report.total_return*100:+.2f}%")
        print(f"  Annualised Return:     {report.annualised_return*100:+.2f}%")
        print(f"  Annualised Volatility: {report.annualised_volatility*100:.2f}%")
        print(f"  Sharpe Ratio:          {report.sharpe_ratio:.3f}")
        print(f"  Sortino Ratio:         {report.sortino_ratio:.3f}")
        print(f"  Calmar Ratio:          {report.calmar_ratio:.3f}")
        print(f"  Max Drawdown:          {report.max_drawdown*100:.2f}%")
        print(f"  Max DD Duration:       {report.max_drawdown_duration_bars} bars")
        print(f"  Avg Drawdown:          {report.avg_drawdown*100:.2f}%")
        print(f"  Worst Day:             {report.worst_day_pct*100:.2f}%")
        print(f"  Worst Week:            {report.worst_week_pct*100:.2f}%")
        print(f"  Worst Month:           {report.worst_month_pct*100:.2f}%")
        print(f"  Max Consec. Losses:    {report.max_consecutive_losses}")
        print(f"  Longest Underwater:    {report.longest_underwater_days} bars")
        print(sep)
        print(f"  Total Trades:          {report.total_trades}")
        print(f"  Win Rate:              {report.win_rate*100:.1f}%")
        print(f"  Avg Win:               {report.avg_win_pct*100:+.2f}%")
        print(f"  Avg Loss:              {report.avg_loss_pct*100:+.2f}%")
        print(f"  Profit Factor:         {report.profit_factor:.2f}")
        if report.benchmark_total_return is not None:
            print(sep)
            print(f"  Benchmark Return:      {report.benchmark_total_return*100:+.2f}%")
            print(f"  Benchmark Sharpe:      {report.benchmark_sharpe:.3f}")
            print(f"  Alpha:                 {(report.alpha or 0.0)*100:+.2f}%")
        if report.regime_breakdown:
            print(sep)
            print("  Regime Breakdown:")
            for label, stats in sorted(report.regime_breakdown.items()):
                print(
                    f"    {label:<18} "
                    f"time={stats.get('pct_time',0)*100:.0f}%  "
                    f"wr={stats.get('win_rate',0)*100:.0f}%  "
                    f"sharpe={stats.get('sharpe',0):.2f}"
                )
        print(sep + "\n")
