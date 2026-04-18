"""
dashboard.py — Terminal-based live trading dashboard powered by ``rich``.

Layout (top → bottom, refresh every N seconds):

  ┌─ REGIME ────────────────────────────────────────────────────────┐
  │  [BULL]  72%  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░  Stable: 14 bars  Flicker: 1/20 │
  ├─ PORTFOLIO ─────────────────────────────────────────────────────┤
  │  Equity: $105,230   Daily: +$340 (+0.32%)                        │
  │  Allocation: 95%    Leverage: 1.25×   Positions: 2/5            │
  ├─ POSITIONS ─────────────────────────────────────────────────────┤
  │  Symbol  Side  Price     PnL%    Stop      Held  Regime          │
  │  SPY     LONG  $520.30  +1.20%  $508.00   3h    BULL             │
  ├─ RECENT SIGNALS ────────────────────────────────────────────────┤
  │  14:30  SPY  Rebalance 60%→95%  Low vol  APPROVED               │
  ├─ RISK STATUS ───────────────────────────────────────────────────┤
  │  Daily DD    0.3% / 3.0%   ████░░░░░░░░░░░░░░░░  10%  NORMAL    │
  │  Weekly DD   1.2% / 7.0%   ████████░░░░░░░░░░░░  17%  NORMAL    │
  │  Peak DD     2.1% / 10.0%  █████░░░░░░░░░░░░░░░  21%  NORMAL    │
  ├─ SYSTEM ────────────────────────────────────────────────────────┤
  │  Data: ●  API: 23ms  HMM: 2d ago  Mode: PAPER  14:30:05 UTC     │
  └─────────────────────────────────────────────────────────────────┘

Color coding:
  - Regime: green (bull), yellow (neutral/transition), red (bear/crash)
  - P&L:    green (positive), red (negative)
  - Risk bars: green (<50% of limit), yellow (50-80%), red (>80%)
  - Data/API: bright_green (OK), red (down)
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Optional

from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Type hints only — import lazily to avoid hard circular deps at module load
# ---------------------------------------------------------------------------
try:
    from broker.position_tracker import PositionTracker, TrackedPosition
    _HAS_TRACKER = True
except ImportError:
    _HAS_TRACKER = False

try:
    from core.hmm_engine import RegimeState
    _HAS_REGIME = True
except ImportError:
    _HAS_REGIME = False

try:
    from core.risk_manager import RiskManager, TradingState, BreakerType, PortfolioState
    _HAS_RISK = True
except ImportError:
    _HAS_RISK = False


# ---------------------------------------------------------------------------
# Colour maps
# ---------------------------------------------------------------------------

_REGIME_COLOURS: dict[str, str] = {
    "CRASH":       "bold red",
    "STRONG_BEAR": "red",
    "BEAR":        "red",
    "WEAK_BEAR":   "dark_orange",
    "NEUTRAL":     "yellow",
    "WEAK_BULL":   "green",
    "BULL":        "bright_green",
    "STRONG_BULL": "bold bright_green",
    "EUPHORIA":    "bold magenta",
    "UNKNOWN":     "dim white",
}

_STATE_COLOURS: dict[str, str] = {
    "NORMAL":  "bright_green",
    "REDUCED": "yellow",
    "HALTED":  "bold red",
}

_BREAKER_COLOURS: dict[str, str] = {
    "NONE":          "bright_green",
    "DAILY_REDUCE":  "yellow",
    "DAILY_HALT":    "red",
    "WEEKLY_REDUCE": "dark_orange",
    "WEEKLY_HALT":   "red",
    "PEAK_DD_HALT":  "bold red",
    "DAILY_TRADES":  "orange3",
}

_BAR_WIDTH = 20   # character width of the DD progress bars


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _regime_color(label: str) -> str:
    return _REGIME_COLOURS.get(label.upper(), "white")


def _dd_bar(current: float, limit: float, width: int = _BAR_WIDTH) -> Text:
    """
    Return a coloured progress bar for a drawdown metric.

    Example:  ████░░░░░░░░░░░░░░░░  21%
    """
    ratio   = min(current / limit, 1.0) if limit > 0 else 0.0
    filled  = max(0, int(ratio * width))
    empty   = width - filled

    if ratio < 0.5:
        color = "bright_green"
    elif ratio < 0.80:
        color = "yellow"
    else:
        color = "bold red"

    bar = Text()
    bar.append("█" * filled, style=color)
    bar.append("░" * empty,  style="dim")
    bar.append(f"  {ratio * 100:.0f}%", style=color)
    return bar


def _pnl_text(value: float, pct: Optional[float] = None) -> Text:
    """Colour a P&L value: green for positive, red for negative."""
    color  = "bright_green" if value >= 0 else "red"
    prefix = "+" if value >= 0 else ""
    t = Text(f"{prefix}${value:,.2f}", style=color)
    if pct is not None:
        sign = "+" if pct >= 0 else ""
        t.append(f"  ({sign}{pct:.2f}%)", style=color)
    return t


def _age_str(dt: Optional[datetime]) -> str:
    """Return a human-readable age string ('2d ago', '3h ago', 'just now')."""
    if dt is None:
        return "never"
    delta = datetime.now() - dt
    s = int(delta.total_seconds())
    if s < 60:
        return "just now"
    if s < 3600:
        return f"{s // 60}m ago"
    if s < 86400:
        return f"{s // 3600}h ago"
    return f"{s // 86400}d ago"


def _fmt_held(bars_held: int, timeframe: str = "1Day") -> str:
    """Convert bars-held count to a human-readable holding-period string."""
    minutes_per_bar = {"1Min": 1, "5Min": 5, "15Min": 15, "1Hour": 60, "1Day": 390}.get(
        timeframe, 390
    )
    total_min = bars_held * minutes_per_bar
    if total_min < 60:
        return f"{total_min}m"
    if total_min < 1440:
        return f"{total_min // 60}h"
    return f"{total_min // 1440}d"


# ---------------------------------------------------------------------------
# SystemStatus dataclass
# ---------------------------------------------------------------------------

class SystemStatus:
    """Mutable container for live system health fields."""

    __slots__ = (
        "feed_ok", "api_latency_ms", "hmm_last_train",
        "mode", "last_bar_time",
    )

    def __init__(self):
        self.feed_ok:        bool             = True
        self.api_latency_ms: int              = 0
        self.hmm_last_train: Optional[datetime] = None
        self.mode:           str              = "PAPER"
        self.last_bar_time:  Optional[datetime] = None


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

class Dashboard:
    """
    Live terminal dashboard rendered with ``rich.live.Live``.

    The dashboard maintains an internal snapshot of regime, portfolio,
    position, signal, and system data.  Call :meth:`update` from the main
    trading loop to push fresh data; the display auto-refreshes at the
    configured interval.

    Usage::

        dashboard = Dashboard(risk_manager, position_tracker, refresh_seconds=5)
        with dashboard.live():
            while running:
                regime_state = hmm.predict_filtered_next(features)
                dashboard.update(
                    regime_state   = regime_state,
                    recent_signals = last_signals,
                )
                time.sleep(bar_interval)
    """

    def __init__(
        self,
        risk_manager:     "RiskManager",
        position_tracker: "PositionTracker",
        refresh_seconds:  int = 5,
    ) -> None:
        """
        Parameters
        ----------
        risk_manager:
            Live :class:`RiskManager` instance for equity / drawdown data.
        position_tracker:
            Live :class:`PositionTracker` instance for open-position data.
        refresh_seconds:
            Auto-refresh interval in seconds (default 5).
        """
        self._risk     = risk_manager
        self._tracker  = position_tracker
        self.refresh_seconds = refresh_seconds

        # ── Mutable display state ─────────────────────────────────────────────
        self._regime_state: Optional[object] = None
        self._signals:      list             = []
        self._log_lines:    list[str]        = []
        self._system       = SystemStatus()
        self._lock         = threading.Lock()

        # ── Live handle ───────────────────────────────────────────────────────
        self._live: Optional[Live] = None
        self._console = Console()

    # =========================================================================
    # Public API
    # =========================================================================

    @contextmanager
    def live(self):
        """
        Context manager that starts and stops the ``rich.live.Live`` renderer.

        Usage::

            with dashboard.live():
                while running:
                    dashboard.update(regime_state=rs)
                    time.sleep(5)
        """
        live = Live(
            self._render(),
            console         = self._console,
            refresh_per_second = max(0.1, 1.0 / self.refresh_seconds),
            screen          = False,
        )
        self._live = live
        try:
            with live:
                yield live
        finally:
            self._live = None

    def update(
        self,
        regime_state:   Optional[object] = None,
        recent_signals: Optional[list]   = None,
        log_lines:      Optional[list[str]] = None,
    ) -> None:
        """
        Push new data to the dashboard and trigger a redraw.

        Parameters
        ----------
        regime_state:
            Latest :class:`RegimeState` from the HMM engine.
        recent_signals:
            Most recent :class:`Signal` objects (last N).
        log_lines:
            Last N log message strings for the log panel.
        """
        with self._lock:
            if regime_state is not None:
                self._regime_state = regime_state
            if recent_signals is not None:
                self._signals = list(recent_signals)
            if log_lines is not None:
                self._log_lines = list(log_lines)

        if self._live is not None:
            self._live.update(self._render())

    def set_system_status(
        self,
        feed_ok:        Optional[bool]     = None,
        api_latency_ms: Optional[int]      = None,
        hmm_last_train: Optional[datetime] = None,
        mode:           Optional[str]      = None,
        last_bar_time:  Optional[datetime] = None,
    ) -> None:
        """
        Update the SYSTEM panel fields.

        All parameters are optional; pass only what has changed.
        """
        with self._lock:
            if feed_ok        is not None: self._system.feed_ok        = feed_ok
            if api_latency_ms is not None: self._system.api_latency_ms = api_latency_ms
            if hmm_last_train is not None: self._system.hmm_last_train = hmm_last_train
            if mode           is not None: self._system.mode           = mode
            if last_bar_time  is not None: self._system.last_bar_time  = last_bar_time

        if self._live is not None:
            self._live.update(self._render())

    def print_snapshot(self) -> None:
        """Print a one-shot static snapshot to the console (no Live needed)."""
        self._console.print(self._render())

    # =========================================================================
    # Top-level renderer
    # =========================================================================

    def _render(self) -> Group:
        """Compose the full dashboard renderable."""
        with self._lock:
            regime_state = self._regime_state
            signals      = list(self._signals)
            log_lines    = list(self._log_lines)
            system       = self._system       # SystemStatus is not deep-copied

        sections = [
            self._regime_panel(regime_state),
            self._portfolio_panel(),
            self._positions_table(),
            self._signals_table(signals),
            self._risk_panel(),
            self._system_panel(),
        ]
        if log_lines:
            sections.append(self._log_panel(log_lines))

        return Group(*sections)

    # =========================================================================
    # Panel builders
    # =========================================================================

    def _regime_panel(self, regime_state: Optional[object]) -> Panel:
        """
        REGIME: label badge, probability bar, stability count, flicker rate.
        """
        if regime_state is None:
            body = Text("Waiting for first bar…", style="dim italic")
            return Panel(body, title="[bold]REGIME[/]", border_style="dim")

        label       = getattr(regime_state, "label",          "UNKNOWN")
        prob        = getattr(regime_state, "probability",     0.0)
        stable_bars = getattr(regime_state, "consecutive_bars", 0)
        flicker     = getattr(regime_state, "flicker_rate",    0.0)
        flicker_win = 20                        # matches HMMConfig default
        confirmed   = getattr(regime_state, "is_confirmed",    False)
        in_trans    = getattr(regime_state, "in_transition",   False)
        candidate   = getattr(regime_state, "candidate_label", None)

        color   = _regime_color(label)
        prob_pct = int(prob * 100)

        # Confidence bar (20 chars)
        conf_filled = max(0, int(prob * _BAR_WIDTH))
        conf_bar    = Text()
        conf_bar.append(f" [{label}] ", style=f"bold {color} on grey19")
        conf_bar.append(f" {prob_pct}% ", style=color)
        conf_bar.append("▓" * conf_filled,       style=color)
        conf_bar.append("░" * (_BAR_WIDTH - conf_filled), style="dim")

        # Status flags
        if in_trans and candidate:
            status = Text(f"  → {candidate}", style="yellow italic")
        elif not confirmed:
            status = Text("  [pending]", style="dim italic")
        else:
            status = Text("")

        meta = Text()
        meta.append(f"  Stable: {stable_bars} bars", style="dim")
        meta.append(f"  |  Flicker: ")
        flicker_color = "red" if flicker >= 4 else "yellow" if flicker >= 2 else "bright_green"
        meta.append(f"{flicker:.0f}/{flicker_win}", style=flicker_color)

        body = Text.assemble(conf_bar, status, meta)
        border = "bright_green" if label in ("BULL", "STRONG_BULL") else \
                 "red"          if label in ("CRASH", "BEAR", "STRONG_BEAR") else "yellow"
        return Panel(body, title="[bold]REGIME[/]", border_style=border)

    def _portfolio_panel(self) -> Panel:
        """PORTFOLIO: equity, daily P&L, allocation, leverage, position count."""
        portfolio: Optional[object] = None
        if hasattr(self._risk, "_portfolio"):
            portfolio = self._risk._portfolio
        if portfolio is None and hasattr(self._tracker, "_portfolio"):
            portfolio = getattr(self._tracker, "_portfolio", None)

        equity       = float(getattr(portfolio, "equity",        0.0) if portfolio else 0.0)
        cash         = float(getattr(portfolio, "cash",          0.0) if portfolio else 0.0)
        daily_pnl    = float(getattr(portfolio, "daily_pnl",     0.0) if portfolio else 0.0)
        sod_equity   = float(getattr(portfolio, "sod_equity",    equity) if portfolio else equity)
        n_positions  = int(getattr(portfolio, "n_positions",      0)   if portfolio else 0)
        daily_trades = int(getattr(portfolio, "daily_trades",     0)   if portfolio else 0)

        pnl_pct = (daily_pnl / sod_equity * 100) if sod_equity > 0 else 0.0

        # Allocation from tracker
        total_mv    = 0.0
        if hasattr(self._tracker, "get_notional_map"):
            try:
                total_mv = sum(self._tracker.get_notional_map().values())
            except Exception:
                pass
        alloc_pct   = (total_mv / equity * 100) if equity > 0 else 0.0

        # Leverage (infer from positions)
        leverage    = 1.0
        if hasattr(self._risk, "config"):
            leverage = getattr(self._risk.config, "max_leverage", 1.0)

        row1 = Text()
        row1.append(f"Equity: ${equity:>12,.2f}", style="bold white")
        row1.append("    ")
        row1.append("Daily P&L: ")
        row1.append(_pnl_text(daily_pnl, pnl_pct))

        row2 = Text()
        alloc_color = "bright_green" if alloc_pct < 80 else "yellow" if alloc_pct < 95 else "red"
        row2.append(f"Allocation: ")
        row2.append(f"{alloc_pct:.0f}%", style=alloc_color)
        row2.append(f"    Leverage: {leverage:.2f}×    ")
        row2.append(f"Positions: {n_positions}/")
        max_c = getattr(getattr(self._risk, "config", None), "max_concurrent", 5)
        row2.append(str(max_c))
        row2.append(f"    Trades today: {daily_trades}")

        body = Group(row1, row2)
        return Panel(body, title="[bold]PORTFOLIO[/]", border_style="cyan")

    def _positions_table(self) -> Panel:
        """POSITIONS: one row per open position with P&L, stop, held time."""
        table = Table(
            show_header  = True,
            header_style = "bold cyan",
            box          = None,
            padding      = (0, 1),
        )
        table.add_column("Symbol", style="bold white", width=7)
        table.add_column("Side",   style="white",      width=5)
        table.add_column("Entry",  justify="right",    width=9)
        table.add_column("Price",  justify="right",    width=9)
        table.add_column("PnL%",   justify="right",    width=8)
        table.add_column("Stop",   justify="right",    width=9)
        table.add_column("Held",   justify="right",    width=6)
        table.add_column("Regime@Entry", width=12)

        positions: dict = {}
        if hasattr(self._tracker, "get_all_positions"):
            try:
                positions = self._tracker.get_all_positions()
            except Exception:
                pass

        if not positions:
            table.add_row(
                Text("—  no open positions —", style="dim italic"),
                "", "", "", "", "", "", "",
            )
        else:
            for sym, pos in positions.items():
                qty    = getattr(pos, "qty",               0)
                entry  = getattr(pos, "avg_entry_price",   0.0)
                price  = getattr(pos, "current_price",     entry)
                pnl_pct = getattr(pos, "unrealised_pnl_pct", 0.0) * 100
                stop   = getattr(pos, "stop_level",        0.0)
                held   = getattr(pos, "bars_held",         0)
                regime = getattr(pos, "regime_at_entry",   "?")
                side   = getattr(pos, "side",              "long").upper()

                pnl_color = "bright_green" if pnl_pct >= 0 else "red"
                pnl_sign  = "+" if pnl_pct >= 0 else ""
                regime_color = _regime_color(regime)

                table.add_row(
                    sym,
                    side,
                    f"${entry:,.2f}",
                    f"${price:,.2f}",
                    Text(f"{pnl_sign}{pnl_pct:.2f}%", style=pnl_color),
                    f"${stop:,.2f}" if stop > 0 else "—",
                    _fmt_held(held),
                    Text(regime, style=regime_color),
                )

        return Panel(table, title="[bold]POSITIONS[/]", border_style="blue")

    def _signals_table(self, signals: list) -> Panel:
        """RECENT SIGNALS: last N signals with time, symbol, action, regime."""
        table = Table(
            show_header  = True,
            header_style = "bold cyan",
            box          = None,
            padding      = (0, 1),
        )
        table.add_column("Time",   width=7)
        table.add_column("Symbol", width=7)
        table.add_column("Dir",    width=5)
        table.add_column("Size",   justify="right", width=7)
        table.add_column("Entry",  justify="right", width=8)
        table.add_column("Stop",   justify="right", width=8)
        table.add_column("Regime", width=12)
        table.add_column("Strategy", width=16)

        display = signals[-8:] if signals else []

        if not display:
            table.add_row(
                Text("—  no signals this session —", style="dim italic"),
                "", "", "", "", "", "", "",
            )
        else:
            for sig in reversed(display):
                ts_str   = ""
                ts_attr  = getattr(sig, "timestamp", None)
                if ts_attr is not None:
                    try:
                        ts_str = str(ts_attr)[-8:-3]   # HH:MM
                    except Exception:
                        pass

                sym       = getattr(sig, "symbol",           "?")
                direction = getattr(sig, "direction",        "?")
                size_pct  = getattr(sig, "position_size_pct", 0.0)
                entry     = getattr(sig, "entry_price",      0.0)
                stop      = getattr(sig, "stop_loss",        0.0)
                regime    = getattr(sig, "regime_name",      "?")
                strategy  = getattr(sig, "strategy_name",   "?")

                dir_color = "bright_green" if direction == "LONG" else "yellow"
                reg_color = _regime_color(regime)

                table.add_row(
                    ts_str,
                    sym,
                    Text(direction, style=dir_color),
                    f"{size_pct:.0%}",
                    f"${entry:,.2f}",
                    f"${stop:,.2f}" if stop > 0 else "—",
                    Text(regime, style=reg_color),
                    strategy,
                )

        return Panel(table, title="[bold]RECENT SIGNALS[/]", border_style="magenta")

    def _risk_panel(self) -> Panel:
        """RISK STATUS: three colour-coded drawdown bars with limits."""
        portfolio: Optional[object] = None
        if hasattr(self._risk, "_portfolio"):
            portfolio = self._risk._portfolio
        if portfolio is None and hasattr(self._tracker, "_portfolio"):
            portfolio = getattr(self._tracker, "_portfolio", None)

        # Drawdown values
        daily_dd  = float(getattr(portfolio, "daily_drawdown",  0.0) if portfolio else 0.0)
        weekly_dd = float(getattr(portfolio, "weekly_drawdown", 0.0) if portfolio else 0.0)
        peak_dd   = float(getattr(portfolio, "peak_drawdown",   0.0) if portfolio else 0.0)

        # Limits from config
        cfg = getattr(self._risk, "config", None)
        daily_halt  = getattr(cfg, "daily_dd_halt",    0.03)
        weekly_halt = getattr(cfg, "weekly_dd_halt",   0.07)
        peak_halt   = getattr(cfg, "max_dd_from_peak", 0.10)

        # Circuit-breaker state
        cb_state = "NORMAL"
        cb_type  = "NONE"
        if hasattr(self._risk, "_circuit_breaker"):
            cb = self._risk._circuit_breaker
            cb_state = getattr(getattr(cb, "state", None), "value", "NORMAL")
            cb_type  = getattr(getattr(cb, "active_type", None), "value", "NONE")

        state_color = _STATE_COLOURS.get(cb_state, "white")

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label",   width=12, style="dim")
        table.add_column("Values",  width=18)
        table.add_column("Bar",     width=_BAR_WIDTH + 6)

        table.add_row(
            "Daily DD",
            f"{daily_dd:.2%} / {daily_halt:.1%}",
            _dd_bar(daily_dd, daily_halt),
        )
        table.add_row(
            "Weekly DD",
            f"{weekly_dd:.2%} / {weekly_halt:.1%}",
            _dd_bar(weekly_dd, weekly_halt),
        )
        table.add_row(
            "Peak DD",
            f"{peak_dd:.2%} / {peak_halt:.1%}",
            _dd_bar(peak_dd, peak_halt),
        )

        status_line = Text()
        status_line.append("\nState: ")
        status_line.append(cb_state, style=f"bold {state_color}")
        if cb_type != "NONE":
            status_line.append(f"  ({cb_type})", style=_BREAKER_COLOURS.get(cb_type, "white"))

        body = Group(table, status_line)
        border = "red" if cb_state == "HALTED" else "yellow" if cb_state == "REDUCED" else "green"
        return Panel(body, title="[bold]RISK STATUS[/]", border_style=border)

    def _system_panel(self) -> Panel:
        """SYSTEM: data feed indicator, API latency, HMM age, mode, wall clock."""
        sys = self._system

        # Data feed indicator
        feed_dot   = Text("●", style="bright_green bold") if sys.feed_ok else Text("●", style="red bold")
        feed_label = Text(" LIVE" if sys.feed_ok else " DOWN", style="bright_green" if sys.feed_ok else "red")

        # API latency
        lat_color = "bright_green" if sys.api_latency_ms < 100 \
               else "yellow"       if sys.api_latency_ms < 300 \
               else "red"
        lat_text = Text(f"{sys.api_latency_ms}ms", style=lat_color)

        # HMM age
        hmm_age = Text(_age_str(sys.hmm_last_train), style="dim")

        # Mode badge
        mode_color = "yellow" if sys.mode == "PAPER" else "bold red"
        mode_text  = Text(sys.mode, style=mode_color)

        # Wall clock UTC
        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

        body = Text()
        body.append("Data: ")
        body.append_text(feed_dot)
        body.append_text(feed_label)
        body.append("    API: ")
        body.append_text(lat_text)
        body.append("    HMM: ")
        body.append_text(hmm_age)
        body.append("    Mode: ")
        body.append_text(mode_text)
        body.append(f"    {now_str}", style="dim")

        return Panel(body, title="[bold]SYSTEM[/]", border_style="dim")

    def _log_panel(self, log_lines: Optional[list[str]]) -> Panel:
        """LOG: last N log message strings in a scrollable panel."""
        if not log_lines:
            body = Text("—  no log messages yet —", style="dim italic")
        else:
            lines = log_lines[-10:]
            body  = Text("\n".join(lines), style="dim")
        return Panel(body, title="[bold]LOG[/]", border_style="dim")

    def _build_layout(self) -> "Layout":  # type: ignore[override]
        """Kept for API compatibility with the Phase-1 stub signature."""
        from rich.layout import Layout
        layout = Layout()
        layout.update(self._render())
        return layout
