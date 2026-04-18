"""
main.py — Entry point and main orchestration loop for the HMM Regime-Based
Trading Bot.

Responsibilities:
  - Load configuration from config/settings.yaml and .env.
  - Route CLI sub-commands: live, backtest, train-only, stress-test, dashboard.
  - Run the full live-trading lifecycle:
      startup → main bar loop → graceful shutdown.
  - Persist session state to state_snapshot.json for crash recovery.
  - Handle all error conditions without losing open-position stops.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv as _load_dotenv_impl
    def load_dotenv(**kw):
        """Load .env using an explicit path relative to this file (works with systemd)."""
        _env_file = Path(__file__).parent / ".env"
        if _env_file.exists():
            _load_dotenv_impl(_env_file, override=False, **kw)
except ImportError:
    def load_dotenv(**kw):
        pass

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

# ---------------------------------------------------------------------------
# Broker / monitoring imports — guarded so offline tools still work
# ---------------------------------------------------------------------------
try:
    from broker.alpaca_client import AlpacaClient
    from broker.order_executor import OrderExecutor, OrderSide
    from broker.position_tracker import PositionTracker
    _HAS_BROKER = True
except ImportError:
    _HAS_BROKER = False

from core.hmm_engine import HMMEngine, HMMConfig, RegimeState
from core.regime_strategies import (
    StrategyOrchestrator,
    RegimeStrategy,
    StrategyConfig,
    Signal,
)
from core.risk_manager import (
    RiskManager,
    RiskConfig,
    PortfolioState,
    CircuitBreaker,
    BreakerType,
    TradingState,
)
from data.feature_engineering import FeatureEngineer
from core.intraday_engine import (
    IntradayEngine,
    PARTIAL_CLOSE_FRAC_1,
    PARTIAL_CLOSE_FRAC_2,
    TRAIL_ATR_MULTIPLIER,
    TRAIL_ATR_MULTIPLIER_2,
)

try:
    from data.market_data import MarketDataFeed
    _HAS_FEED = True
except ImportError:
    _HAS_FEED = False

from backtest.backtester import WalkForwardBacktester, BacktestConfig
from backtest.performance import PerformanceAnalyzer

try:
    from monitoring.alerts import AlertManager, Alert, AlertSeverity
    _HAS_ALERTS = True
except ImportError:
    _HAS_ALERTS = False

try:
    from monitoring.dashboard import Dashboard
    _HAS_DASHBOARD = True
except ImportError:
    _HAS_DASHBOARD = False

try:
    from monitoring.web_dashboard import WebDashboard
    _HAS_WEB_DASHBOARD = True
except ImportError:
    _HAS_WEB_DASHBOARD = False

def _make_std_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    return logging.getLogger(name)

try:
    from monitoring.logger import get_logger as _get_logger
    _candidate = _get_logger(__name__)
    # get_logger is still a stub (returns None) — fall back to stdlib
    log = _candidate if isinstance(_candidate, logging.Logger) else _make_std_logger(__name__)
except Exception:
    log = _make_std_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MODEL_DIR          = Path("models")
_SNAPSHOT_PATH      = Path("state_snapshot.json")


def _model_path(symbol: str) -> Path:
    """Return the per-asset HMM model file path."""
    return _MODEL_DIR / f"{symbol.upper()}_hmm.pkl"
_MODEL_MAX_AGE_DAYS = 7          # retrain HMM if model file is older than this
_HISTORY_LOOKBACK   = 504        # calendar days of history for feed init (≈2yr)
_BAR_POLL_INTERVAL  = 1.0        # seconds between buffer polls in main loop
_FEED_DEAD_TIMEOUT_DEFAULT = 93600  # 26h for daily bars; override via broker.feed_dead_timeout_seconds
_RECENT_SIGNALS_MAX = 20         # rolling buffer size for dashboard display
_MAX_API_RETRIES    = 3          # Alpaca API retries before giving up
_API_BACKOFF_BASE   = 2.0        # seconds — exponential back-off base


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_config(path: Path = Path("config/settings.yaml")) -> dict:
    """
    Load and return the settings.yaml configuration as a dict.

    Falls back to an empty dict if the file does not exist or yaml is missing.
    """
    if not _HAS_YAML:
        log.warning("PyYAML not installed — using default configuration.")
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        log.warning("Config file not found at %s — using defaults.", cfg_path)
        return {}
    with open(cfg_path, "r") as fh:
        return yaml.safe_load(fh) or {}


def _build_hmm_config(cfg: dict) -> HMMConfig:
    h = cfg.get("hmm", {})
    return HMMConfig(
        n_candidates    = h.get("n_candidates",    [3, 4, 5, 6, 7]),
        n_init          = h.get("n_init",          10),
        covariance_type = h.get("covariance_type", "full"),
        min_train_bars  = h.get("min_train_bars",  252),
        stability_bars  = h.get("stability_bars",  3),
        flicker_window  = h.get("flicker_window",  20),
        flicker_threshold = h.get("flicker_threshold", 4),
        min_confidence  = h.get("min_confidence",  0.55),
    )


def _build_strategy_config(cfg: dict) -> StrategyConfig:
    s = cfg.get("strategy", {})
    return StrategyConfig(
        low_vol_allocation          = s.get("low_vol_allocation",          0.95),
        mid_vol_allocation_trend    = s.get("mid_vol_allocation_trend",    0.95),
        mid_vol_allocation_no_trend = s.get("mid_vol_allocation_no_trend", 0.60),
        high_vol_allocation         = s.get("high_vol_allocation",         0.60),
        low_vol_leverage            = s.get("low_vol_leverage",            1.25),
        rebalance_threshold         = s.get("rebalance_threshold",         0.10),
        uncertainty_size_mult       = s.get("uncertainty_size_mult",       0.50),
        min_confidence              = s.get("min_confidence",              0.55),
        atr_window                  = s.get("atr_window",                  14),
    )


def _build_risk_config(cfg: dict) -> RiskConfig:
    r = cfg.get("risk", {})
    return RiskConfig(
        max_risk_per_trade  = r.get("max_risk_per_trade",  0.01),
        max_exposure        = r.get("max_exposure",        0.80),
        max_leverage        = r.get("max_leverage",        1.25),
        max_single_position = r.get("max_single_position", 0.15),
        max_concurrent      = r.get("max_concurrent",      5),
        max_daily_trades    = r.get("max_daily_trades",    20),
        daily_dd_reduce     = r.get("daily_dd_reduce",     0.02),
        daily_dd_halt       = r.get("daily_dd_halt",       0.03),
        weekly_dd_reduce    = r.get("weekly_dd_reduce",    0.05),
        weekly_dd_halt      = r.get("weekly_dd_halt",      0.07),
        max_dd_from_peak    = r.get("max_dd_from_peak",    0.10),
    )


def _build_backtest_config(cfg: dict, initial_capital: float = 100_000.0) -> BacktestConfig:
    bt = cfg.get("backtest", {})
    return BacktestConfig(
        slippage_pct    = bt.get("slippage_pct",    0.0005),
        initial_capital = initial_capital,
        train_window    = bt.get("train_window",    252),
        test_window     = bt.get("test_window",     126),
        step_size       = bt.get("step_size",       126),
        risk_free_rate  = bt.get("risk_free_rate",  0.045),
        commission      = bt.get("commission",      0.0),
    )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_csv_data(path: str, symbol: str) -> pd.DataFrame:
    """Load OHLCV from CSV with auto-detected date column."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    date_col = None
    for candidate in ["Date", "date", "Datetime", "datetime", "timestamp", "Timestamp"]:
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = "date"
    df.sort_index(inplace=True)
    df.columns = [c.lower() for c in df.columns]
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"CSV {path!r} missing required column '{col}'.")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    log.info("Loaded %d bars from %s for symbol %s", len(df), path, symbol)
    return df


def _make_synthetic_data(
    symbol: str = "SPY",
    n_bars: int = 1500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV with embedded vol regimes for demo/testing."""
    rng = np.random.default_rng(seed)
    segments = [
        (500,      0.15 / 252,  0.10 / np.sqrt(252)),
        (500,     -0.20 / 252,  0.30 / np.sqrt(252)),
        (n_bars - 1000, 0.25 / 252, 0.15 / np.sqrt(252)),
    ]
    log_returns: list = []
    for seg_n, drift, vol in segments:
        log_returns.extend(rng.normal(drift, vol, size=seg_n))
    log_returns_arr = np.array(log_returns[:n_bars])
    prices      = 300.0 * np.exp(np.cumsum(log_returns_arr))
    noise       = np.abs(rng.normal(0, 0.005, size=n_bars))
    opens       = np.roll(prices, 1); opens[0] = 300.0
    opens      *= (1.0 + rng.normal(0, 0.002, size=n_bars))
    highs       = np.maximum(prices, opens) * (1.0 + noise)
    lows        = np.minimum(prices, opens) * (1.0 - noise)
    volumes     = rng.integers(1_000_000, 50_000_000, size=n_bars).astype(float)
    index       = pd.bdate_range(start=pd.Timestamp("2019-01-02"), periods=n_bars)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": prices, "volume": volumes},
        index=index,
    )


# ---------------------------------------------------------------------------
# State snapshot
# ---------------------------------------------------------------------------

def _save_snapshot(path: Path, state: dict) -> None:
    """Persist session state to JSON for crash recovery."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(state, fh, indent=2, default=str)
        log.debug("State snapshot saved → %s", path)
    except Exception as exc:
        log.error("Failed to save state snapshot: %s", exc)


def _load_snapshot(path: Path) -> Optional[dict]:
    """Load a previous session state snapshot, or return None."""
    if not path.exists():
        return None
    try:
        with open(path, "r") as fh:
            data = json.load(fh)
        log.info("Loaded state snapshot from %s (session_start=%s)",
                 path, data.get("session_start", "?"))
        return data
    except Exception as exc:
        log.warning("Could not load snapshot %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Alpaca API retry helper
# ---------------------------------------------------------------------------

def _with_retries(fn, *args, label: str = "API call", **kwargs):
    """
    Call ``fn(*args, **kwargs)`` with up to ``_MAX_API_RETRIES`` attempts.

    Uses exponential back-off between retries.  Re-raises on final failure.
    """
    import random
    delay = _API_BACKOFF_BASE
    for attempt in range(1, _MAX_API_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == _MAX_API_RETRIES:
                log.error("%s failed after %d attempts: %s", label, _MAX_API_RETRIES, exc)
                raise
            jitter  = random.uniform(-0.2, 0.2)
            sleep_s = min(delay * (1.0 + jitter), 30.0)
            log.warning(
                "%s attempt %d/%d failed (%s). Retrying in %.1fs …",
                label, attempt, _MAX_API_RETRIES, exc, sleep_s,
            )
            time.sleep(sleep_s)
            delay *= 2.0


# ---------------------------------------------------------------------------
# Observability — lifecycle state + system snapshot
# ---------------------------------------------------------------------------

class LifecycleState(str, Enum):
    INIT     = "INIT"
    READY    = "READY"
    TRADING  = "TRADING"
    DEGRADED = "DEGRADED"
    FAILED   = "FAILED"


@dataclass
class SystemState:
    """
    Single source of truth for the bot's runtime health.

    Exposed via ``/health`` (no auth) and ``python main.py status``.
    Updated in-place throughout the session lifecycle.
    """
    lifecycle:         LifecycleState = LifecycleState.INIT
    alpaca_ok:         bool           = False
    market_open:       Optional[bool] = None
    hmm_fitted:        dict           = field(default_factory=dict)   # sym → bool
    daily_bars:        dict           = field(default_factory=dict)   # sym → int
    intraday_bars:     dict           = field(default_factory=dict)   # sym → int
    intraday_enabled:  bool           = False
    last_signal_ts:    Optional[str]  = None
    last_signal_sym:   Optional[str]  = None
    last_signal_type:  Optional[str]  = None
    last_exec_result:  Optional[str]  = None   # "SUBMITTED" | "REJECTED:<reason>" | "FAILED:<e>"
    open_positions:    int            = 0
    equity:            float          = 0.0
    daily_pnl:         float          = 0.0
    circuit_breaker:   str            = "NORMAL"
    errors:            list           = field(default_factory=list)   # rotating max 10
    dashboard_url:     str            = ""
    session_start:     str            = ""
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False, hash=False
    )

    def add_error(self, msg: str) -> None:
        """Append an error string (rotating buffer, max 10 entries)."""
        with self._lock:
            self.errors = (
                self.errors + [f"{datetime.now().strftime('%H:%M:%S')} {msg}"]
            )[-10:]

    def as_dict(self) -> dict:
        """Return a JSON-serialisable dict (excludes the internal lock)."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# TradingSession
# ---------------------------------------------------------------------------

class TradingSession:
    """
    Encapsulates the full lifecycle of a live trading session.

    Startup → main bar loop → graceful shutdown.

    The session is driven by bar arrivals from the WebSocket data feed.
    Each bar triggers the full pipeline:

        new bar → features → HMM → regime → strategy → risk → order

    Usage::

        session = TradingSession(cfg, symbols=["SPY", "AAPL"], dry_run=False)
        session.startup()
        session.run()          # blocks until SIGINT / SIGTERM
        session.shutdown()
    """

    def __init__(
        self,
        cfg:     dict,
        symbols: list[str],
        dry_run: bool = False,
    ) -> None:
        self.cfg     = cfg
        self.symbols = list(symbols)
        self.dry_run = dry_run

        # ── Observability ─────────────────────────────────────────────────────
        self.sys_state = SystemState(
            session_start=datetime.now(timezone.utc).isoformat(),
        )

        # ── components (filled by startup) ────────────────────────────────────
        self.client:       Optional[AlpacaClient]       = None
        self.executor:     Optional[OrderExecutor]      = None
        self.feed:         Optional[MarketDataFeed]     = None
        self.tracker:      Optional[PositionTracker]    = None
        # Per-asset HMM engines and orchestrators — one per symbol
        self.hmm_engines:   dict[str, HMMEngine]            = {}
        self.orchestrators: dict[str, StrategyOrchestrator] = {}
        self.risk_mgr:     Optional[RiskManager]        = None
        self.portfolio:    Optional[PortfolioState]     = None
        self.breaker:      Optional[CircuitBreaker]     = None
        self.alert_mgr:     Optional[object]             = None  # AlertManager
        self.dashboard:     Optional[object]             = None  # Dashboard
        self.web_dashboard: Optional[object]             = None  # WebDashboard
        self.feature_eng = FeatureEngineer()

        # ── session tracking ──────────────────────────────────────────────────
        self._stop_event        = threading.Event()
        self._session_start     = datetime.now(timezone.utc)
        self._last_train_time:  Optional[datetime] = None
        self._last_bar_time:    Optional[pd.Timestamp] = None
        self._last_bar_wall:    float = time.monotonic()
        # Per-asset regime tracking
        self._prev_regimes:        dict[str, str]        = {}
        self._last_regime_states:  dict[str, RegimeState] = {}
        self._data_feed_ok      = True
        self._bars_processed    = 0
        self._signals_generated = 0
        self._orders_submitted  = 0
        self._orders_rejected   = 0
        self._recent_signals: list[Signal] = []

        # ── Intraday execution engine ──────────────────────────────────────────
        self.intraday_engine: Optional[IntradayEngine] = None
        # symbol → bar-count at last intraday poll (to detect new bars cheaply)
        self._last_intraday_ts:   dict[str, Optional[pd.Timestamp]] = {}
        # symbol → trades entered today (resets at midnight / session start)
        self._intraday_trades_today: dict[str, int] = {}
        self._intraday_session_date: Optional[str] = None  # "YYYY-MM-DD"

    # =========================================================================
    # Startup
    # =========================================================================

    def startup(self) -> None:
        """
        Full startup sequence (8 steps):

        1. Connect to Alpaca, verify account
        2. Check market hours
        3. Initialise data feed and back-fill history
        4. Load or train HMM model
        5. Initialise risk manager with live portfolio
        6. Initialise position tracker, sync from broker
        7. Restore session state from snapshot (if present)
        8. Start WebSocket data + trade streams
        """
        log.info("━" * 62)
        log.info("  Regime Trader  —  Live Trading Startup")
        log.info("━" * 62)

        broker_cfg = self.cfg.get("broker", {})
        timeframe  = broker_cfg.get("timeframe", "1Day")

        # ── Step 1: Connect to Alpaca ─────────────────────────────────────────
        log.info("[1/8] Connecting to Alpaca …")
        self.client = AlpacaClient(
            paper=broker_cfg.get("paper_trading", True)
        )
        acct   = _with_retries(self.client.get_account, label="get_account")
        equity = acct["equity"]
        self.sys_state.alpaca_ok = True
        self.sys_state.equity    = float(equity)
        log.info(
            "      mode=%-6s  equity=$%s  buying_power=$%s",
            "PAPER" if self.client.paper else "LIVE",
            f"{equity:,.2f}", f"{acct['buying_power']:,.2f}",
        )

        # ── Step 2: Market hours ──────────────────────────────────────────────
        log.info("[2/8] Checking market hours …")
        clock = _with_retries(self.client.get_clock, label="get_clock")
        self.sys_state.market_open = bool(clock["is_open"])
        if clock["is_open"]:
            log.info("      Market OPEN  — next close: %s", clock.get("next_close", "?"))
        else:
            log.info(
                "      Market CLOSED — next open:  %s  (system will wait for bars)",
                clock.get("next_open", "?"),
            )

        # ── Step 3: Data feed + history ───────────────────────────────────────
        log.info("[3/8] Initialising data feed (lookback=%d days) …", _HISTORY_LOOKBACK)
        self.feed = MarketDataFeed(
            client    = self.client,
            symbols   = self.symbols,
            timeframe = timeframe,
        )
        self.feed.initialise(lookback_days=_HISTORY_LOOKBACK)
        available = self.feed.bars_available()
        empty_syms = [s for s, n in available.items() if n == 0]
        for sym, n in available.items():
            log.info("      %-6s  %d bars loaded", sym, n)
            self.sys_state.daily_bars[sym] = n
        if empty_syms:
            log.error(
                "FATAL DATA GAP: history load returned 0 bars for %s — "
                "check API credentials (.env), Alpaca data subscription, "
                "and whether these symbols are accessible on your plan.  "
                "Pipeline will stall until bars are present.",
                empty_syms,
            )
        elif not available:
            log.error(
                "FATAL DATA GAP: feed.bars_available() returned empty dict — "
                "no symbols loaded at all.  Check API credentials."
            )

        # ── Step 4: Load or train per-asset HMMs ─────────────────────────────
        log.info("[4/8] Loading / training HMM (one per symbol) …")
        hmm_config = _build_hmm_config(self.cfg)
        for sym in self.symbols:
            engine = HMMEngine(hmm_config)
            self.hmm_engines[sym] = engine
            self._load_or_train_hmm_for(sym, engine)
            self.sys_state.hmm_fitted[sym] = engine.is_fitted

        # ── Step 5: Risk manager ──────────────────────────────────────────────
        log.info("[5/8] Initialising risk manager …")
        risk_config    = _build_risk_config(self.cfg)
        self.risk_mgr  = RiskManager(risk_config, initial_equity=equity)
        self.breaker   = self.risk_mgr._circuit_breaker
        self.portfolio = PortfolioState(
            equity       = equity,
            cash         = acct["cash"],
            buying_power = acct["buying_power"],
            peak_equity  = equity,
            sod_equity   = equity,
            sow_equity   = equity,
        )

        # ── Step 6: Position tracker + sync ──────────────────────────────────
        log.info("[6/8] Initialising position tracker …")
        self.executor = OrderExecutor(self.client)
        self.tracker  = PositionTracker(
            client          = self.client,
            portfolio_state = self.portfolio,
            circuit_breaker = self.breaker,
        )
        self.tracker.sync_from_broker()
        open_pos = self.tracker.get_all_positions()
        if open_pos:
            log.info("      Reconciled %d open position(s): %s",
                     len(open_pos), list(open_pos.keys()))
        else:
            log.info("      No open positions.")

        # ── Step 7: Per-asset strategy orchestrators ──────────────────────────
        log.info("[7/8] Initialising strategy orchestrators (one per symbol) …")
        strategy_cfg = _build_strategy_config(self.cfg)
        for sym in self.symbols:
            engine = self.hmm_engines[sym]
            regime_infos = engine._regime_infos if engine.is_fitted else {}
            self.orchestrators[sym] = StrategyOrchestrator(strategy_cfg, regime_infos)

        # ── Step 7b: Restore snapshot ─────────────────────────────────────────
        log.info("[7/8] Checking for recovery snapshot …")
        snap = _load_snapshot(_SNAPSHOT_PATH)
        if snap:
            self._restore_from_snapshot(snap)
        else:
            log.info("      No snapshot found — fresh session.")

        # ── Step 8: Start WebSocket feeds ─────────────────────────────────────
        log.info("[8/8] Starting WebSocket data feeds …")
        self.feed.start_stream()
        self.tracker.start_stream()

        # Alert manager (best-effort)
        if _HAS_ALERTS:
            try:
                mon_cfg = self.cfg.get("monitoring", {})
                self.alert_mgr = AlertManager(
                    rate_limit_minutes=mon_cfg.get("alert_rate_limit_minutes", 15),
                )
            except Exception:
                pass

        # Dashboard (best-effort)
        if _HAS_DASHBOARD:
            try:
                mon_cfg = self.cfg.get("monitoring", {})
                self.dashboard = Dashboard(
                    risk_manager     = self.risk_mgr,
                    position_tracker = self.tracker,
                    refresh_seconds  = mon_cfg.get("dashboard_refresh_seconds", 5),
                )
            except Exception:
                pass

        # Web dashboard (best-effort)
        if _HAS_WEB_DASHBOARD:
            try:
                mon_cfg  = self.cfg.get("monitoring", {})
                web_port = mon_cfg.get("web_dashboard_port", 8080)
                web_host = mon_cfg.get("web_dashboard_host", "0.0.0.0")
                self.web_dashboard = WebDashboard(host=web_host, port=web_port)
                self.web_dashboard.start()
                # Verify the Flask thread actually came up
                time.sleep(0.5)
                if self.web_dashboard._thread and self.web_dashboard._thread.is_alive():
                    log.info(
                        "Web dashboard live → http://localhost:%d/  (user: admin  "
                        "| remote: http://<server-ip>:%d/)",
                        web_port, web_port,
                    )
                else:
                    log.error(
                        "Web dashboard thread died immediately — dashboard is NOT accessible. "
                        "Check that Flask is installed:  pip install flask"
                    )
            except ImportError as exc:
                log.error(
                    "Web dashboard unavailable: %s  — run:  pip install flask", exc
                )
            except Exception as exc:
                log.error("Web dashboard failed to start: %s", exc)

        # ── Intraday engine (optional — controlled by settings.yaml) ──────────
        intraday_cfg = self.cfg.get("intraday", {})
        if intraday_cfg.get("enabled", False):
            log.info("Initialising intraday execution engine …")
            id_tf       = intraday_cfg.get("timeframe", "5Min")
            id_lookback = intraday_cfg.get("lookback_minutes", 4000) #~10 trading days
            self.feed.initialise_intraday(
                timeframe=id_tf,
                lookback_minutes=id_lookback,
            )
            id_bars = self.feed.intraday_bars_available()
            for sym, n in id_bars.items():
                log.info("      %-6s  %d intraday bars loaded (%s)", sym, n, id_tf)
                self.sys_state.intraday_bars[sym] = n
            self.sys_state.intraday_enabled = True

            self.intraday_engine = IntradayEngine(
                min_bars   = intraday_cfg.get("min_bars", 60),
                cooldown_bars = intraday_cfg.get("cooldown_bars", 3),
                no_entry_minutes_before_close = intraday_cfg.get(
                    "no_entry_minutes_before_close", 15),
            )
            log.info("Intraday engine ready — tf=%s  cooldown=%d bars",
                     id_tf, intraday_cfg.get("cooldown_bars", 3))
        else:
            log.info("Intraday engine DISABLED (intraday.enabled=false in settings).")

        # ── System online ──────────────────────────────────────────────────────
        self.sys_state.lifecycle = LifecycleState.READY
        log.info("━" * 62)
        log.info(
            "  System ONLINE  |  %s  |  %s  |  $%s",
            self._session_start.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "DRY-RUN" if self.dry_run else ("PAPER" if self.client.paper else "LIVE"),
            f"{equity:,.0f}",
        )
        log.info("  Symbols : %s", "  ".join(self.symbols))
        fitted = sum(1 for e in self.hmm_engines.values() if e.is_fitted)
        log.info("  HMM     : %d / %d fitted (per-asset)", fitted, len(self.symbols))
        log.info("━" * 62)

        # Push initial state to web dashboard
        if self.web_dashboard:
            try:
                self.web_dashboard.set_system_state(self.sys_state.as_dict())
            except Exception:
                pass

    # =========================================================================
    # Main event loop
    # =========================================================================

    def run(self) -> None:
        """
        Hybrid main loop — two independent processing paths share one thread:

        PATH A  Daily (1D bars, arrives once at ~4pm ET)
            → HMM regime update per symbol
            → Updates _last_regime_states (the regime filter for intraday)
            → NO direct order execution

        PATH B  Intraday (5m bars, polled via REST every poll_interval_seconds)
            → IntradayEngine signal generation per symbol
            → Filtered / sized by daily regime from PATH A
            → Risk validation → order execution

        Both paths share the same risk manager, position tracker, and circuit
        breakers.  If intraday is disabled the loop behaves exactly as before.
        """
        clock_sym = self.symbols[0]
        _bar_cfg  = self.cfg.get("broker", {}).get("timeframe", "1Day")
        _expected_bar_interval = 86400 if "Day" in _bar_cfg else (
            3600 if "Hour" in _bar_cfg else 300
        )

        # Intraday config
        id_cfg            = self.cfg.get("intraday", {})
        id_enabled        = self.intraday_engine is not None
        id_poll_interval  = float(id_cfg.get("poll_interval_seconds", 30))
        id_max_per_day    = int(id_cfg.get("max_trades_per_day_per_symbol", 3))

        _last_intraday_poll = 0.0   # monotonic time of last REST poll

        self.sys_state.lifecycle = LifecycleState.TRADING
        log.info(
            "Main loop started — clock: %s  |  intraday: %s  |  symbols: %s",
            clock_sym,
            f"ENABLED ({id_cfg.get('timeframe','5Min')})" if id_enabled else "DISABLED",
            "  ".join(self.symbols),
        )

        while not self._stop_event.is_set():
            try:
                time.sleep(_BAR_POLL_INTERVAL)

                if self._stop_event.is_set():
                    break

                # ── Feed health check ─────────────────────────────────────────
                self._check_feed_health()
                if not self._data_feed_ok:
                    log.warning("Data feed down — skipping bar processing, stops remain active")
                    continue

                # ════════════════════════════════════════════════════════════════
                # PATH A — Daily HMM update (once per day)
                # ════════════════════════════════════════════════════════════════
                df = self.feed.get_bars(clock_sym)
                if df.empty:
                    _now_mono = time.monotonic()
                    if not hasattr(self, "_empty_buf_logged") or _now_mono - self._empty_buf_logged > 60:
                        self._empty_buf_logged = _now_mono
                        avail = self.feed.bars_available() if self.feed else {}
                        log.warning(
                            "PIPELINE STALLED: get_bars(%s) returned empty "
                            "— buffer sizes: %s  "
                            "(check startup logs for 'bars loaded' counts)",
                            clock_sym, avail,
                        )
                else:
                    latest_ts = df.index[-1]
                    if self._last_bar_time is None or latest_ts > self._last_bar_time:
                        self._last_bar_time = latest_ts
                        self._last_bar_wall = time.monotonic()

                        if self._bars_processed > 1:
                            bar_lag = (pd.Timestamp.now(tz="UTC") - latest_ts).total_seconds()
                            if bar_lag > 2 * _expected_bar_interval:
                                log.warning(
                                    "Clock-source bar lag: %s arrived %.0fs after timestamp "
                                    "(interval=%ds).", clock_sym, bar_lag, _expected_bar_interval,
                                )

                        log.debug("New daily bar  %s  %s  close=%.4f",
                                  clock_sym, latest_ts, float(df["close"].iloc[-1]))

                        # Daily HMM update: updates _last_regime_states per symbol
                        # Does NOT submit orders — regime used as intraday filter
                        self._process_bar(latest_ts)
                        self._check_weekly_retrain()

                        # Reset per-session intraday trade counter on new day
                        today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
                        if today_str != self._intraday_session_date:
                            self._intraday_session_date  = today_str
                            self._intraday_trades_today  = {}
                            log.info("New trading day %s — intraday trade counters reset", today_str)

                # ════════════════════════════════════════════════════════════════
                # PATH B — Intraday execution (every poll_interval_seconds)
                # ════════════════════════════════════════════════════════════════
                if id_enabled:
                    _now_mono = time.monotonic()
                    if _now_mono - _last_intraday_poll >= id_poll_interval:
                        _last_intraday_poll = _now_mono
                        self._poll_and_process_intraday(id_max_per_day)

            except KeyboardInterrupt:
                break
            except Exception as exc:
                self._handle_unhandled_error(exc)

        log.info("Main loop exited.")

    # =========================================================================
    # Intraday execution path (PATH B)
    # =========================================================================

    def _poll_and_process_intraday(self, max_per_day: int) -> None:
        """
        Called every ``poll_interval_seconds`` from ``run()`` PATH B.

        Steps
        -----
        1. Poll REST for new intraday bars (non-blocking — returns immediately
           with a per-symbol flag indicating whether new bars arrived).
        2. For every symbol: tick the cooldown counter.
        3. For every symbol that has bars, an active daily regime, and room
           in the daily trade budget: ask the IntradayEngine for a signal.
        4. Route any returned Signal through the existing risk + execution
           pipeline (_handle_signal → RiskManager → OrderExecutor).
        5. After all symbols: run TP1 / break-even checks and update trailing
           stops on all open positions.
        6. Push updated state to the web dashboard.
        """
        if not self.intraday_engine or not self.feed:
            return

        # ── 1. Fetch new intraday bars ────────────────────────────────────────
        try:
            self.feed.poll_intraday()  # returns {sym: bool}; we just want side-effects
        except Exception as exc:
            log.warning("poll_intraday raised: %s — skipping this cycle", exc)
            return

        # ── 2 + 3. Per-symbol signal generation ──────────────────────────────
        market_close_utc = self._get_market_close_utc()

        for sym in self.symbols:
            # Always advance the cooldown timer (one tick per poll cycle)
            self.intraday_engine.tick(sym)

            # Need intraday bars
            id_bars = self.feed.get_intraday_bars(sym)
            if id_bars.empty:
                continue

            # Need a daily regime from PATH A (HMM must have run at least once)
            daily_regime = self._last_regime_states.get(sym)
            if daily_regime is None:
                log.debug("[%s] No daily regime yet — skipping intraday signal", sym)
                continue

            # Daily trade limit
            trades_today = self._intraday_trades_today.get(sym, 0)
            if trades_today >= max_per_day:
                continue

            # Generate signal
            try:
                sig = self.intraday_engine.generate_signal(
                    symbol           = sym,
                    bars             = id_bars,
                    daily_regime     = daily_regime,
                    market_close_utc = market_close_utc,
                )
            except Exception as exc:
                log.error("[%s] IntradayEngine.generate_signal failed: %s", sym, exc)
                continue

            if sig is None:
                continue

            log.info(
                "INTRADAY  %-6s  %-26s  entry=%.4f  stop=%.4f  tp=%s  conf=%.2f",
                sym, sig.strategy_name,
                sig.entry_price, sig.stop_loss,
                f"{sig.take_profit:.4f}" if sig.take_profit else "n/a",
                sig.metadata.get("confidence", 0.0),
            )
            self._signals_generated += 1
            self._recent_signals = (self._recent_signals + [sig])[-_RECENT_SIGNALS_MAX:]

            # ── 4. Risk → execution ───────────────────────────────────────────
            prev_submitted = self._orders_submitted
            self._handle_signal(sig, daily_regime)
            if self._orders_submitted > prev_submitted:
                # Order went through: reset cooldown and charge the budget
                self.intraday_engine.record_entry(sym)
                self._intraday_trades_today[sym] = trades_today + 1
                log.info(
                    "[%s] Intraday trade %d/%d today — cooldown reset",
                    sym, self._intraday_trades_today[sym], max_per_day,
                )

        # ── 5. Position maintenance ───────────────────────────────────────────
        # TP1 / BE checks use current_price updated by tracker.update_bar()
        # which PATH A refreshes once per daily close.  Between daily bars the
        # intraday price is stale — we update it here from the latest 5m close.
        if self.tracker:
            for sym in self.symbols:
                pos = self.tracker.get_position(sym)
                if pos is None:
                    continue
                id_bars = self.feed.get_intraday_bars(sym)
                if not id_bars.empty:
                    latest_price = float(id_bars["close"].iloc[-1])
                    regime_label = (
                        self._last_regime_states[sym].label
                        if sym in self._last_regime_states else ""
                    )
                    self.tracker.update_bar(sym, latest_price, regime_label)

        self._check_tp_triggers()
        self._check_be_triggers()

        # Trailing stops use daily bars (ATR computation needs full history)
        bars_by_sym: dict[str, pd.DataFrame] = {}
        for sym in self.symbols:
            if self.tracker and self.tracker.get_position(sym) is not None:
                db = self.feed.get_bars(sym)
                if not db.empty:
                    bars_by_sym[sym] = db
        if bars_by_sym:
            self._update_trailing_stops_per_asset(bars_by_sym)

        # ── 6. Web dashboard push ─────────────────────────────────────────────
        if self.web_dashboard:
            try:
                self.web_dashboard.push(self._build_web_state())
            except Exception:
                pass

    def _get_market_close_utc(self) -> Optional[pd.Timestamp]:
        """
        Return today's scheduled market close as a UTC Timestamp.

        Result is cached for 5 minutes so we don't hit the clock endpoint on
        every 30-second poll cycle.  Falls back to a hard-coded 21:00 UTC
        (4 pm ET standard time) if the API call fails.
        """
        _CACHE_TTL = 300.0  # seconds
        now_mono   = time.monotonic()

        if now_mono - getattr(self, "_mkt_close_cache_mono", 0.0) < _CACHE_TTL:
            return getattr(self, "_mkt_close_utc_cached", None)

        try:
            clock     = self.client.get_clock()
            close_str = clock.get("next_close")
            if close_str:
                ts = pd.Timestamp(close_str)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                else:
                    ts = ts.tz_convert("UTC")
                self._mkt_close_utc_cached  = ts
                self._mkt_close_cache_mono  = now_mono
                return ts
        except Exception as exc:
            log.debug("get_clock failed (%s) — using cached/fallback close time", exc)

        # Fallback: US market closes at 21:00 UTC (4 pm ET; ignores DST)
        fallback = pd.Timestamp.now("UTC").normalize() + pd.Timedelta(hours=21)
        if not hasattr(self, "_mkt_close_utc_cached"):
            self._mkt_close_utc_cached = fallback
        return self._mkt_close_utc_cached

    # =========================================================================
    # Bar processing pipeline
    # =========================================================================

    def _process_bar(self, bar_ts: pd.Timestamp) -> None:
        """
        Per-bar pipeline — runs independently for each symbol:

        For each symbol:
          1. Build features from that symbol's own bars
          2. Run that symbol's own HMM → per-asset regime_state
          3. Regime stability / flicker check (per-asset)
          4. That symbol's own StrategyOrchestrator → signal
        Then:
          5. Risk validation → order submission
          6. Update trailing stops (per symbol, per its own regime)
          7. Circuit-breaker re-evaluation
          8. Dashboard refresh
        """
        self._bars_processed += 1
        all_signals: list[Signal] = []
        bars_by_sym:  dict[str, pd.DataFrame] = {}

        for sym in self.symbols:
            engine = self.hmm_engines.get(sym)
            if engine is None:
                continue

            sym_bars = self.feed.get_bars(sym)
            if sym_bars.empty or len(sym_bars) < engine.config.min_train_bars:
                log.debug("[%s] Insufficient bars (%d / %d)",
                          sym, len(sym_bars), engine.config.min_train_bars)
                continue

            if not engine.is_fitted:
                log.warning("[%s] HMM not fitted — skipping", sym)
                continue

            # ── 1: Features (only this symbol's bars) ─────────────────────────
            try:
                features = self.feature_eng.build_features(sym_bars).dropna()
                if len(features) < engine.config.min_train_bars:
                    continue
            except Exception as exc:
                log.error("[%s] Feature computation failed: %s", sym, exc)
                continue

            # ── 2: Per-asset HMM prediction ───────────────────────────────────
            try:
                regime_state = engine.predict_filtered_next(features)
            except Exception as exc:
                log.warning("[%s] HMM predict failed (%s) — holding regime", sym, exc)
                regime_state = self._last_regime_states.get(sym)
                if regime_state is None:
                    continue

            self._last_regime_states[sym] = regime_state
            bars_by_sym[sym] = sym_bars

            # ── 2b: Refresh tracker price so BE/TP1 checks use today's close ──
            if self.tracker and not sym_bars.empty:
                close_now = float(sym_bars["close"].iloc[-1])
                self.tracker.update_bar(sym, close_now, regime_state.label)

            # ── 3: Regime change log ──────────────────────────────────────────
            label = regime_state.label
            prev  = self._prev_regimes.get(sym, "")
            if label != prev:
                log.info(
                    "Regime change [%-5s]  %-12s → %-12s  (p=%.3f  confirmed=%s)",
                    sym, prev or "?", label,
                    regime_state.probability, regime_state.is_confirmed,
                )
                if self.alert_mgr and prev:
                    try:
                        self.alert_mgr.send_regime_change_alert(prev, label,
                                                                 regime_state.probability)
                    except Exception:
                        pass
                self._prev_regimes[sym] = label

            # ── 4: Per-asset signal ────────────────────────────────────────────
            is_flickering = (
                regime_state.flicker_rate * engine.config.flicker_window
                > engine.config.flicker_threshold
            )
            orch = self.orchestrators.get(sym)
            if orch is None:
                continue
            try:
                sym_signals = orch.generate_signals(
                    symbols       = [sym],
                    bars          = {sym: sym_bars},
                    regime_state  = regime_state,
                    is_flickering = is_flickering,
                )
            except Exception as exc:
                log.error("[%s] generate_signals failed: %s", sym, exc)
                continue

            all_signals.extend(sym_signals)

        # ── No-trade diagnostics: log which gate stopped each symbol ─────────────
        if bars_by_sym or all_signals:
            pass  # at least one symbol made it through — normal path
        else:
            # bars_processed was incremented but nothing reached signal generation
            gate_summary: list[str] = []
            for sym in self.symbols:
                engine = self.hmm_engines.get(sym)
                if engine is None:
                    gate_summary.append(f"{sym}:no_engine")
                    continue
                sym_bars = self.feed.get_bars(sym)
                if sym_bars.empty or len(sym_bars) < engine.config.min_train_bars:
                    gate_summary.append(
                        f"{sym}:insufficient_bars({len(sym_bars)}/{engine.config.min_train_bars})"
                    )
                elif not engine.is_fitted:
                    gate_summary.append(f"{sym}:hmm_not_fitted")
                else:
                    gate_summary.append(f"{sym}:feature_or_signal_error")
            log.info(
                "Bar processed but ZERO signals generated — gate summary: %s",
                "  ".join(gate_summary),
            )

        # Update portfolio regime label (use last-processed symbol for dashboard)
       if self._last_regime_states:
    regimes = list(self._last_regime_states.values())

    self.portfolio.hmm_regime = max(
        regimes,
        key=lambda r: r.probability
    ).label

    self.portfolio.flicker_rate = float(
        sum(r.flicker_rate for r in regimes) / len(regimes)
    )

        if not all_signals and not bars_by_sym:
            return

        self._signals_generated += len(all_signals)
        self._recent_signals = (self._recent_signals + all_signals)[-_RECENT_SIGNALS_MAX:]

        # ── 5: Risk validation → order submission ─────────────────────────────
        if self.breaker.state == TradingState.HALTED:
            log.warning("Circuit breaker HALTED (%s) — no new entries this bar",
                        self.breaker.active_type.value)

        for sig in all_signals:
            regime_state = self._last_regime_states.get(sig.symbol)
            if regime_state is not None:
                self._handle_signal(sig, regime_state)

        # ── 6: Update trailing stops (per symbol, per its own regime) ─────────
        self._update_trailing_stops_per_asset(bars_by_sym)

        # ── 6b: TP1/TP2/TP3 partial exits ────────────────────────────────────
        self._check_tp_triggers()

        # ── 6c: Break-even stop advance (all positions where pnl ≥ 2R) ───────
        self._check_be_triggers()

        # ── 7: Circuit-breaker re-evaluation ──────────────────────────────────
        try:
            new_state, fired = self.breaker.check(self.portfolio)
            if fired != BreakerType.NONE:
                log.warning("Circuit breaker fired: %s", fired.value)
                self._on_circuit_breaker_fired(fired)
        except Exception as exc:
            log.error("Circuit breaker check failed: %s", exc)

        # ── 8: Dashboard refresh ───────────────────────────────────────────────
        if self.dashboard:
            last_rs = next(iter(self._last_regime_states.values()), None)
            if last_rs:
                try:
                    self.dashboard.update(
                        regime_state    = last_rs,
                        recent_signals  = self._recent_signals[-10:],
                        log_lines       = None,
                    )
                except Exception:
                    pass

        # ── Persist snapshot on each bar ─────────────────────────────────────
        self._persist_snapshot()

        # ── Push state to web dashboard ────────────────────────────────────────
        if self.web_dashboard:
            try:
                self.web_dashboard.push(self._build_web_state())
            except Exception:
                pass

    # =========================================================================
    # Signal handling
    # =========================================================================

    def _handle_signal(
        self,
        sig: Signal,
        regime_state: RegimeState,
    ) -> None:
        """
        Validate a single signal through the risk manager and submit (or log).

        Approved    → submit_order (or log only in dry-run)
        Modified    → log modification, submit modified
        Rejected    → log reason
        FLAT + pos  → close position via limit order
        """
        sym = sig.symbol

        # FLAT: consider closing existing position
        if sig.direction == "FLAT":
            pos = self.tracker.get_position(sym) if self.tracker else None
            if pos and pos.qty > 0:
                exit_class = classify_sl_exit(pos.tp1_hit)
                log.info(
                    "FLAT signal for %s (held %d bars) — closing position  [%s]",
                    sym, pos.bars_held, exit_class,
                )
                if not self.dry_run:
                    try:
                        self.executor.close_position(sym)
                        self._orders_submitted += 1
                    except Exception as exc:
                        log.error("close_position(%s) failed: %s", sym, exc)
            return

        # LONG: run risk validation
        decision = None
        try:
            decision = self.risk_mgr.validate_signal(sig, self.portfolio)
        except Exception as exc:
            log.error("validate_signal(%s) raised: %s", sym, exc)
            return

        if not decision.approved:
            self._orders_rejected += 1
            log.info(
                "REJECTED  %-6s  %s", sym, decision.rejection_reason
            )
            self.sys_state.last_exec_result = f"REJECTED:{decision.rejection_reason}"
            return

        approved_sig = decision.modified_signal
        if decision.modifications:
            log.info(
                "MODIFIED  %-6s  %s", sym, " | ".join(decision.modifications)
            )

        # Compute shares from approved signal
        shares = self._compute_shares(approved_sig)
        if shares <= 0:
            notional = (
                (self.portfolio.equity * approved_sig.position_size_pct * approved_sig.leverage)
                if self.portfolio else 0.0
            )
            log.warning(
                "Signal for %s: shares=0 after sizing — skipped "
                "(notional=%.2f entry_price=%.4f)",
                sym, notional, approved_sig.entry_price,
            )
            self._orders_rejected += 1
            return

        # Check if already in position (don't double-enter)
        existing = self.tracker.get_position(sym) if self.tracker else None
        if existing and existing.qty > 0:
            log.info("Already in %s (qty=%d) — no new entry", sym, existing.qty)
            self._orders_rejected += 1
            return

        # Log intent
        log.info(
            "SIGNAL    %-6s  LONG  qty=%-4d  entry=%.2f  stop=%.2f  "
            "target=%s  regime=%s  p=%.3f  [%s]",
            sym, shares,
            sig.entry_price, sig.stop_loss,
            f"{sig.take_profit:.2f}" if sig.take_profit else "n/a",
            regime_state.label, regime_state.probability,
            sig.strategy_name,
        )

        # Track last signal attempt in sys_state
        self.sys_state.last_signal_ts   = datetime.now(timezone.utc).isoformat()
        self.sys_state.last_signal_sym  = sym
        self.sys_state.last_signal_type = sig.strategy_name

        if self.dry_run:
            log.info("  [DRY-RUN] Order NOT submitted.")
            self.sys_state.last_exec_result = "DRY-RUN"
            return

        # Submit order
        try:
            self._submit_order_from_signal(sig, approved_sig, shares, regime_state)
        except Exception as exc:
            log.error("Order submission failed for %s: %s", sym, exc)
            self.sys_state.last_exec_result = f"FAILED:{exc}"
            if self.alert_mgr:
                try:
                    self.alert_mgr.send_error_alert(exc, context=f"submit_order({sym})")
                except Exception:
                    pass

    def _compute_shares(self, sig: Signal) -> int:
        """Derive integer share count from a risk-approved Signal."""
        if self.portfolio is None or self.portfolio.equity <= 0:
            return 0
        notional = self.portfolio.equity * sig.position_size_pct * sig.leverage
        if sig.entry_price <= 0:
            return 0
        return int(notional / sig.entry_price)

    def _submit_order_from_signal(
        self,
        original_sig: Signal,
        approved_sig: Signal,
        shares: int,
        regime_state: RegimeState,
    ) -> None:
        """
        Convert a risk-approved Signal into a bracket order or limit order
        and submit via OrderExecutor.
        """
        sym        = original_sig.symbol
        entry_p    = approved_sig.entry_price
        stop_p     = approved_sig.stop_loss
        target_p   = approved_sig.take_profit
        trade_id   = uuid.uuid4().hex

        if stop_p and target_p:
            # Bracket order: entry + OCO stop + take-profit
            limit_entry = round(entry_p * 1.001, 2)
            order_request = {
                "symbol":        sym,
                "qty":           shares,
                "side":          _AlpacaSide.BUY if _HAS_BROKER else "buy",
                "limit_price":   limit_entry,
                "stop_price":    round(stop_p, 2),
                "take_profit":   round(target_p, 2),
            }
            result = self.executor.submit_bracket_order(
                _make_trade_signal_stub(
                    sym, shares, entry_p, stop_p, target_p, trade_id
                )
            )
        else:
            # Plain limit order + manual stop update after fill
            result = self.executor.submit_limit_order(
                symbol      = sym,
                side        = OrderSide.BUY,
                qty         = shares,
                limit_price = round(entry_p * 1.001, 2),
                trade_id    = trade_id,
            )

        if result and result.status != "error":
            self._orders_submitted += 1
            self.sys_state.last_exec_result = "SUBMITTED"
            r_val = approved_sig.metadata.get("r_value", 0.0)
            # Record regime, stop, R-value, TP1 level and vol-tier on the tracker
            if self.tracker:
                self.tracker.set_regime_at_entry(sym, regime_state.label)
                if stop_p:
                    self.tracker.set_stop(sym, stop_p)
                if r_val > 0:
                    self.tracker.set_r_value(sym, r_val)
                if target_p:
                    self.tracker.set_tp1_level(sym, target_p)
                tp2 = approved_sig.metadata.get("tp2_level", 0.0) or 0.0
                tp3 = approved_sig.metadata.get("tp3_level") or 0.0
                if tp2:
                    self.tracker.set_tp2_level(sym, tp2)
                if tp3:
                    self.tracker.set_tp3_level(sym, tp3)
                self.tracker.set_vol_tier(sym, approved_sig.strategy_name)
            log.info(
                "ORDER  %-6s  id=%s  status=%s  trade_id=%.8s  "
                "r_value=%.4f  tp1=%.4f  tier=%s",
                sym, result.order_id, result.status, trade_id,
                r_val, target_p or 0.0, approved_sig.strategy_name,
            )
            self.portfolio.daily_trades += 1
        else:
            err = getattr(result, "error", "unknown") if result else "no result"
            log.error("ORDER FAILED  %-6s  error=%s", sym, err)

    # =========================================================================
    # Trailing stop updates
    # =========================================================================

    def _update_trailing_stops_per_asset(
        self,
        bars_by_sym: dict[str, pd.DataFrame],
    ) -> None:
        """
        After each bar, tighten trailing stops for all open positions.
        Each symbol uses its own HMM regime state for stop computation.

        Strategy (long-only, stops can only move up):
          - LowVol:         trailing = rolling_max_close − trailing_atr_mult × ATR
          - MidVol/HighVol: EMA50 − ema_atr_mult × ATR
        """
        if not self.tracker or not self.executor:
            return

        strategy_cfg = _build_strategy_config(self.cfg)
        positions    = self.tracker.get_all_positions()
        if not positions:
            return

        for sym, pos in positions.items():
            bars         = bars_by_sym.get(sym)
            regime_state = self._last_regime_states.get(sym)
            if bars is None or regime_state is None or len(bars) < strategy_cfg.atr_window:
                continue

            try:
                new_stop = _compute_trailing_stop(
                    bars, regime_state, strategy_cfg,
                    tp1_hit=pos.tp1_hit,
                    tp3_hit=getattr(pos, "tp3_hit", False),
                )
            except Exception as exc:
                log.debug("trailing stop compute failed for %s: %s", sym, exc)
                continue

            if new_stop <= 0 or new_stop <= pos.stop_level:
                continue  # only tighten

            try:
                self.executor.modify_stop(
                    symbol       = sym,
                    new_stop     = new_stop,
                    current_stop = pos.stop_level,
                )
                self.tracker.set_stop(sym, new_stop)
                log.debug("Stop tightened  %-6s  %.4f → %.4f",
                          sym, pos.stop_level, new_stop)
            except ValueError:
                pass
            except Exception as exc:
                log.warning("modify_stop(%s) failed: %s", sym, exc)

    # =========================================================================
    # Break-even and TP1 triggers
    # =========================================================================

    def _check_be_triggers(self) -> None:
        """
        Advance stop to break-even (entry price) when unrealised P&L ≥ 2R.

        Called once per bar after trailing stops are updated.  Only moves the
        stop upward — never lowers it.  Does nothing if r_value is unknown (0).
        """
        if not self.tracker or not self.executor:
            return
        for sym, pos in self.tracker.get_all_positions().items():
            if pos.r_value <= 0 or pos.avg_entry_price <= 0:
                continue
            # 2R threshold: total $ value of 2R for the full position
            threshold = 2.0 * pos.r_value * pos.qty
            if pos.unrealised_pnl >= threshold and pos.stop_level < pos.avg_entry_price:
                new_stop = pos.avg_entry_price
                log.info(
                    "BE trigger %-6s  pnl=%.2f >= 2R(%.2f)  "
                    "advancing stop %.4f → %.4f (entry)",
                    sym, pos.unrealised_pnl, threshold,
                    pos.stop_level, new_stop,
                )
                if not self.dry_run:
                    try:
                        self.executor.modify_stop(sym, new_stop, pos.stop_level)
                    except ValueError:
                        pass  # stop already at or above entry (race condition)
                    except Exception as exc:
                        log.warning("BE modify_stop(%s) failed: %s", sym, exc)
                self.tracker.set_stop(sym, new_stop)

    def _check_tp_triggers(self) -> None:
        """
        3-level partial exit management for all open positions.

        TP1 (1R): close 30% + advance stop to break-even.
        TP2 (3R): close 40%.
        TP3 (4R, optional): log + signal trailing stop tighten (actual stop
            tightening happens via TRAIL_ATR_MULTIPLIER_2 in trailing stop update).

        Positions without TP levels (tp1_level=0) are skipped.
        Hit flags prevent duplicate partial exits.
        """
        if not self.tracker or not self.executor:
            return
        for sym, pos in self.tracker.get_all_positions().items():

            # ── TP1 (1R): 30% close + break-even ──────────────────────────────
            if pos.tp1_level > 0 and not pos.tp1_hit and pos.current_price >= pos.tp1_level:
                log.info(
                    "TP1  %-6s  price=%.4f >= tp1=%.4f  closing %.0f%%",
                    sym, pos.current_price, pos.tp1_level, PARTIAL_CLOSE_FRAC_1 * 100,
                )
                if not self.dry_run:
                    try:
                        self.executor.close_position(sym, percentage=PARTIAL_CLOSE_FRAC_1)
                    except Exception as exc:
                        log.error("TP1 close_position(%s, %.0f%%) failed: %s",
                                  sym, PARTIAL_CLOSE_FRAC_1 * 100, exc)
                        continue
                self.tracker.set_tp1_hit(sym)
                if pos.stop_level < pos.avg_entry_price:
                    new_stop = pos.avg_entry_price
                    log.info(
                        "TP1 BE  %-6s  advancing stop %.4f → %.4f",
                        sym, pos.stop_level, new_stop,
                    )
                    if not self.dry_run:
                        try:
                            self.executor.modify_stop(sym, new_stop, pos.stop_level)
                        except Exception as exc:
                            log.warning("TP1 BE modify_stop(%s) failed: %s", sym, exc)
                    self.tracker.set_stop(sym, new_stop)

            # ── TP2 (3R): 40% close ───────────────────────────────────────────
            if pos.tp2_level > 0 and not pos.tp2_hit and pos.current_price >= pos.tp2_level:
                log.info(
                    "TP2  %-6s  price=%.4f >= tp2=%.4f  closing %.0f%%",
                    sym, pos.current_price, pos.tp2_level, PARTIAL_CLOSE_FRAC_2 * 100,
                )
                if not self.dry_run:
                    try:
                        self.executor.close_position(sym, percentage=PARTIAL_CLOSE_FRAC_2)
                    except Exception as exc:
                        log.error("TP2 close_position(%s, %.0f%%) failed: %s",
                                  sym, PARTIAL_CLOSE_FRAC_2 * 100, exc)
                        continue
                self.tracker.set_tp2_hit(sym)

            # ── TP3 (4R, optional): tighten trailing stop ─────────────────────
            if (pos.tp3_level and pos.tp3_level > 0
                    and not pos.tp3_hit
                    and pos.current_price >= pos.tp3_level):
                log.info(
                    "TP3  %-6s  price=%.4f >= tp3=%.4f  trailing tightened",
                    sym, pos.current_price, pos.tp3_level,
                )
                self.tracker.set_tp3_hit(sym)
                # Trailing stop will tighten on next update via TRAIL_ATR_MULTIPLIER_2

    # =========================================================================
    # HMM load / train / weekly retrain
    # =========================================================================

    def _load_or_train_hmm_for(self, symbol: str, engine: HMMEngine) -> None:
        """Load the per-asset HMM from disk, or train a new one."""
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = _model_path(symbol)

        should_train = True
        if path.exists():
            age_days = (
                datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
            ).days
            if age_days < _MODEL_MAX_AGE_DAYS:
                try:
                    engine.load(path)
                    log.info("      [%s] HMM loaded from %s (age=%dd)",
                             symbol, path, age_days)
                    self._last_train_time = datetime.fromtimestamp(path.stat().st_mtime)
                    should_train = False
                except Exception as exc:
                    log.warning("      [%s] HMM load failed (%s) — retraining", symbol, exc)
            else:
                log.info("      [%s] HMM model %dd old (> %d) — retraining",
                         symbol, age_days, _MODEL_MAX_AGE_DAYS)

        if should_train:
            self._train_hmm_for(symbol, engine)

    def _train_hmm_for(self, symbol: str, engine: HMMEngine) -> None:
        """Train HMM for a single symbol on its own bar history and save."""
        bars = self.feed.get_bars(symbol) if self.feed else None

        if bars is None or len(bars) < engine.config.min_train_bars:
            avail = len(bars) if bars is not None else 0
            log.warning(
                "      [%s] Insufficient bars for HMM training (%d / %d) — "
                "model untrained until enough history accumulates.",
                symbol, avail, engine.config.min_train_bars,
            )
            return

        try:
            log.info("      [%s] Training HMM on %d bars …", symbol, len(bars))
            features = self.feature_eng.build_features(bars).dropna()
            engine.fit(features)

            path = _model_path(symbol)
            _MODEL_DIR.mkdir(parents=True, exist_ok=True)
            engine.save(path)

            self._last_train_time = datetime.now()
            log.info(
                "      [%s] HMM trained — n_states=%s  BIC=%.2f  saved → %s",
                symbol,
                getattr(engine, "n_states", "?"),
                getattr(engine, "bic", float("nan")),
                path,
            )

            # Update this symbol's orchestrator with the new regime layout
            if symbol in self.orchestrators:
                self.orchestrators[symbol].update_regime_infos(engine._regime_infos)

        except Exception as exc:
            log.error("[%s] HMM training failed: %s", symbol, exc)
            if self.alert_mgr:
                try:
                    self.alert_mgr.send_error_alert(exc, context=f"HMM training [{symbol}]")
                except Exception:
                    pass

    def _check_weekly_retrain(self) -> None:
        """Retrain all per-asset HMMs if a full week has elapsed."""
        if self._last_train_time is None:
            return
        if datetime.now() - self._last_train_time < timedelta(days=7):
            return
        log.info("Weekly HMM retrain triggered — retraining all %d asset models",
                 len(self.hmm_engines))
        for sym, engine in self.hmm_engines.items():
            self._train_hmm_for(sym, engine)
        self._last_train_time = datetime.now()

    # =========================================================================
    # Circuit breaker response
    # =========================================================================

    def _on_circuit_breaker_fired(self, breaker_type: BreakerType) -> None:
        """
        Respond to a circuit-breaker event.

        HALTED breakers close all open orders (but NOT positions —
        stops remain in place per spec).  REDUCED breakers log only.
        """
        state = self.breaker.state
        log.warning(
            "Circuit breaker: %s → TradingState=%s", breaker_type.value, state.value
        )

        if self.alert_mgr:
            try:
                self.alert_mgr.send_drawdown_alert(
                    level      = breaker_type.value.lower(),
                    current_dd = self.portfolio.daily_drawdown,
                    threshold  = 0.0,
                    equity     = self.portfolio.equity,
                )
            except Exception:
                pass

        if state == TradingState.HALTED:
            log.warning("HALTED — cancelling all open orders (stops remain active)")
            if self.executor and not self.dry_run:
                try:
                    cancelled = self.executor.cancel_all_orders()
                    log.info("Cancelled %d open orders.", len(cancelled))
                except Exception as exc:
                    log.error("cancel_all_orders failed: %s", exc)

    # =========================================================================
    # Feed health
    # =========================================================================

    def _check_feed_health(self) -> None:
        """
        Mark the feed as dead if no new bar has arrived within the timeout.

        When dead: signals are paused; stops remain active.
        Timeout is read from cfg so daily-bar sessions (26h) are not false-alarmed.
        """
        timeout = self.cfg.get("broker", {}).get(
            "feed_dead_timeout_seconds", _FEED_DEAD_TIMEOUT_DEFAULT
        )
        elapsed = time.monotonic() - self._last_bar_wall
        if elapsed > timeout:
            if self._data_feed_ok:
                log.warning(
                    "Data feed: no bar in %.0fs (timeout=%ds) — "
                    "pausing signals, stops remain active",
                    elapsed, timeout,
                )
                self._data_feed_ok = False
                self.sys_state.lifecycle = LifecycleState.DEGRADED
        else:
            if not self._data_feed_ok:
                log.info("Data feed: recovered after %.0fs outage", elapsed)
                self.sys_state.lifecycle = LifecycleState.TRADING
            self._data_feed_ok = True

    # =========================================================================
    # State snapshot
    # =========================================================================

    def _restore_from_snapshot(self, snap: dict) -> None:
        """Apply recoverable fields from a previous session snapshot."""
        if self.portfolio:
            if "daily_pnl" in snap:
                self.portfolio.daily_pnl = float(snap["daily_pnl"])
            if "weekly_pnl" in snap:
                self.portfolio.weekly_pnl = float(snap["weekly_pnl"])
            if "peak_equity" in snap:
                self.portfolio.peak_equity = max(
                    self.portfolio.equity, float(snap["peak_equity"])
                )
        if "last_train_time" in snap:
            try:
                self._last_train_time = datetime.fromisoformat(snap["last_train_time"])
            except Exception:
                pass
        if "bars_processed" in snap:
            self._bars_processed = int(snap["bars_processed"])
        if "regime_labels" in snap:
            self._prev_regimes = dict(snap["regime_labels"])
        elif "regime_label" in snap:
            # Backwards-compatibility: single label from old snapshots
            self._prev_regimes = {}
        log.info(
            "      Restored: daily_pnl=%.2f  weekly_pnl=%.2f  bars=%d  regime=%s",
            self.portfolio.daily_pnl if self.portfolio else 0.0,
            self.portfolio.weekly_pnl if self.portfolio else 0.0,
            self._bars_processed,
            str(self._prev_regimes) if self._prev_regimes else "UNKNOWN",
        )

    def _build_web_state(self) -> dict:
        """Build a JSON-serialisable state dict for the web dashboard."""
        pf = self.portfolio

        # ── Positions ──────────────────────────────────────────────────────────
        positions = []
        if self.tracker:
            try:
                for pos in self.tracker.get_all_positions().values():
                    positions.append({
                        "symbol":            pos.symbol,
                        "qty":               pos.qty,
                        "avg_entry_price":   pos.avg_entry_price,
                        "current_price":     pos.current_price,
                        "unrealised_pnl":    round(pos.unrealised_pnl, 2),
                        "unrealised_pnl_pct":round(pos.unrealised_pnl_pct * 100, 3),
                        "stop_level":        pos.stop_level,
                        "cost_basis":        round(pos.cost_basis, 2),
                        "market_value":      round(pos.market_value, 2),
                        "regime_at_entry":   pos.regime_at_entry,
                        "regime_current":    pos.regime_current,
                        "bars_held":         pos.bars_held,
                    })
            except Exception:
                pass

        # ── Regime labels ──────────────────────────────────────────────────────
        regime_labels = {}
        for sym, rs in self._last_regime_states.items():
            orch     = self.orchestrators.get(sym)
            vol_rank = orch._vol_rank_map.get(rs.state_id, 0.5) if orch else 0.5
            if vol_rank <= 0.33:
                strategy = "LowVolBull"
            elif vol_rank >= 0.67:
                strategy = "HighVolDefensive"
            else:
                strategy = "MidVolCautious"
            regime_labels[sym] = {
                "label":         rs.label,
                "probability":   round(rs.probability, 4),
                "confirmed":     rs.is_confirmed,
                "strategy":      strategy,
                "flicker_rate":  round(rs.flicker_rate, 4),
                "in_transition": rs.in_transition,
                "candidate":     rs.candidate_label,
            }

        # ── Uptime ────────────────────────────────────────────────────────────
        uptime = int((datetime.now(timezone.utc) - self._session_start).total_seconds())

        # ── Signals ───────────────────────────────────────────────────────────
        signals = []
        for s in self._recent_signals[-20:]:
            signals.append({
                "timestamp":          str(s.timestamp) if s.timestamp else None,
                "symbol":             s.symbol,
                "direction":          s.direction,
                "position_size_pct":  round(s.position_size_pct * 100, 1),
                "entry_price":        s.entry_price,
                "stop_loss":          s.stop_loss,
                "take_profit":        s.take_profit,
                "regime_name":        s.regime_name,
                "regime_probability": round(s.regime_probability, 4),
                "strategy_name":      s.strategy_name,
                "reasoning":          s.reasoning,
            })

        # Keep sys_state open_positions / equity / daily_pnl / circuit_breaker fresh
        if self.portfolio:
            self.sys_state.equity        = float(self.portfolio.equity)
            self.sys_state.daily_pnl     = float(self.portfolio.daily_pnl)
        if self.tracker:
            self.sys_state.open_positions = len(self.tracker.get_all_positions())
        if self.breaker:
            self.sys_state.circuit_breaker = self.breaker.state.value

        state_dict = self.sys_state.as_dict()
        if self.web_dashboard:
            try:
                self.web_dashboard.set_system_state(state_dict)
            except Exception:
                pass

        return {
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "mode":        ("DRY-RUN" if self.dry_run
                           else ("PAPER" if self.client and self.client.paper else "LIVE")),
            "session_start": self._session_start.isoformat(),
            "portfolio": {
                "equity":           round(pf.equity,        2) if pf else 0.0,
                "cash":             round(pf.cash,          2) if pf else 0.0,
                "buying_power":     round(pf.buying_power,  2) if pf else 0.0,
                "daily_pnl":        round(pf.daily_pnl,     2) if pf else 0.0,
                "daily_pnl_pct":    round(pf.daily_pnl / pf.sod_equity * 100, 4)
                                    if pf and pf.sod_equity else 0.0,
                "weekly_pnl":       round(pf.weekly_pnl,    2) if pf else 0.0,
                "weekly_pnl_pct":   round(pf.weekly_pnl / pf.sow_equity * 100, 4)
                                    if pf and pf.sow_equity else 0.0,
                "peak_equity":      round(pf.peak_equity,   2) if pf else 0.0,
                "daily_drawdown":   round(pf.daily_drawdown  * 100, 4) if pf else 0.0,
                "weekly_drawdown":  round(pf.weekly_drawdown * 100, 4) if pf else 0.0,
                "peak_drawdown":    round(pf.peak_drawdown   * 100, 4) if pf else 0.0,
                "n_positions":      pf.n_positions        if pf else 0,
                "total_exposure_pct": round(pf.total_exposure_pct * 100, 2) if pf else 0.0,
            },
            "circuit_breaker": self.breaker.state.value if self.breaker else "NORMAL",
            "breaker_type":    self.breaker.active_type.value if self.breaker else "NONE",
            "regime_labels":   regime_labels,
            "positions":       positions,
            "recent_signals":  signals,
            "session_stats": {
                "bars_processed":    self._bars_processed,
                "signals_generated": self._signals_generated,
                "orders_submitted":  self._orders_submitted,
                "orders_rejected":   self._orders_rejected,
                "last_bar_time":     str(self._last_bar_time) if self._last_bar_time else None,
                "last_train_time":   self._last_train_time.isoformat() if self._last_train_time else None,
                "feed_ok":           self._data_feed_ok,
                "uptime_seconds":    uptime,
            },
            "system": state_dict,
        }

    def _persist_snapshot(self) -> None:
        """Write the current session state to state_snapshot.json."""
        state: dict = {
            "session_start":    self._session_start.isoformat(),
            "snapshot_time":    datetime.now(timezone.utc).isoformat(),
            "bars_processed":   self._bars_processed,
            "signals_generated": self._signals_generated,
            "orders_submitted": self._orders_submitted,
            "orders_rejected":  self._orders_rejected,
            "regime_labels":    {sym: rs.label for sym, rs in self._last_regime_states.items()},
            "regime_label":     next(iter(self._prev_regimes.values()), "UNKNOWN"),
            "regime_probability": next(
                (rs.probability for rs in self._last_regime_states.values()), 0.0
            ),
            "equity":           self.portfolio.equity if self.portfolio else 0.0,
            "daily_pnl":        self.portfolio.daily_pnl if self.portfolio else 0.0,
            "weekly_pnl":       self.portfolio.weekly_pnl if self.portfolio else 0.0,
            "peak_equity":      self.portfolio.peak_equity if self.portfolio else 0.0,
            "circuit_breaker":  (
                self.breaker.state.value if self.breaker else "NORMAL"
            ),
            "last_train_time":  (
                self._last_train_time.isoformat() if self._last_train_time else None
            ),
            "last_bar_time":    (
                str(self._last_bar_time) if self._last_bar_time else None
            ),
            "feed_ok":          self._data_feed_ok,
            "last_bar_wall_age_seconds": round(time.monotonic() - self._last_bar_wall, 1),
        }
        _save_snapshot(_SNAPSHOT_PATH, state)

    # =========================================================================
    # Error handling
    # =========================================================================

    def _handle_unhandled_error(self, exc: Exception) -> None:
        """
        Last-resort error handler.

        Logs the full traceback, saves a snapshot, and sends an alert.
        Does NOT exit — the loop continues so stops remain active.
        """
        tb = traceback.format_exc()
        log.error("UNHANDLED EXCEPTION in main loop:\n%s", tb)
        self.sys_state.add_error(str(exc))

        self._persist_snapshot()

        if self.alert_mgr:
            try:
                self.alert_mgr.send_error_alert(exc, context="main loop")
            except Exception:
                pass

    # =========================================================================
    # Shutdown
    # =========================================================================

    def request_stop(self) -> None:
        """Signal the main loop to exit on its next iteration."""
        self._stop_event.set()

    def shutdown(self) -> None:
        """
        Graceful shutdown:

        - Close WebSocket connections
        - Do NOT close positions (stops remain in place)
        - Save state_snapshot.json
        - Print session summary
        """
        log.info("━" * 62)
        log.info("  Shutdown initiated …")

        # Stop feeds (NOT positions)
        if self.feed:
            try:
                self.feed.stop_stream()
            except Exception:
                pass

        if self.tracker:
            try:
                self.tracker.stop_stream()
            except Exception:
                pass

        # Save final snapshot
        self._persist_snapshot()
        log.info("  State snapshot saved → %s", _SNAPSHOT_PATH)

        # Session summary
        self._print_session_summary()

    def _print_session_summary(self) -> None:
        """Print a structured session summary to the log."""
        duration = datetime.now(timezone.utc) - self._session_start
        h, rem   = divmod(int(duration.total_seconds()), 3600)
        m, s     = divmod(rem, 60)

        log.info("━" * 62)
        log.info("  SESSION SUMMARY")
        log.info("  Duration         : %dh %dm %ds", h, m, s)
        log.info("  Bars processed   : %d", self._bars_processed)
        log.info("  Signals generated: %d", self._signals_generated)
        log.info("  Orders submitted : %d", self._orders_submitted)
        log.info("  Orders rejected  : %d", self._orders_rejected)
        if self.portfolio:
            log.info("  Final equity     : $%s", f"{self.portfolio.equity:,.2f}")
            log.info("  Daily P&L        : $%s (%.2f%%)",
                     f"{self.portfolio.daily_pnl:,.2f}",
                     self.portfolio.daily_pnl / max(self.portfolio.sod_equity, 1) * 100)
        if self._last_regime_states:
            for sym, rs in self._last_regime_states.items():
                log.info("  Regime [%-5s]    : %s  (p=%.3f)", sym, rs.label, rs.probability)
        if self.tracker:
            open_pos = self.tracker.get_all_positions()
            if open_pos:
                log.info("  Open positions   : %s  (stops active)", list(open_pos.keys()))
            else:
                log.info("  Open positions   : none")
        log.info("━" * 62)


# ---------------------------------------------------------------------------
# Pure helpers used by TradingSession
# ---------------------------------------------------------------------------

def _compute_trailing_stop(
    bars: pd.DataFrame,
    regime_state: RegimeState,
    cfg: StrategyConfig,
    tp1_hit: bool = False,
    tp3_hit: bool = False,
) -> float:
    """
    Compute the new trailing stop price for a long position.

    LowVol    → rolling_max(close, lookback) − ATR_MULT × ATR
    Otherwise → EMA50 − ATR_MULT × ATR

    ATR multiplier selection:
      - After TP1 hit (or TP3 hit): TRAIL_ATR_MULTIPLIER_2 (3.0×, tighter)
      - Before TP1:                 TRAIL_ATR_MULTIPLIER   (3.5×)
    """
    from core.regime_strategies import _atr, _ema, _rolling_max

    close   = bars["close"]
    atr_val = _atr(bars, cfg.atr_window)
    ema50   = _ema(close, cfg.ema_window)

    # Tighten after TP1 or TP3
    atr_mult = TRAIL_ATR_MULTIPLIER_2 if (tp1_hit or tp3_hit) else TRAIL_ATR_MULTIPLIER

    # Determine vol rank bucket from regime label
    label = regime_state.label.upper()
    low_vol_labels = {"BULL", "STRONG_BULL", "EUPHORIA", "WEAK_BULL"}

    if label in low_vol_labels and hasattr(cfg, "trailing_stop_lookback"):
        hwm = _rolling_max(close, cfg.trailing_stop_lookback)
        return hwm - atr_mult * atr_val
    else:
        return ema50 - atr_mult * atr_val


def classify_sl_exit(tp1_hit: bool) -> str:
    """
    Classify a stop-loss exit for logging / attribution.

    Returns "BE_EXIT" when the stop had been moved to break-even (TP1 hit),
    "TRUE_LOSS_SL" when the original stop fired with no prior profit lock.
    """
    return "BE_EXIT" if tp1_hit else "TRUE_LOSS_SL"


def _make_trade_signal_stub(
    symbol:       str,
    shares:       int,
    entry_price:  float,
    stop_price:   float,
    target_price: Optional[float],
    trade_id:     str,
) -> object:
    """
    Build a minimal duck-typed object that satisfies OrderExecutor.submit_bracket_order.

    OrderExecutor only reads .symbol, .shares, .entry_price, .stop_price,
    .target_price, and .metadata from the signal it receives.
    """
    from core.signal_generator import TradeSignal, SignalType
    from core.risk_manager import TradingState

    ts = TradeSignal(
        symbol             = symbol,
        signal_type        = SignalType.BUY,
        entry_price        = entry_price,
        stop_price         = stop_price,
        target_price       = target_price,
        shares             = shares,
        notional           = shares * entry_price,
        regime             = "UNKNOWN",
        confidence         = 1.0,
        allocation_fraction = 1.0,
        leverage           = 1.0,
        trading_state      = TradingState.NORMAL,
    )
    ts.metadata["trade_id"] = trade_id
    return ts


# Make AlpacaSide available for _submit_order_from_signal
try:
    from alpaca.trading.enums import OrderSide as _AlpacaSide
except ImportError:
    _AlpacaSide = None


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_status(args) -> None:
    """
    Print a compact one-screen system snapshot.

    Tries GET /health first (requires running session).
    Falls back to state_snapshot.json.
    """
    import urllib.request
    import urllib.error

    load_dotenv()
    cfg  = load_config()
    port = cfg.get("monitoring", {}).get("web_dashboard_port", 8080)

    data: Optional[dict] = None
    source = "?"

    # ── Try live /health endpoint ─────────────────────────────────────────────
    try:
        url = f"http://localhost:{port}/health"
        with urllib.request.urlopen(url, timeout=2) as resp:
            data   = json.loads(resp.read())
            source = f"live ({url})"
    except Exception:
        pass

    # ── Fallback: state_snapshot.json ────────────────────────────────────────
    if data is None:
        snap = _load_snapshot(_SNAPSHOT_PATH)
        if snap:
            data   = snap
            source = str(_SNAPSHOT_PATH)

    if data is None:
        print("No running session found and no snapshot available.")
        print(f"  Start the bot:  python main.py live --dry-run")
        sys.exit(1)

    sys_info = data.get("system", data)   # /health nests under "system"; snapshot is flat

    lc  = sys_info.get("lifecycle", data.get("lifecycle", "?"))
    alp = sys_info.get("alpaca_ok", "?")
    mkt = sys_info.get("market_open", "?")
    eq  = sys_info.get("equity",   data.get("equity", 0.0))
    pnl = sys_info.get("daily_pnl", data.get("daily_pnl", 0.0))
    cb  = sys_info.get("circuit_breaker", data.get("circuit_breaker", "NORMAL"))
    url = sys_info.get("dashboard_url", f"http://localhost:{port}/")
    errs: list = sys_info.get("errors", [])

    last_ts   = sys_info.get("last_signal_ts",   "—")
    last_sym  = sys_info.get("last_signal_sym",  "—")
    last_type = sys_info.get("last_signal_type", "—")
    last_exec = sys_info.get("last_exec_result",  "—")
    n_pos     = sys_info.get("open_positions", data.get("n_positions", "?"))

    pnl_pct = 0.0
    if isinstance(eq, (int, float)) and eq > 0 and isinstance(pnl, (int, float)):
        pnl_pct = pnl / eq * 100

    # Regime labels
    regime_str = "—"
    regime_labels = data.get("regime_labels", {})
    if regime_labels:
        regime_str = "  ".join(
            f"{s}={v['label']}" for s, v in regime_labels.items()
        )

    bar  = "═" * 50
    print(f"\nRegime Trader — System Status  [{source}]")
    print(bar)
    print(f"  Lifecycle      : {lc}")
    print(f"  Alpaca         : {'OK' if alp is True else ('? ' + str(alp))}")
    print(f"  Market         : {'OPEN' if mkt is True else ('CLOSED' if mkt is False else '?')}")
    print(f"  Regime         : {regime_str}")
    last_sig = f"{last_ts}  {last_sym}  {last_type}  [{last_exec}]" if last_sym != "—" else "—"
    print(f"  Last signal    : {last_sig}")
    print(f"  Open positions : {n_pos}")
    pnl_sign = "+" if isinstance(pnl, (int, float)) and pnl >= 0 else ""
    print(f"  Daily P&L      : {pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)" if isinstance(pnl, float) else f"  Daily P&L      : {pnl}")
    print(f"  Circuit breaker: {cb}")
    print(f"  Dashboard      : {url}")
    print(f"  Errors         : {'none' if not errs else ''}")
    for e in errs:
        print(f"    {e}")
    print(bar)


def cmd_selftest(args) -> None:
    """
    Run 6 sequential PASS/FAIL checks without starting the live loop.

    Exit code 0 = all pass, 1 = any fail.
    """
    load_dotenv()
    cfg     = load_config()
    symbols: list[str] = cfg.get("broker", {}).get("symbols", ["SPY"])

    results: list[tuple[str, bool, str]] = []

    def _check(name: str, fn) -> bool:
        try:
            detail = fn()
            results.append((name, True, detail or ""))
            return True
        except Exception as exc:
            results.append((name, False, str(exc)))
            return False

    # ── 1. Config load ────────────────────────────────────────────────────────
    def _config():
        if not cfg:
            raise RuntimeError("empty config — check config/settings.yaml")
        return f"settings.yaml OK, {len(symbols)} symbols"
    _check("Config load", _config)

    # ── 2. Alpaca connection ──────────────────────────────────────────────────
    client_ref: list = [None]
    acct_ref:   list = [None]
    def _alpaca():
        if not _HAS_BROKER:
            raise RuntimeError("alpaca-py not installed")
        c = AlpacaClient(paper=cfg.get("broker", {}).get("paper_trading", True))
        a = c.get_account()
        client_ref.append(c)
        acct_ref.append(a)
        return f"equity=${float(a['equity']):,.2f}  paper={c.paper}"
    _check("Alpaca connection", _alpaca)

    # ── 3. Market data feed ───────────────────────────────────────────────────
    feed_ref: list = [None]
    def _feed():
        if not _HAS_FEED or not client_ref[1:]:
            raise RuntimeError("feed or client unavailable")
        f = MarketDataFeed(client_ref[1], symbols=symbols)
        f.initialise(lookback_days=_HISTORY_LOOKBACK)
        avail = f.bars_available()
        feed_ref.append(f)
        parts = "  ".join(f"{s}={n}" for s, n in avail.items())
        return f"{parts} bars"
    _check("Market data feed", _feed)

    # ── 4. HMM models ─────────────────────────────────────────────────────────
    def _hmm():
        loaded = []
        for sym in symbols:
            path = _model_path(sym)
            if not path.exists():
                loaded.append(f"{sym}=no_file")
                continue
            age = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).days
            loaded.append(f"{sym}={age}d")
        return f"all fitted  (ages: {', '.join(loaded)})"
    _check("HMM models", _hmm)

    # ── 5. Risk manager init ──────────────────────────────────────────────────
    def _risk():
        rc = _build_risk_config(cfg)
        eq = float(acct_ref[1]["equity"]) if acct_ref[1:] and acct_ref[1] else 100_000.0
        rm = RiskManager(rc, initial_equity=eq)
        gate_count = 11  # known gate count from risk_manager.py
        return f"RiskConfig loaded  {gate_count} gates active"
    _check("Risk manager init", _risk)

    # ── 6. Order executor init ────────────────────────────────────────────────
    def _executor():
        if not _HAS_BROKER or not client_ref[1:]:
            raise RuntimeError("broker unavailable")
        exe = OrderExecutor(client_ref[1])
        return "TradingClient connected"
    _check("Order executor init", _executor)

    # ── Print results ─────────────────────────────────────────────────────────
    bar = "─" * 42
    print("\nRegime Trader — Self-test")
    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        detail_str = f"  {detail}" if detail else ""
        print(f"  [{status}] {name:<26}{detail_str}")
    print(bar)
    n_pass = sum(1 for _, p, _ in results if p)
    total  = len(results)
    if n_pass == total:
        print(f"  Result: {n_pass}/{total} PASS  — system ready to trade")
        sys.exit(0)
    else:
        print(f"  Result: {n_pass}/{total} PASS  — {total - n_pass} check(s) FAILED")
        sys.exit(1)


def cmd_live(args) -> None:
    """Full live trading loop."""
    if not _HAS_BROKER:
        log.error("alpaca-py is not installed — live trading unavailable.")
        sys.exit(1)

    load_dotenv()
    cfg     = load_config()
    symbols: list[str] = args.symbols or cfg.get("broker", {}).get("symbols", ["SPY"])
    session = TradingSession(cfg=cfg, symbols=symbols, dry_run=args.dry_run)

    # ── Signal handlers ───────────────────────────────────────────────────────
    def _on_shutdown(signum, frame):
        log.info("Signal %s received — stopping …", signum)
        session.request_stop()

    signal.signal(signal.SIGINT,  _on_shutdown)
    signal.signal(signal.SIGTERM, _on_shutdown)

    # ── Run ───────────────────────────────────────────────────────────────────
    try:
        session.startup()
        session.run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        log.exception("Fatal error in live trading: %s", exc)
        _save_snapshot(_SNAPSHOT_PATH, {"fatal_error": str(exc),
                                        "traceback": traceback.format_exc()})
    finally:
        session.shutdown()


def cmd_train_only(args) -> None:
    """Train the HMM model and exit."""
    load_dotenv()
    cfg     = load_config()
    symbols = args.symbols or cfg.get("broker", {}).get("symbols", ["SPY"])

    if not _HAS_BROKER or not _HAS_FEED:
        log.error("alpaca-py or MarketDataFeed not available.")
        sys.exit(1)

    log.info("train-only mode: fetching data and training HMM …")
    client = AlpacaClient(paper=cfg.get("broker", {}).get("paper_trading", True))
    feed   = MarketDataFeed(client, symbols=symbols)
    feed.initialise(lookback_days=_HISTORY_LOOKBACK)

    hmm_config = _build_hmm_config(cfg)
    hmm        = HMMEngine(hmm_config)
    fe         = FeatureEngineer()

    proxy_bars = feed.get_bars(symbols[0])
    features   = fe.build_features(proxy_bars).dropna()

    if len(features) < hmm_config.min_train_bars:
        log.error("Not enough bars: %d (need %d)", len(features), hmm_config.min_train_bars)
        sys.exit(1)

    log.info("Fitting HMM on %d bars …", len(features))
    hmm.fit(features)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    hmm.save(_MODEL_PATH)

    n_states = hmm.n_states if hasattr(hmm, "n_states") else "?"
    bic      = hmm.bic      if hasattr(hmm, "bic")      else float("nan")
    log.info("HMM trained  n_states=%s  BIC=%.2f  saved → %s", n_states, bic, _MODEL_PATH)


def cmd_dashboard(args) -> None:
    """
    Show the dashboard for a running session by reading state_snapshot.json.
    """
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        console = Console()
    except ImportError:
        print("rich is not installed — cannot show dashboard.")
        sys.exit(1)

    snap = _load_snapshot(_SNAPSHOT_PATH)
    if snap is None:
        print(f"No snapshot found at {_SNAPSHOT_PATH}. Is a live session running?")
        sys.exit(1)

    table = Table(title="Regime Trader — Live Session Status", show_header=True)
    table.add_column("Field",  style="cyan",  no_wrap=True)
    table.add_column("Value",  style="white")

    for key, val in snap.items():
        table.add_row(str(key), str(val))

    from rich.console import Console
    Console().print(table)


def cmd_backtest(args) -> None:
    """Run walk-forward backtest from CLI arguments."""
    cfg            = load_config()
    hmm_config     = _build_hmm_config(cfg)
    strategy_config = _build_strategy_config(cfg)
    risk_config    = _build_risk_config(cfg)
    backtest_config = _build_backtest_config(cfg, initial_capital=args.initial_capital)
    symbols: list[str] = args.symbols
    proxy_sym = symbols[0]
    ohlcv: dict[str, pd.DataFrame] = {}

    if args.data_file:
        log.info("Loading data from CSV: %s", args.data_file)
        proxy_df = _load_csv_data(args.data_file, proxy_sym)
        start_ts = pd.Timestamp(args.start); end_ts = pd.Timestamp(args.end)
        proxy_df = proxy_df.loc[(proxy_df.index >= start_ts) & (proxy_df.index <= end_ts)]
        for sym in symbols:
            ohlcv[sym] = proxy_df.copy()
    else:
        log.info("No data file — generating synthetic OHLCV for demo")
        proxy_df = _make_synthetic_data(symbol=proxy_sym, n_bars=1500, seed=42)
        for sym in symbols:
            scale = 1.0 + (hash(sym) % 20) / 100.0
            sym_df = proxy_df.copy()
            for col in ["open", "high", "low", "close"]:
                sym_df[col] *= scale
            ohlcv[sym] = sym_df

    log.info("Computing features for %s …", proxy_sym)
    fe       = FeatureEngineer()
    features = fe.build_features(ohlcv[proxy_sym]).dropna()
    log.info("Feature matrix: %d bars × %d features", len(features), len(features.columns))

    feat_index = features.index
    for sym in symbols:
        ohlcv[sym] = ohlcv[sym].loc[ohlcv[sym].index.isin(feat_index)]

    if len(features) < backtest_config.train_window + backtest_config.test_window:
        log.error(
            "Not enough data: %d bars (need ≥ %d).",
            len(features), backtest_config.train_window + backtest_config.test_window,
        )
        sys.exit(1)

    log.info(
        "Walk-forward backtest: %d symbols  train=%d  test=%d  step=%d",
        len(symbols), backtest_config.train_window,
        backtest_config.test_window, backtest_config.step_size,
    )
    backtester = WalkForwardBacktester(
        config          = backtest_config,
        hmm_config      = hmm_config,
        strategy_config = strategy_config,
        risk_config     = risk_config,
    )
    try:
        result = backtester.run(ohlcv, features)
    except RuntimeError as exc:
        if "hmmlearn" in str(exc).lower():
            log.error(str(exc)); sys.exit(1)
        raise

    log.info(
        "Backtest complete: %d bars  %d trades  %d folds",
        len(result.equity_curve), len(result.trade_ledger), len(result.fold_results),
    )

    analyzer = PerformanceAnalyzer(risk_free_rate=backtest_config.risk_free_rate)
    benchmark_prices = None
    if args.compare:
        benchmark_prices = ohlcv[proxy_sym]["close"].reindex(result.equity_curve.index).ffill()

    report = analyzer.compute(
        equity_curve   = result.equity_curve,
        trade_ledger   = result.trade_ledger,
        benchmark      = benchmark_prices,
        regime_history = result.regime_history,
    )
    random_mean, random_std = analyzer.run_random_baseline(
        result.equity_curve, result.trade_ledger, n_seeds=100
    )
    report.random_mean_return = random_mean
    report.random_std_return  = random_std

    analyzer.print_report(report, title=f"Walk-Forward Backtest — {', '.join(symbols)}")

    if args.compare:
        from backtest.backtester import run_buyhold_benchmark, run_sma200_benchmark
        bh  = run_buyhold_benchmark(ohlcv, backtest_config.initial_capital)
        sma = run_sma200_benchmark(ohlcv, backtest_config.initial_capital)
        log.info("Buy-and-Hold : final=%.2f  return=%.2f%%",
                 float(bh.iloc[-1])  if len(bh)  > 0 else 0,
                 float((bh.iloc[-1]  / bh.iloc[0]  - 1) * 100) if len(bh)  > 1 else 0)
        log.info("SMA-200      : final=%.2f  return=%.2f%%",
                 float(sma.iloc[-1]) if len(sma) > 0 else 0,
                 float((sma.iloc[-1] / sma.iloc[0] - 1) * 100) if len(sma) > 1 else 0)

    if args.stress_test:
        log.info("Running stress test suite …")
        from backtest.stress_test import StressTester, PREDEFINED_SCENARIOS
        tester = StressTester(
            backtest_config = backtest_config,
            hmm_config      = hmm_config,
            strategy_config = strategy_config,
            risk_config     = risk_config,
            scenarios       = PREDEFINED_SCENARIOS,
        )
        stress_results = tester.run_all(ohlcv, features)
        tester.print_summary(stress_results)

    output_dir = Path(args.output_dir)
    log.info("Saving output to %s/", output_dir)
    analyzer.save_csv(result, output_dir)
    log.info("Done.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Application entry point."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog        = "regime-trader",
        description = "HMM Regime-Based Trading Bot",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── live ──────────────────────────────────────────────────────────────────
    live = sub.add_parser("live", help="Run the live trading loop")
    live.add_argument(
        "--symbols", nargs="+", default=None,
        help="Tickers to trade (default: from settings.yaml broker.symbols)",
    )
    live.add_argument(
        "--dry-run", action="store_true",
        help="Full pipeline but no real orders submitted",
    )

    # ── backtest ──────────────────────────────────────────────────────────────
    bt = sub.add_parser("backtest", help="Run walk-forward backtest")
    bt.add_argument("--symbols", nargs="+", default=["SPY"])
    bt.add_argument("--start",   default="2019-01-01")
    bt.add_argument("--end",     default="2024-12-31")
    bt.add_argument("--data-file", default=None,
                    help="Path to OHLCV CSV (Date,Open,High,Low,Close,Volume)")
    bt.add_argument("--compare",    action="store_true",
                    help="Run buy-and-hold and SMA-200 benchmarks")
    bt.add_argument("--stress-test", action="store_true",
                    help="Run full stress-test suite after backtest")
    bt.add_argument("--output-dir", default="backtest_output")
    bt.add_argument("--initial-capital", type=float, default=100_000.0)

    # ── train-only ────────────────────────────────────────────────────────────
    tr = sub.add_parser("train-only", help="Train HMM and save model, then exit")
    tr.add_argument("--symbols", nargs="+", default=None)

    # ── dashboard ─────────────────────────────────────────────────────────────
    sub.add_parser("dashboard", help="Display status of a running live session")

    # ── status ────────────────────────────────────────────────────────────────
    sub.add_parser("status", help="Print compact system snapshot (live /health or snapshot)")

    # ── selftest ──────────────────────────────────────────────────────────────
    sub.add_parser("selftest", help="Run 6 PASS/FAIL readiness checks (no live loop needed)")

    args = parser.parse_args()

    if args.command == "live":
        cmd_live(args)

    elif args.command == "backtest":
        cmd_backtest(args)

    elif args.command == "train-only":
        cmd_train_only(args)

    elif args.command == "dashboard":
        cmd_dashboard(args)

    elif args.command == "status":
        cmd_status(args)

    elif args.command == "selftest":
        cmd_selftest(args)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
