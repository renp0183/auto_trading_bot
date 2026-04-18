"""
logger.py — Structured, levelled logging for the regime-trader application.

Four rotating log files (10 MB each, 30 backups ≈ 30 days of trading):
  logs/main.log    — general application events
  logs/trades.log  — order submissions, fills, and P&L events
  logs/alerts.log  — circuit breakers, drawdown, and system alerts
  logs/regime.log  — HMM regime changes and confidence updates

Every JSON log record contains a standard context envelope:
  timestamp, level, logger, message,
  regime, probability, equity, positions, daily_pnl,
  plus any ``extra`` kwargs the caller supplies.

The context fields are maintained in a thread-local ``LogContext`` store.
Call ``update_log_context(regime="BULL", equity=105_000, ...)`` from the
main loop and every subsequent log call will include those fields.

Usage::

    log = get_logger(__name__)
    update_log_context(regime="BULL", probability=0.87, equity=105_230)
    log.info("Bar processed", extra={"symbol": "SPY", "close": 520.3})

    trade_log = get_trade_logger()
    trade_log.info("Order filled", extra={"symbol": "SPY", "qty": 10, "price": 520.3})
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LOG_DIR       = Path("logs")
_MAX_BYTES     = 10 * 1024 * 1024   # 10 MB per file
_BACKUP_COUNT  = 30                  # 30 rotations ≈ 30 active-trading days

# Logger name hierarchy
_ROOT_LOGGER   = "regime_trader"
_TRADE_LOGGER  = "regime_trader.trades"
_ALERT_LOGGER  = "regime_trader.alerts"
_REGIME_LOGGER = "regime_trader.regime"

# Track which loggers have already been configured (prevent duplicate handlers)
_configured: set[str] = set()
_config_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Thread-local trading context (regime, equity, …)
# ---------------------------------------------------------------------------

_ctx = threading.local()


def update_log_context(**kwargs: Any) -> None:
    """
    Update the thread-local trading context used by ``JsonFormatter``.

    Call this from the main loop after each bar so every log record that
    fires during processing automatically carries current regime/equity.

    Parameters
    ----------
    **kwargs:
        Any combination of: ``regime``, ``probability``, ``equity``,
        ``positions``, ``daily_pnl``, ``trading_state``.

    Example::

        update_log_context(
            regime="BULL",
            probability=0.87,
            equity=105_230.0,
            positions=["SPY", "AAPL"],
            daily_pnl=340.0,
        )
    """
    ctx = _get_context()
    ctx.update(kwargs)


def get_log_context() -> dict:
    """Return a copy of the current thread-local trading context."""
    return dict(_get_context())


def clear_log_context() -> None:
    """Reset the thread-local context (call at session end or in tests)."""
    _ctx.data = {}


def _get_context() -> dict:
    if not hasattr(_ctx, "data"):
        _ctx.data = {}
    return _ctx.data


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """
    Emits one compact JSON object per log record.

    Every record includes:
      - ``timestamp``   — ISO-8601 UTC with microseconds
      - ``level``       — DEBUG / INFO / WARNING / ERROR / CRITICAL
      - ``logger``      — logger name (module path)
      - ``message``     — formatted log message
      - ``regime``      — current HMM regime label (from LogContext)
      - ``probability`` — regime posterior (from LogContext)
      - ``equity``      — portfolio equity (from LogContext)
      - ``positions``   — list of open symbols (from LogContext)
      - ``daily_pnl``   — day's P&L in USD (from LogContext)
      - Any extra fields passed via ``extra={}`` to the log call.
    """

    # Context fields to include from the thread-local store
    _CTX_FIELDS = ("regime", "probability", "equity", "positions",
                   "daily_pnl", "trading_state")

    def format(self, record: logging.LogRecord) -> str:
        ctx = _get_context()
        payload: dict[str, Any] = {
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "level":       record.levelname,
            "logger":      record.name,
            "message":     record.getMessage(),
            # Trading context
            "regime":      ctx.get("regime",      "UNKNOWN"),
            "probability": ctx.get("probability", 0.0),
            "equity":      ctx.get("equity",      0.0),
            "positions":   ctx.get("positions",   []),
            "daily_pnl":   ctx.get("daily_pnl",   0.0),
        }
        # Optional context fields
        for field in ("trading_state",):
            if field in ctx:
                payload[field] = ctx[field]

        # Caller-supplied extra fields (skip internal logging attrs)
        _skip = {
            "args", "asctime", "created", "exc_info", "exc_text",
            "filename", "funcName", "id", "levelname", "levelno",
            "lineno", "message", "module", "msecs", "msg", "name",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "thread", "threadName",
        }
        for key, val in record.__dict__.items():
            if key not in _skip and not key.startswith("_"):
                try:
                    json.dumps(val)     # cheap serializability check
                    payload[key] = val
                except (TypeError, ValueError):
                    payload[key] = str(val)

        # Exception info
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        try:
            return json.dumps(payload, default=str)
        except Exception:
            # Last-resort fallback — never drop a log record
            return json.dumps({"timestamp": payload["timestamp"],
                               "level": record.levelname,
                               "message": str(record.getMessage()),
                               "error": "json_serialization_failed"})


# ---------------------------------------------------------------------------
# Human-readable console formatter (for RichHandler)
# ---------------------------------------------------------------------------

class _ConsoleFormatter(logging.Formatter):
    """Compact human-readable formatter for the terminal handler."""

    _FMT = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    _DATE = "%H:%M:%S"

    def __init__(self):
        super().__init__(fmt=self._FMT, datefmt=self._DATE)


# ---------------------------------------------------------------------------
# Handler factory
# ---------------------------------------------------------------------------

def _make_file_handler(path: Path) -> logging.handlers.RotatingFileHandler:
    """Create a rotating JSON file handler at the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        filename   = str(path),
        maxBytes   = _MAX_BYTES,
        backupCount = _BACKUP_COUNT,
        encoding   = "utf-8",
    )
    handler.setFormatter(JsonFormatter())
    handler.setLevel(logging.DEBUG)
    return handler


def _make_console_handler() -> logging.Handler:
    """Create a coloured console handler using rich (falls back to StreamHandler)."""
    try:
        from rich.logging import RichHandler
        handler = RichHandler(
            rich_tracebacks = True,
            show_path       = False,
            markup          = True,
        )
        handler.setFormatter(logging.Formatter(fmt="%(message)s", datefmt="[%H:%M:%S]"))
    except ImportError:
        handler = logging.StreamHandler()
        handler.setFormatter(_ConsoleFormatter())
    handler.setLevel(logging.INFO)
    return handler


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------

def get_logger(
    name: str,
    level: int = logging.DEBUG,
    log_dir: Optional[Path] = None,
) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    On first call for a given ``name`` the logger is created and handlers
    are attached.  Subsequent calls return the same instance (idempotent).

    All loggers in the ``regime_trader.*`` hierarchy write to both the
    console and to ``logs/main.log``.  Specialised sub-loggers additionally
    write to their own rotating file.

    Parameters
    ----------
    name:
        Logger name, typically ``__name__`` of the calling module.
    level:
        Minimum logging level (default: ``logging.DEBUG``).
    log_dir:
        Directory for file handlers.  Defaults to ``logs/`` in the CWD.

    Returns
    -------
    Configured :class:`logging.Logger` instance.
    """
    log_dir = Path(log_dir) if log_dir else _LOG_DIR

    with _config_lock:
        logger = logging.getLogger(name)
        if name in _configured:
            return logger

        logger.setLevel(level)
        logger.propagate = False   # prevent double-logging to root

        # ── Console ───────────────────────────────────────────────────────────
        logger.addHandler(_make_console_handler())

        # ── main.log (all loggers under regime_trader.*) ──────────────────────
        if name == _ROOT_LOGGER or name.startswith(_ROOT_LOGGER + ".") or True:
            logger.addHandler(_make_file_handler(log_dir / "main.log"))

        # ── Specialised file per sub-logger ───────────────────────────────────
        _sub_files = {
            _TRADE_LOGGER:  "trades.log",
            _ALERT_LOGGER:  "alerts.log",
            _REGIME_LOGGER: "regime.log",
        }
        if name in _sub_files:
            logger.addHandler(_make_file_handler(log_dir / _sub_files[name]))

        _configured.add(name)
        return logger


def get_trade_logger(log_dir: Optional[Path] = None) -> logging.Logger:
    """
    Return the dedicated trades logger.

    Writes to ``logs/trades.log`` in addition to ``logs/main.log``.
    Use for order submissions, fills, cancellations, and P&L records.
    """
    return get_logger(_TRADE_LOGGER, log_dir=log_dir)


def get_alert_logger(log_dir: Optional[Path] = None) -> logging.Logger:
    """
    Return the dedicated alerts logger.

    Writes to ``logs/alerts.log`` in addition to ``logs/main.log``.
    Use for circuit-breaker events, drawdown breaches, and system alerts.
    """
    return get_logger(_ALERT_LOGGER, log_dir=log_dir)


def get_regime_logger(log_dir: Optional[Path] = None) -> logging.Logger:
    """
    Return the dedicated regime logger.

    Writes to ``logs/regime.log`` in addition to ``logs/main.log``.
    Use for HMM regime changes, confidence updates, and flicker events.
    """
    return get_logger(_REGIME_LOGGER, log_dir=log_dir)
