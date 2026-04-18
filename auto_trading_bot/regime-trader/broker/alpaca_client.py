"""
alpaca_client.py — Thin wrapper around the Alpaca REST and WebSocket APIs.

Responsibilities:
  - Authenticate using environment variables (ALPACA_API_KEY, ALPACA_SECRET_KEY,
    ALPACA_PAPER) loaded from .env — credentials are NEVER hardcoded.
  - Expose paper vs. live trading endpoint selection.
    Paper is the DEFAULT; switching to live requires interactive confirmation.
  - Provide account info, asset lookup, and market clock helpers.
  - Stream real-time quotes and bar data via WebSocket.
  - Health-check on startup; auto-reconnect with exponential back-off.
"""

from __future__ import annotations

import logging
import os
import time
import random
from pathlib import Path
from typing import Optional, Callable, Any

import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# alpaca-py lazy import — allows the rest of the project to import this
# module even when alpaca-py is not installed.
# ---------------------------------------------------------------------------
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.live import StockDataStream
    from alpaca.data.requests import (
        StockBarsRequest,
        StockLatestBarRequest,
        StockLatestQuoteRequest,
        StockSnapshotRequest,
    )
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL  = "https://api.alpaca.markets"

_LIVE_CONFIRM_PHRASE = "YES I UNDERSTAND THE RISKS"

# Exponential back-off parameters
_BACKOFF_BASE_S  = 2.0   # Initial retry interval (seconds)
_BACKOFF_MAX_S   = 60.0  # Maximum retry interval (seconds)
_BACKOFF_JITTER  = 0.25  # ±25 % random jitter to avoid thundering-herd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_bool(value: Any, default: bool = True) -> bool:
    """Parse a string / bool / int environment variable as a boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off", "")
    return default


def _timeframe_from_str(tf: str) -> "TimeFrame":
    """
    Convert a human-readable string (e.g. '1Day', '5Min') to an
    alpaca-py ``TimeFrame`` object.

    Raises ``ValueError`` for unrecognised strings.
    """
    if not _ALPACA_AVAILABLE:
        raise RuntimeError("alpaca-py is not installed. Run: pip install alpaca-py")
    mapping = {
        "1min":   TimeFrame(1,  TimeFrameUnit.Minute),
        "5min":   TimeFrame(5,  TimeFrameUnit.Minute),
        "15min":  TimeFrame(15, TimeFrameUnit.Minute),
        "30min":  TimeFrame(30, TimeFrameUnit.Minute),
        "1hour":  TimeFrame(1,  TimeFrameUnit.Hour),
        "4hour":  TimeFrame(4,  TimeFrameUnit.Hour),
        "1day":   TimeFrame(1,  TimeFrameUnit.Day),
        "1week":  TimeFrame(1,  TimeFrameUnit.Week),
        "1month": TimeFrame(1,  TimeFrameUnit.Month),
    }
    key = tf.strip().lower()
    if key not in mapping:
        raise ValueError(
            f"Unknown timeframe '{tf}'. "
            f"Valid options: {sorted(mapping.keys())}"
        )
    return mapping[key]


def _order_to_dict(order) -> dict:
    """Convert an alpaca-py ``Order`` object to a plain serialisable dict."""
    def _val(x):
        return x.value if hasattr(x, "value") else str(x) if x is not None else None

    return {
        "id":               str(order.id),
        "client_order_id":  str(order.client_order_id),
        "symbol":           order.symbol,
        "side":             _val(order.side),
        "type":             _val(order.type),
        "qty":              float(order.qty) if order.qty is not None else None,
        "filled_qty":       float(order.filled_qty) if order.filled_qty is not None else 0.0,
        "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price is not None else None,
        "limit_price":      float(order.limit_price) if order.limit_price is not None else None,
        "stop_price":       float(order.stop_price) if order.stop_price is not None else None,
        "status":           _val(order.status),
        "time_in_force":    _val(order.time_in_force),
        "order_class":      _val(getattr(order, "order_class", None)),
        "submitted_at":     order.submitted_at.isoformat() if order.submitted_at else None,
        "filled_at":        order.filled_at.isoformat() if order.filled_at else None,
        "canceled_at":      order.canceled_at.isoformat() if order.canceled_at else None,
        "legs":             [_order_to_dict(leg) for leg in (order.legs or [])],
    }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AlpacaClient:
    """
    Authenticated Alpaca client providing access to trading, data, and
    streaming endpoints.

    Credentials are read from the environment (via python-dotenv):
      - ``ALPACA_API_KEY``    — Alpaca API key
      - ``ALPACA_SECRET_KEY`` — Alpaca secret key
      - ``ALPACA_PAPER``      — "true" / "false" (default: "true")

    Paper trading is the DEFAULT.  Switching to live trading prints a warning
    and requires the user to type ``"YES I UNDERSTAND THE RISKS"`` before the
    client is created.

    A connectivity health-check is performed at construction time.  If the
    API is unreachable, the client retries with exponential back-off before
    raising ``ConnectionError``.

    Usage::

        client = AlpacaClient()
        account = client.get_account()
        bars = client.get_historical_bars("AAPL", "2023-01-01", "2024-01-01")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: Optional[bool] = None,
    ) -> None:
        """
        Parameters
        ----------
        api_key:
            Alpaca API key.  Defaults to ``ALPACA_API_KEY`` env var.
        secret_key:
            Alpaca secret key.  Defaults to ``ALPACA_SECRET_KEY`` env var.
        paper:
            ``True`` for paper-trading endpoint (default).  ``False`` for live.
            Defaults to the ``ALPACA_PAPER`` env var.
        """
        if not _ALPACA_AVAILABLE:
            raise RuntimeError(
                "alpaca-py is not installed. Run: pip install alpaca-py"
            )

        # Load .env file if present — use explicit path so this works regardless
        # of CWD (e.g. when launched by systemd with WorkingDirectory set).
        # override=False means env vars already set (e.g. via systemd EnvironmentFile) win.
        _env_file = Path(__file__).parent.parent / ".env"
        if _env_file.exists():
            load_dotenv(_env_file, override=False)

        self._api_key    = api_key    or os.getenv("ALPACA_API_KEY",    "")
        self._secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY", "")

        if not self._api_key or not self._secret_key:
            raise ValueError(
                "Alpaca credentials not found.  Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY in your .env file or pass them explicitly."
            )

        if paper is None:
            paper = _parse_bool(os.getenv("ALPACA_PAPER", "true"), default=True)

        # ── Live-trading confirmation gate ────────────────────────────────────
        if not paper:
            self._confirm_live_trading()

        self.paper     = paper
        self._base_url = PAPER_BASE_URL if paper else LIVE_BASE_URL

        logger.info(
            "AlpacaClient initialising  mode=%-5s  endpoint=%s",
            "PAPER" if paper else "LIVE",
            self._base_url,
        )

        self._trading_client = TradingClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
            paper=paper,
        )
        self._data_client = StockHistoricalDataClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
        )
        self._stream: Optional[StockDataStream] = None

        # Verify connectivity; raises ConnectionError after repeated failures
        self._health_check(max_attempts=5)

    # -------------------------------------------------------------------------
    # Startup helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _confirm_live_trading() -> None:
        """
        Demand explicit typed confirmation before enabling live trading.

        Raises ``RuntimeError`` if the user does not type the confirmation
        phrase exactly.
        """
        print(
            "\n"
            "╔══════════════════════════════════════════════════╗\n"
            "║          *** LIVE TRADING MODE ***               ║\n"
            "║  This will place REAL orders with REAL money.    ║\n"
            "╚══════════════════════════════════════════════════╝\n"
            f"\nType exactly '{_LIVE_CONFIRM_PHRASE}' to continue: ",
            end="",
            flush=True,
        )
        response = input().strip()
        if response != _LIVE_CONFIRM_PHRASE:
            raise RuntimeError(
                "Live-trading confirmation rejected — AlpacaClient not created."
            )
        print("Live-trading confirmed.\n")

    def _health_check(self, max_attempts: int = 5) -> None:
        """
        Verify API connectivity.  Retries with exponential back-off up to
        ``max_attempts`` before raising ``ConnectionError``.
        """
        delay = _BACKOFF_BASE_S
        for attempt in range(1, max_attempts + 1):
            try:
                self._trading_client.get_account()
                logger.info("Alpaca health-check passed (attempt %d/%d)", attempt, max_attempts)
                return
            except Exception as exc:  # noqa: BLE001
                if attempt == max_attempts:
                    raise ConnectionError(
                        f"Alpaca health-check failed after {max_attempts} "
                        f"attempts.  Last error: {exc}"
                    ) from exc
                jitter   = random.uniform(-_BACKOFF_JITTER, _BACKOFF_JITTER)
                sleep_s  = min(delay * (1.0 + jitter), _BACKOFF_MAX_S)
                logger.warning(
                    "Health-check attempt %d/%d failed (%s). Retrying in %.1fs …",
                    attempt, max_attempts, exc, sleep_s,
                )
                time.sleep(sleep_s)
                delay = min(delay * 2.0, _BACKOFF_MAX_S)

    def reconnect(self, max_attempts: int = 10) -> None:
        """
        Re-initialise the REST client after a connection loss.

        Uses the same exponential back-off as the startup health-check.
        """
        logger.info("Attempting reconnect to Alpaca …")
        delay = _BACKOFF_BASE_S
        for attempt in range(1, max_attempts + 1):
            try:
                self._trading_client = TradingClient(
                    api_key=self._api_key,
                    secret_key=self._secret_key,
                    paper=self.paper,
                )
                self._trading_client.get_account()
                logger.info("Reconnected to Alpaca (attempt %d/%d)", attempt, max_attempts)
                return
            except Exception as exc:  # noqa: BLE001
                if attempt == max_attempts:
                    raise ConnectionError(
                        f"Reconnection failed after {max_attempts} attempts: {exc}"
                    ) from exc
                jitter  = random.uniform(-_BACKOFF_JITTER, _BACKOFF_JITTER)
                sleep_s = min(delay * (1.0 + jitter), _BACKOFF_MAX_S)
                logger.warning(
                    "Reconnect attempt %d/%d failed. Sleeping %.1fs …",
                    attempt, max_attempts, sleep_s,
                )
                time.sleep(sleep_s)
                delay = min(delay * 2.0, _BACKOFF_MAX_S)

    # -------------------------------------------------------------------------
    # Account
    # -------------------------------------------------------------------------

    def get_account(self) -> dict:
        """Return the current account snapshot (equity, cash, buying power)."""
        acct = self._trading_client.get_account()
        return {
            "id":               str(acct.id),
            "status":           acct.status.value if hasattr(acct.status, "value") else str(acct.status),
            "equity":           float(acct.equity),
            "cash":             float(acct.cash),
            "buying_power":     float(acct.buying_power),
            "portfolio_value":  float(acct.portfolio_value),
            "pattern_day_trader": bool(acct.pattern_day_trader),
            "trading_blocked":  bool(acct.trading_blocked),
            "transfers_blocked": bool(acct.transfers_blocked),
            "daytrade_count":   int(acct.daytrade_count),
        }

    def get_equity(self) -> float:
        """Return current portfolio equity as a float."""
        return float(self._trading_client.get_account().equity)

    def get_positions(self) -> list[dict]:
        """Return all open positions as a list of plain dicts."""
        positions = self._trading_client.get_all_positions()
        result = []
        for p in positions:
            result.append({
                "symbol":          p.symbol,
                "qty":             float(p.qty),
                "side":            p.side.value if hasattr(p.side, "value") else str(p.side),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price":   float(p.current_price),
                "market_value":    float(p.market_value),
                "cost_basis":      float(p.cost_basis),
                "unrealized_pl":   float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
                "change_today":    float(p.change_today) if p.change_today is not None else 0.0,
            })
        return result

    def get_order_history(
        self,
        status: str = "all",
        limit: int = 100,
        after: Optional[str] = None,
        until: Optional[str] = None,
    ) -> list[dict]:
        """
        Return historical orders.

        Parameters
        ----------
        status:
            ``"open"``, ``"closed"``, or ``"all"`` (default).
        limit:
            Maximum number of orders to return (default 100).
        after:
            ISO-8601 date string — return orders submitted after this date.
        until:
            ISO-8601 date string — return orders submitted before this date.
        """
        status_map = {
            "open":   QueryOrderStatus.OPEN,
            "closed": QueryOrderStatus.CLOSED,
            "all":    QueryOrderStatus.ALL,
        }
        request = GetOrdersRequest(
            status=status_map.get(status.lower(), QueryOrderStatus.ALL),
            limit=limit,
            after=after,
            until=until,
        )
        orders = self._trading_client.get_orders(filter=request)
        return [_order_to_dict(o) for o in orders]

    def get_available_margin(self) -> float:
        """Return available buying power (margin) in USD."""
        return float(self._trading_client.get_account().buying_power)

    def is_market_open(self) -> bool:
        """Return ``True`` if the primary US equity market is currently open."""
        return bool(self._trading_client.get_clock().is_open)

    def get_clock(self) -> dict:
        """
        Return the Alpaca market clock.

        Keys: ``is_open``, ``next_open``, ``next_close``, ``timestamp``.
        All datetime values are ISO-8601 strings.
        """
        clock = self._trading_client.get_clock()
        return {
            "is_open":    bool(clock.is_open),
            "next_open":  clock.next_open.isoformat()  if clock.next_open  else None,
            "next_close": clock.next_close.isoformat() if clock.next_close else None,
            "timestamp":  clock.timestamp.isoformat()  if clock.timestamp  else None,
        }

    # -------------------------------------------------------------------------
    # Asset info
    # -------------------------------------------------------------------------

    def get_asset(self, symbol: str) -> dict:
        """Return asset metadata (tradable, fractionable, exchange, etc.)."""
        asset = self._trading_client.get_asset(symbol)

        def _v(x):
            return x.value if hasattr(x, "value") else str(x)

        return {
            "symbol":       asset.symbol,
            "name":         asset.name,
            "exchange":     _v(asset.exchange),
            "asset_class":  _v(asset.asset_class),
            "status":       _v(asset.status),
            "tradable":     bool(asset.tradable),
            "fractionable": bool(asset.fractionable),
            "shortable":    bool(asset.shortable),
            "easy_to_borrow": bool(asset.easy_to_borrow),
        }

    def get_latest_quote(self, symbol: str) -> dict:
        """Return the most recent bid/ask quote for a symbol."""
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        result  = self._data_client.get_stock_latest_quote(request)
        quote   = result[symbol]
        return {
            "symbol":    symbol,
            "bid_price": float(quote.bid_price),
            "bid_size":  float(quote.bid_size),
            "ask_price": float(quote.ask_price),
            "ask_size":  float(quote.ask_size),
            "spread":    float(quote.ask_price) - float(quote.bid_price),
            "mid_price": (float(quote.bid_price) + float(quote.ask_price)) / 2.0,
            "timestamp": quote.timestamp.isoformat() if quote.timestamp else None,
        }

    def get_latest_bar(self, symbol: str) -> dict:
        """Return the most recent OHLCV bar for a symbol."""
        request = StockLatestBarRequest(symbol_or_symbols=symbol)
        result  = self._data_client.get_stock_latest_bar(request)
        bar     = result[symbol]
        return {
            "symbol":    symbol,
            "open":      float(bar.open),
            "high":      float(bar.high),
            "low":       float(bar.low),
            "close":     float(bar.close),
            "volume":    float(bar.volume),
            "vwap":      float(bar.vwap) if bar.vwap is not None else None,
            "timestamp": bar.timestamp.isoformat() if bar.timestamp else None,
        }

    def get_snapshot(self, symbol: str) -> dict:
        """
        Return a full market snapshot for a symbol.

        Includes the latest bar, latest quote, and today's daily bar.
        """
        request = StockSnapshotRequest(symbol_or_symbols=symbol)
        result  = self._data_client.get_stock_snapshot(request)
        snap    = result[symbol]
        out: dict = {"symbol": symbol}

        if snap.latest_bar:
            b = snap.latest_bar
            out["latest_bar"] = {
                "open": float(b.open), "high": float(b.high),
                "low":  float(b.low),  "close": float(b.close),
                "volume": float(b.volume),
                "vwap": float(b.vwap) if b.vwap is not None else None,
                "timestamp": b.timestamp.isoformat() if b.timestamp else None,
            }
        if snap.latest_quote:
            q = snap.latest_quote
            out["latest_quote"] = {
                "bid_price": float(q.bid_price),
                "ask_price": float(q.ask_price),
                "spread":    float(q.ask_price) - float(q.bid_price),
                "mid_price": (float(q.bid_price) + float(q.ask_price)) / 2.0,
                "timestamp": q.timestamp.isoformat() if q.timestamp else None,
            }
        if snap.daily_bar:
            d = snap.daily_bar
            out["daily_bar"] = {
                "open": float(d.open), "high": float(d.high),
                "low":  float(d.low),  "close": float(d.close),
                "volume": float(d.volume),
                "timestamp": d.timestamp.isoformat() if d.timestamp else None,
            }
        return out

    # -------------------------------------------------------------------------
    # Historical data
    # -------------------------------------------------------------------------

    def get_historical_bars(
        self,
        symbol: str,
        start: str,
        end: str,
        timeframe: str = "1Day",
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars for a single symbol.

        Parameters
        ----------
        symbol:
            Ticker string (e.g. ``"AAPL"``).
        start:
            ISO-8601 date/datetime string — inclusive lower bound.
        end:
            ISO-8601 date/datetime string — inclusive upper bound.
        timeframe:
            Bar width: ``"1Min"``, ``"5Min"``, ``"15Min"``, ``"1Hour"``,
            ``"1Day"`` (default), ``"1Week"``.
        limit:
            Optional cap on the number of bars returned.

        Returns
        -------
        :class:`pandas.DataFrame`
            Columns: ``open``, ``high``, ``low``, ``close``, ``volume``,
            ``vwap`` (where available).
            Index is a timezone-aware (UTC) ``DatetimeIndex``.
        """
        tf = _timeframe_from_str(timeframe)
        kwargs: dict = dict(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
        )
        if limit is not None:
            kwargs["limit"] = limit

        bar_set = self._data_client.get_stock_bars(StockBarsRequest(**kwargs))
        df = bar_set.df

        # alpaca-py returns a MultiIndex (symbol, timestamp) even for a single
        # symbol — drop the outer level so callers get a plain DatetimeIndex.
        if isinstance(df.index, pd.MultiIndex):
            level0 = df.index.get_level_values(0)
            if symbol in level0:
                df = df.loc[symbol]
            else:
                df = df.droplevel(0)

        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "timestamp"
        df.columns = [c.lower() for c in df.columns]
        return df

    def get_multi_bars(
        self,
        symbols: list[str],
        start: str,
        end: str,
        timeframe: str = "1Day",
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch historical OHLCV bars for multiple symbols in a single request.

        Returns a dict mapping ``symbol → DataFrame``.
        """
        tf      = _timeframe_from_str(timeframe)
        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=tf,
            start=start,
            end=end,
        )
        bar_set = self._data_client.get_stock_bars(request)
        df_all  = bar_set.df
        result: dict[str, pd.DataFrame] = {}

        if isinstance(df_all.index, pd.MultiIndex):
            for sym in symbols:
                lvl0 = df_all.index.get_level_values(0)
                if sym in lvl0:
                    df = df_all.loc[sym].copy()
                    df.index = pd.to_datetime(df.index, utc=True)
                    df.index.name = "timestamp"
                    df.columns = [c.lower() for c in df.columns]
                    result[sym] = df
        else:
            # Fallback: single symbol was returned without multi-index
            df_all = df_all.copy()
            df_all.index = pd.to_datetime(df_all.index, utc=True)
            df_all.index.name = "timestamp"
            df_all.columns = [c.lower() for c in df_all.columns]
            for sym in symbols:
                result[sym] = df_all.copy()

        return result

    # -------------------------------------------------------------------------
    # WebSocket streaming
    # -------------------------------------------------------------------------

    def _get_stream(self) -> "StockDataStream":
        """Return the shared ``StockDataStream``, creating it on first call."""
        if self._stream is None:
            self._stream = StockDataStream(
                api_key=self._api_key,
                secret_key=self._secret_key,
            )
        return self._stream

    def subscribe_bars(
        self,
        symbols: list[str],
        handler: Callable,
    ) -> None:
        """
        Subscribe to real-time bar updates for the given symbols.

        Parameters
        ----------
        symbols:
            List of ticker strings.
        handler:
            Async callable invoked with each incoming :class:`Bar` message.
            Signature: ``async def handler(bar) -> None``.

        Call :meth:`run_stream` to start the event loop (blocking).
        """
        self._get_stream().subscribe_bars(handler, *symbols)
        logger.debug("Subscribed to bars for: %s", symbols)

    def subscribe_quotes(
        self,
        symbols: list[str],
        handler: Callable,
    ) -> None:
        """
        Subscribe to real-time quote updates for the given symbols.

        Parameters
        ----------
        symbols:
            List of ticker strings.
        handler:
            Async callable invoked with each incoming :class:`Quote` message.
            Signature: ``async def handler(quote) -> None``.
        """
        self._get_stream().subscribe_quotes(handler, *symbols)
        logger.debug("Subscribed to quotes for: %s", symbols)

    def run_stream(self) -> None:
        """
        Start the WebSocket event loop (blocking).

        Intended to run in a dedicated background thread so the main thread
        remains free for order management and signal generation.
        """
        self._get_stream().run()

    def disconnect(self) -> None:
        """Close the WebSocket stream gracefully."""
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None
            logger.info("Alpaca WebSocket stream disconnected")
