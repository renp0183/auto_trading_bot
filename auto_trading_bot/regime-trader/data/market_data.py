"""
market_data.py — Real-time and historical market data fetching.

Responsibilities:
  - Build and maintain a per-symbol OHLCV history buffer (in-memory).
  - Back-fill history via the Alpaca REST API on initialisation.
  - Append live bars via WebSocket subscription as they arrive.
  - Subscribe to real-time quotes for spread checks.
  - Expose get_historical_bars(), get_latest_bar(), get_latest_quote(),
    get_snapshot() as standalone REST-backed methods.
  - Handle data gaps (weekends, holidays, trading halts) gracefully:
    missing bars are forward-filled so the buffer always has a complete index.
  - Provide aligned multi-symbol close matrices for the feature-engineering layer.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional, Callable

import numpy as np
import pandas as pd

from broker.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gap-handling constants
# ---------------------------------------------------------------------------
# Maximum consecutive missing bars to forward-fill before issuing a warning
_GAP_WARN_BARS = 5

# Columns that must be present in every bar DataFrame
_REQUIRED_COLS = ("open", "high", "low", "close", "volume")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fill_gaps(df: pd.DataFrame, max_warn: int = _GAP_WARN_BARS) -> pd.DataFrame:
    """
    Forward-fill a bar DataFrame to close gaps caused by weekends, holidays,
    or trading halts.

    Only the OHLCV columns are forward-filled; the timestamp index is kept
    as-is (no synthetic rows are injected).  Large gaps emit a warning.

    Parameters
    ----------
    df:
        Bar DataFrame with a DatetimeIndex.
    max_warn:
        Warn when a gap spans more than this many bars.

    Returns
    -------
    DataFrame with NaN values forward-filled.
    """
    if df.empty:
        return df

    # Detect gap runs before filling
    nan_counts = df["close"].isna().sum() if "close" in df.columns else 0
    if nan_counts > max_warn:
        logger.warning(
            "fill_gaps: %d missing 'close' values — large gap detected", nan_counts
        )

    return df.ffill()


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case all column names."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MarketDataFeed:
    """
    Unified market-data provider for historical and live bar/quote data.

    Maintains an in-memory buffer for each tracked symbol and merges live
    WebSocket bars as they arrive.

    History is loaded via Alpaca REST on :meth:`initialise`.
    Live updates stream via :meth:`start_stream` (runs in a daemon thread).

    Usage::

        feed = MarketDataFeed(client, symbols=["AAPL", "MSFT"], timeframe="1Day")
        feed.initialise(lookback_days=504)   # fill history buffer (~2 years)
        feed.start_stream()                  # begin live updates
        df = feed.get_bars("AAPL")           # returns buffered DataFrame
    """

    def __init__(
        self,
        client: AlpacaClient,
        symbols: list[str],
        timeframe: str = "1Day",
    ) -> None:
        """
        Parameters
        ----------
        client:
            Authenticated :class:`AlpacaClient` instance.
        symbols:
            List of tickers to track.
        timeframe:
            Bar width: ``"1Min"``, ``"5Min"``, ``"15Min"``, ``"1Hour"``,
            ``"1Day"`` (default), ``"1Week"``.
        """
        self._client    = client
        self._symbols   = list(symbols)
        self._timeframe = timeframe

        # symbol → DataFrame (OHLCV, DatetimeIndex UTC)
        self._buffers: dict[str, pd.DataFrame] = {s: pd.DataFrame() for s in self._symbols}

        # Latest quote per symbol (for spread checks)
        self._latest_quotes: dict[str, dict] = {}

        self._lock = threading.Lock()

        # Stream handles
        self._bar_handlers:   list[Callable] = []
        self._quote_handlers: list[Callable] = []
        self._stream_thread:  Optional[threading.Thread] = None

    # -------------------------------------------------------------------------
    # Initialisation
    # -------------------------------------------------------------------------

    def initialise(self, lookback_days: int = 504) -> None:
        """
        Back-fill the history buffer for all tracked symbols.

        Parameters
        ----------
        lookback_days:
            Number of calendar days of history to load.
            504 ≈ 2 trading years (default).  Must be ≥ the HMM
            ``min_train_bars`` setting (typically 252).
        """
        end   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        logger.info(
            "Initialising MarketDataFeed  symbols=%s  tf=%s  "
            "start=%s  end=%s",
            self._symbols, self._timeframe, start, end,
        )

        try:
            multi = self._client.get_multi_bars(
                symbols   = self._symbols,
                start     = start,
                end       = end,
                timeframe = self._timeframe,
            )
        except Exception as exc:
            logger.warning(
                "get_multi_bars failed (%s) — falling back to per-symbol fetch", exc
            )
            multi = {}
            for sym in self._symbols:
                try:
                    multi[sym] = self._client.get_historical_bars(
                        symbol    = sym,
                        start     = start,
                        end       = end,
                        timeframe = self._timeframe,
                    )
                except Exception as sym_exc:  # noqa: BLE001
                    logger.error("Failed to fetch history for %s: %s", sym, sym_exc)

        with self._lock:
            for sym in self._symbols:
                if sym in multi and not multi[sym].empty:
                    df = _normalise_columns(multi[sym])
                    df = _fill_gaps(df)
                    self._buffers[sym] = df
                    logger.info(
                        "Loaded %d bars for %s", len(df), sym
                    )
                else:
                    logger.warning("No data returned for %s", sym)

    # -------------------------------------------------------------------------
    # Streaming
    # -------------------------------------------------------------------------

    def start_stream(self) -> None:
        """
        Start the WebSocket bar (and optionally quote) stream for all tracked
        symbols in a daemon background thread.
        """
        if self._stream_thread and self._stream_thread.is_alive():
            logger.warning("MarketDataFeed stream is already running")
            return

        self._client.subscribe_bars(self._symbols, self._on_bar)
        logger.debug("Subscribed to bars for %s", self._symbols)

        def _run():
            logger.info("MarketDataFeed WebSocket stream started")
            self._client.run_stream()

        self._stream_thread = threading.Thread(
            target = _run,
            name   = "market-data-stream",
            daemon = True,
        )
        self._stream_thread.start()

    def stop_stream(self) -> None:
        """Gracefully stop the WebSocket stream."""
        self._client.disconnect()
        logger.info("MarketDataFeed stream stopped")

    def subscribe_bars(
        self,
        symbols: list[str],
        timeframe: str,
        callback: Callable,
    ) -> None:
        """
        Subscribe an external callback to real-time bar updates.

        Parameters
        ----------
        symbols:
            Tickers to subscribe.
        timeframe:
            Timeframe string (must match the feed's timeframe or be compatible).
        callback:
            Async callable invoked with each incoming bar.
            Signature: ``async def callback(bar) -> None``.
        """
        # Add any new symbols to our tracked list
        for sym in symbols:
            if sym not in self._symbols:
                self._symbols.append(sym)
                with self._lock:
                    if sym not in self._buffers:
                        self._buffers[sym] = pd.DataFrame()

        self._bar_handlers.append(callback)
        self._client.subscribe_bars(symbols, callback)
        logger.debug("External bar subscription added for %s", symbols)

    def subscribe_quotes(
        self,
        symbols: list[str],
        callback: Callable,
    ) -> None:
        """
        Subscribe an external callback to real-time quote updates.

        Also starts an internal quote buffer used by :meth:`get_latest_quote`.

        Parameters
        ----------
        symbols:
            Tickers to subscribe.
        callback:
            Async callable invoked with each incoming quote.
            Signature: ``async def callback(quote) -> None``.
        """
        async def _internal_and_external(quote):
            symbol = getattr(quote, "symbol", "")
            if symbol:
                self._latest_quotes[symbol] = {
                    "symbol":    symbol,
                    "bid_price": float(getattr(quote, "bid_price", 0) or 0),
                    "ask_price": float(getattr(quote, "ask_price", 0) or 0),
                    "bid_size":  float(getattr(quote, "bid_size",  0) or 0),
                    "ask_size":  float(getattr(quote, "ask_size",  0) or 0),
                    "timestamp": getattr(quote, "timestamp", None),
                }
            await callback(quote)

        self._quote_handlers.append(callback)
        self._client.subscribe_quotes(symbols, _internal_and_external)
        logger.debug("External quote subscription added for %s", symbols)

    async def _on_bar(self, bar) -> None:
        """
        Internal WebSocket callback — append an incoming bar to the symbol buffer.

        Gaps between the last buffered bar and the new bar are forward-filled.
        """
        symbol = getattr(bar, "symbol", None)
        if symbol is None or symbol not in self._buffers:
            return

        ts = getattr(bar, "timestamp", None)
        if ts is None:
            return

        row = pd.DataFrame(
            [{
                "open":   float(getattr(bar, "open",   0) or 0),
                "high":   float(getattr(bar, "high",   0) or 0),
                "low":    float(getattr(bar, "low",    0) or 0),
                "close":  float(getattr(bar, "close",  0) or 0),
                "volume": float(getattr(bar, "volume", 0) or 0),
                "vwap":   float(getattr(bar, "vwap",   0) or 0) or None,
            }],
            index=pd.DatetimeIndex([pd.Timestamp(ts, tz="UTC")], name="timestamp"),
        )

        with self._lock:
            existing = self._buffers.get(symbol, pd.DataFrame())
            if not existing.empty and row.index[0] <= existing.index[-1]:
                return  # duplicate or out-of-order bar — discard
            self._buffers[symbol] = pd.concat([existing, row])

        logger.debug(
            "Bar appended  %s  %s  close=%.4f",
            symbol, ts, row["close"].iloc[0],
        )

    # -------------------------------------------------------------------------
    # Standalone REST-backed methods (Phase 6 spec)
    # -------------------------------------------------------------------------

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars directly from the Alpaca REST API.

        This method bypasses the in-memory buffer and always hits the network.
        Use :meth:`get_bars` for the buffered version.

        Parameters
        ----------
        symbol:
            Ticker string.
        timeframe:
            Bar width (e.g. ``"1Day"``, ``"1Hour"``).
        start:
            ISO-8601 lower-bound date string.
        end:
            ISO-8601 upper-bound date string.

        Returns
        -------
        DataFrame with a UTC DatetimeIndex.  Gaps are forward-filled.
        """
        df = self._client.get_historical_bars(
            symbol    = symbol,
            start     = start,
            end       = end,
            timeframe = timeframe,
        )
        return _fill_gaps(_normalise_columns(df))

    def get_latest_bar(self, symbol: str) -> dict:
        """
        Return the most recent OHLCV bar for a symbol via the REST API.

        Falls back to the last row of the buffer when the API call fails.
        """
        try:
            return self._client.get_latest_bar(symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "get_latest_bar REST failed for %s (%s) — using buffer", symbol, exc
            )
            with self._lock:
                df = self._buffers.get(symbol, pd.DataFrame())
            if df.empty:
                return {}
            last = df.iloc[-1]
            return {
                "symbol": symbol,
                "open":   float(last.get("open",   0)),
                "high":   float(last.get("high",   0)),
                "low":    float(last.get("low",    0)),
                "close":  float(last.get("close",  0)),
                "volume": float(last.get("volume", 0)),
                "vwap":   float(last.get("vwap",   0)) if "vwap" in last.index else None,
                "timestamp": df.index[-1].isoformat(),
            }

    def get_latest_quote(self, symbol: str) -> dict:
        """
        Return the most recent bid/ask quote for a symbol.

        Uses the WebSocket-buffered quote if available (lower latency);
        falls back to a REST call.
        """
        cached = self._latest_quotes.get(symbol)
        if cached:
            return cached
        try:
            return self._client.get_latest_quote(symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_latest_quote failed for %s: %s", symbol, exc)
            return {}

    def get_snapshot(self, symbol: str) -> dict:
        """
        Return a full market snapshot (latest bar + quote + daily bar) for a
        symbol via the Alpaca REST API.
        """
        try:
            return self._client.get_snapshot(symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_snapshot failed for %s: %s", symbol, exc)
            return {"symbol": symbol}

    # -------------------------------------------------------------------------
    # Buffer access
    # -------------------------------------------------------------------------

    def get_bars(
        self,
        symbol: str,
        n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Return the buffered OHLCV history for a symbol.

        Parameters
        ----------
        symbol:
            Ticker string.
        n:
            When provided, return only the last ``n`` bars.

        Returns
        -------
        DataFrame with a UTC DatetimeIndex.  Returns an empty DataFrame
        when the symbol is not tracked.
        """
        with self._lock:
            df = self._buffers.get(symbol, pd.DataFrame())
        if df.empty:
            return df
        return df.iloc[-n:] if n is not None else df.copy()

    def get_aligned_closes(self) -> pd.DataFrame:
        """
        Return a DataFrame of close prices aligned to a common DatetimeIndex
        (one column per tracked symbol).

        Missing bars (weekends, holidays, symbol-specific halts) are forward-
        filled so the matrix is always rectangular and gap-free for the feature
        engineering layer.
        """
        frames: dict[str, pd.Series] = {}
        with self._lock:
            for sym, df in self._buffers.items():
                if not df.empty and "close" in df.columns:
                    frames[sym] = df["close"].rename(sym)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames.values(), axis=1)
        combined = combined.sort_index()
        # Forward-fill to handle any per-symbol gaps
        combined = combined.ffill()
        return combined

    def get_latest_price(self, symbol: str) -> float:
        """Return the most recent close price from the buffer, or ``0.0``."""
        with self._lock:
            df = self._buffers.get(symbol, pd.DataFrame())
        if df.empty or "close" not in df.columns:
            return 0.0
        return float(df["close"].iloc[-1])

    def symbols(self) -> list[str]:
        """Return the list of tracked symbols."""
        return list(self._symbols)

    def is_ready(self, min_bars: int) -> bool:
        """
        Return ``True`` when ALL tracked symbols have at least ``min_bars``
        in their history buffer.

        Parameters
        ----------
        min_bars:
            Minimum bar count required (e.g. 252 for a 1-year daily HMM
            training window).
        """
        with self._lock:
            for sym in self._symbols:
                df = self._buffers.get(sym, pd.DataFrame())
                if len(df) < min_bars:
                    return False
        return True

    def bars_available(self) -> dict[str, int]:
        """Return ``symbol → bar_count`` for all tracked symbols."""
        with self._lock:
            return {sym: len(df) for sym, df in self._buffers.items()}

    # -------------------------------------------------------------------------
    # Intraday (5m / 15m) REST-polled buffer
    # -------------------------------------------------------------------------
    # The intraday layer uses REST polling rather than a second WebSocket
    # stream.  Every INTRADAY_POLL_INTERVAL seconds the main loop calls
    # poll_intraday(), which fetches the last N bars from the REST API and
    # merges them into _intraday_buffers.  No second stream thread needed.
    # -------------------------------------------------------------------------

    def initialise_intraday(
        self,
        timeframe: str = "5Min",
        lookback_minutes: int = 500,
    ) -> None:
        """
        Back-fill the intraday buffer via REST for all tracked symbols.

        Parameters
        ----------
        timeframe:
            Bar width string — ``"5Min"`` (default) or ``"15Min"``.
        lookback_minutes:
            How many minutes of intraday history to load.
            500 min ≈ 5 full trading days (enough for indicator warm-up).
        """
        self._intraday_timeframe = timeframe
        self._intraday_buffers: dict[str, pd.DataFrame] = {
            s: pd.DataFrame() for s in self._symbols
        }

        from datetime import datetime, timedelta, timezone as _tz
        end   = datetime.now(_tz.utc)
        start = end - timedelta(minutes=lookback_minutes)
        start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str   = end.strftime("%Y-%m-%dT%H:%M:%SZ")

        logger.info(
            "Initialising intraday feed  tf=%s  lookback=%dm  "
            "start=%s  end=%s",
            timeframe, lookback_minutes, start_str[:16], end_str[:16],
        )

        for sym in self._symbols:
            try:
                df = self._client.get_historical_bars(
                    symbol=sym, start=start_str, end=end_str, timeframe=timeframe
                )
                if not df.empty:
                    df = _normalise_columns(df)
                    self._intraday_buffers[sym] = df
                    logger.info("Intraday loaded %d bars for %s", len(df), sym)
                else:
                    logger.warning("Intraday: no data returned for %s", sym)
            except Exception as exc:
                logger.error("Intraday init failed for %s: %s", sym, exc)

    def poll_intraday(self) -> dict[str, bool]:
        """
        Fetch the latest intraday bars from REST and append any new ones to
        the buffer.

        Should be called periodically (e.g. every 30 s) from the main loop.

        Returns
        -------
        dict[symbol, bool]
            True for each symbol that received at least one new bar.
        """
        if not hasattr(self, "_intraday_buffers"):
            return {}

        from datetime import datetime, timedelta, timezone as _tz
        # Fetch slightly more than one bar interval to catch any missed bars
        lookback_minutes = 30
        end   = datetime.now(_tz.utc)
        start = end - timedelta(minutes=lookback_minutes)
        start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str   = end.strftime("%Y-%m-%dT%H:%M:%SZ")

        new_bars: dict[str, bool] = {}
        for sym in self._symbols:
            try:
                fresh = self._client.get_historical_bars(
                    symbol=sym, start=start_str, end=end_str,
                    timeframe=self._intraday_timeframe,
                )
                if fresh.empty:
                    new_bars[sym] = False
                    continue

                fresh = _normalise_columns(fresh)
                existing = self._intraday_buffers.get(sym, pd.DataFrame())

                if existing.empty:
                    self._intraday_buffers[sym] = fresh
                    new_bars[sym] = True
                    continue

                last_ts = existing.index[-1]
                added   = fresh[fresh.index > last_ts]
                if not added.empty:
                    self._intraday_buffers[sym] = pd.concat([existing, added])
                    logger.debug("Intraday %s: +%d bars (latest %s)",
                                 sym, len(added), added.index[-1])
                    new_bars[sym] = True
                else:
                    new_bars[sym] = False

            except Exception as exc:
                logger.warning("Intraday poll failed for %s: %s", sym, exc)
                new_bars[sym] = False

        return new_bars

    def get_intraday_bars(
        self,
        symbol: str,
        n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Return buffered intraday OHLCV bars for ``symbol``.

        Parameters
        ----------
        symbol : str
        n : int, optional
            Return only the last ``n`` bars.

        Returns
        -------
        DataFrame (empty when symbol not in buffer or intraday not initialised).
        """
        if not hasattr(self, "_intraday_buffers"):
            return pd.DataFrame()
        buf = self._intraday_buffers.get(symbol, pd.DataFrame())
        if buf.empty:
            return buf
        return buf.iloc[-n:].copy() if n is not None else buf.copy()

    def intraday_bars_available(self) -> dict[str, int]:
        """Return ``symbol → intraday_bar_count`` for all tracked symbols."""
        if not hasattr(self, "_intraday_buffers"):
            return {}
        return {sym: len(df) for sym, df in self._intraday_buffers.items()}
