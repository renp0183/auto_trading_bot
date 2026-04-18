"""
order_executor.py — Order placement, modification, and cancellation.

Responsibilities:
  - Translate TradeSignal objects into Alpaca order requests.
  - submit_order(): LIMIT by default at ±0.1% of current price; cancels after
    30 s if unfilled; optionally retries at market price.
  - submit_bracket_order(): entry + OCO stop-loss + take-profit.
  - modify_stop(): only allowed to tighten (raise) a stop — never widen.
  - cancel_order(), close_position(), close_all_positions().
  - Every signal is tagged with a unique trade_id (UUID4) that flows through
    signal → risk_decision → order → fill for end-to-end traceability.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from broker.alpaca_client import AlpacaClient
from core.signal_generator import TradeSignal, SignalType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# alpaca-py lazy import
# ---------------------------------------------------------------------------
try:
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        ReplaceOrderRequest,
        TakeProfitRequest,
        StopLossRequest,
        ClosePositionRequest,
    )
    from alpaca.trading.enums import (
        OrderSide    as _AlpacaSide,
        OrderType    as _AlpacaType,
        TimeInForce  as _AlpacaTIF,
        OrderClass   as _AlpacaClass,
    )
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LIMIT_OFFSET_PCT      = 0.001   # 0.1% offset from current price for limit orders
_LIMIT_CANCEL_TIMEOUT  = 30      # seconds before an unfilled limit order is cancelled
_MARKET_RETRY_DELAY    = 1.0     # seconds to wait before submitting the market retry


# ---------------------------------------------------------------------------
# Local enumerations (mirrored from Alpaca for type-safe internal use)
# ---------------------------------------------------------------------------

class OrderType(str, Enum):
    MARKET     = "market"
    LIMIT      = "limit"
    STOP       = "stop"
    STOP_LIMIT = "stop_limit"
    BRACKET    = "bracket"


class OrderSide(str, Enum):
    BUY  = "buy"
    SELL = "sell"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    OPG = "opg"
    CLS = "cls"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OrderResult:
    """Result of an order submission attempt."""

    trade_id:          str             # UUID linking signal → order → fill
    order_id:          str
    client_order_id:   str
    symbol:            str
    side:              OrderSide
    order_type:        OrderType
    qty:               int
    limit_price:       Optional[float]
    stop_price:        Optional[float]
    status:            str             # Alpaca order status string
    filled_qty:        int   = 0
    filled_avg_price:  Optional[float] = None
    legs:              list  = field(default_factory=list)  # bracket legs
    error:             Optional[str]  = None                # set when submission failed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_trade_id() -> str:
    """Return a new UUID4 hex string to tag this trade end-to-end."""
    return uuid.uuid4().hex


def _alpaca_side(side: OrderSide) -> "_AlpacaSide":
    return _AlpacaSide.BUY if side == OrderSide.BUY else _AlpacaSide.SELL


def _alpaca_tif(tif: TimeInForce) -> "_AlpacaTIF":
    return _AlpacaTIF(tif.value)


def _order_dict_to_result(raw: dict, trade_id: str, order_type: OrderType) -> OrderResult:
    """Convert the dict from ``AlpacaClient._order_to_dict`` to an ``OrderResult``."""
    side = OrderSide.BUY if (raw.get("side") or "").lower() == "buy" else OrderSide.SELL
    return OrderResult(
        trade_id         = trade_id,
        order_id         = raw.get("id", ""),
        client_order_id  = raw.get("client_order_id", ""),
        symbol           = raw.get("symbol", ""),
        side             = side,
        order_type       = order_type,
        qty              = int(raw.get("qty") or 0),
        limit_price      = raw.get("limit_price"),
        stop_price       = raw.get("stop_price"),
        status           = raw.get("status", "unknown"),
        filled_qty       = int(raw.get("filled_qty") or 0),
        filled_avg_price = raw.get("filled_avg_price"),
        legs             = raw.get("legs", []),
    )


def _alpaca_order_to_result(order, trade_id: str, order_type: OrderType) -> OrderResult:
    """Convert an alpaca-py Order object directly to an ``OrderResult``."""
    def _v(x):
        return x.value if hasattr(x, "value") else str(x) if x is not None else None

    raw_side = _v(order.side) or "buy"
    side     = OrderSide.BUY if raw_side.lower() == "buy" else OrderSide.SELL

    legs: list = []
    for leg in (order.legs or []):
        legs.append({
            "id":     str(leg.id),
            "side":   _v(leg.side),
            "type":   _v(leg.type),
            "status": _v(leg.status),
            "limit_price": float(leg.limit_price) if leg.limit_price is not None else None,
            "stop_price":  float(leg.stop_price)  if leg.stop_price  is not None else None,
        })

    return OrderResult(
        trade_id         = trade_id,
        order_id         = str(order.id),
        client_order_id  = str(order.client_order_id),
        symbol           = order.symbol,
        side             = side,
        order_type       = order_type,
        qty              = int(order.qty) if order.qty is not None else 0,
        limit_price      = float(order.limit_price)      if order.limit_price      is not None else None,
        stop_price       = float(order.stop_price)       if order.stop_price       is not None else None,
        status           = _v(order.status) or "unknown",
        filled_qty       = int(order.filled_qty)         if order.filled_qty       is not None else 0,
        filled_avg_price = float(order.filled_avg_price) if order.filled_avg_price is not None else None,
        legs             = legs,
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class OrderExecutor:
    """
    Converts TradeSignal objects into live Alpaca orders and manages their
    full lifecycle (submission → fill / cancellation / retry).

    Design rules
    ------------
    * All entry orders are submitted as **LIMIT** orders by default, priced at
      ±0.1 % of the current market price.  A background timer cancels the order
      after 30 seconds if it remains unfilled; the caller can optionally request
      a market-order retry at that point.
    * Bracket orders use Alpaca's native OCO mechanism: a single parent order
      carries both a take-profit leg and a stop-loss leg.
    * Stop modification is **one-directional**: for long positions only a
      tightening (upward) move is accepted.  Widening raises ``ValueError``.
    * Every order carries a ``trade_id`` (UUID4) that links the originating
      signal, the risk decision, the order, and its eventual fill.

    Usage::

        executor = OrderExecutor(alpaca_client)
        result   = executor.submit_order(signal, retry_at_market=True)
        executor.cancel_order(result.order_id)
    """

    def __init__(self, client: AlpacaClient) -> None:
        """
        Parameters
        ----------
        client:
            Authenticated :class:`AlpacaClient` instance.
        """
        self._client = client
        # Maps order_id → (trade_id, cancel_timer) for active limit orders
        self._pending_limits: dict[str, tuple[str, threading.Timer]] = {}
        self._lock = threading.Lock()

    # -------------------------------------------------------------------------
    # Primary signal-to-order interface
    # -------------------------------------------------------------------------

    def submit_order(
        self,
        signal: TradeSignal,
        retry_at_market: bool = False,
    ) -> OrderResult:
        """
        Submit a **LIMIT** order derived from a TradeSignal.

        The limit price is set at ±0.1 % from the signal's ``entry_price``
        (+ for BUY, − for SELL) to improve fill probability while controlling
        slippage.

        If the order remains unfilled after 30 seconds, it is cancelled
        automatically.  When ``retry_at_market=True``, a market order is
        submitted immediately after cancellation.

        Parameters
        ----------
        signal:
            :class:`TradeSignal` produced by the signal generator.
        retry_at_market:
            When ``True``, submit a market order if the limit times out.

        Returns
        -------
        :class:`OrderResult`
            The result for the initially submitted limit order.  If a market
            retry fires, it is logged but the original ``OrderResult`` is
            returned so callers can track via ``trade_id``.
        """
        trade_id = _generate_trade_id()
        # Propagate trade_id into the signal for downstream tracing
        signal.metadata["trade_id"] = trade_id

        side = OrderSide.BUY if signal.signal_type in (SignalType.BUY,) else OrderSide.SELL

        offset  = _LIMIT_OFFSET_PCT * signal.entry_price
        limit_p = round(
            signal.entry_price + offset if side == OrderSide.BUY
            else signal.entry_price - offset,
            2,
        )

        result = self._submit_limit_raw(
            trade_id    = trade_id,
            symbol      = signal.symbol,
            side        = side,
            qty         = signal.shares,
            limit_price = limit_p,
            tif         = TimeInForce.DAY,
        )

        if result.error:
            return result

        self._schedule_cancel(
            order_id         = result.order_id,
            trade_id         = trade_id,
            signal           = signal,
            retry_at_market  = retry_at_market,
        )
        return result

    def submit_bracket_order(
        self,
        signal: TradeSignal,
    ) -> OrderResult:
        """
        Submit an entry + OCO stop-loss + take-profit bracket order.

        Uses the signal's ``entry_price``, ``stop_price``, and
        ``target_price`` to construct the Alpaca bracket.

        Parameters
        ----------
        signal:
            :class:`TradeSignal` with all three price levels populated.

        Returns
        -------
        :class:`OrderResult` with bracket legs attached.

        Raises
        ------
        ``ValueError``
            When ``signal.target_price`` is ``None`` or ``signal.stop_price``
            is not positive.
        """
        if signal.target_price is None:
            raise ValueError(
                f"submit_bracket_order requires a target_price on the signal "
                f"(symbol={signal.symbol})"
            )
        if signal.stop_price <= 0:
            raise ValueError(
                f"Invalid stop_price={signal.stop_price} for {signal.symbol}"
            )

        trade_id = _generate_trade_id()
        signal.metadata["trade_id"] = trade_id

        offset  = _LIMIT_OFFSET_PCT * signal.entry_price
        entry_p = round(signal.entry_price + offset, 2)  # BUY only (long-only system)

        try:
            order_request = LimitOrderRequest(
                symbol           = signal.symbol,
                qty              = signal.shares,
                side             = _AlpacaSide.BUY,
                type             = _AlpacaType.LIMIT,
                time_in_force    = _AlpacaTIF.GTC,
                limit_price      = entry_p,
                order_class      = _AlpacaClass.BRACKET,
                take_profit      = TakeProfitRequest(limit_price=round(signal.target_price, 2)),
                stop_loss        = StopLossRequest(stop_price=round(signal.stop_price, 2)),
            )
            order = self._client._trading_client.submit_order(order_request)
            result = _alpaca_order_to_result(order, trade_id, OrderType.BRACKET)
            logger.info(
                "Bracket order submitted  symbol=%-6s  qty=%d  "
                "entry=%.2f  stop=%.2f  target=%.2f  trade_id=%s",
                signal.symbol, signal.shares,
                entry_p, signal.stop_price, signal.target_price,
                trade_id,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Bracket order FAILED  symbol=%s  error=%s  trade_id=%s",
                signal.symbol, exc, trade_id,
            )
            return OrderResult(
                trade_id    = trade_id,
                order_id    = "",
                client_order_id = "",
                symbol      = signal.symbol,
                side        = OrderSide.BUY,
                order_type  = OrderType.BRACKET,
                qty         = signal.shares,
                limit_price = entry_p,
                stop_price  = signal.stop_price,
                status      = "error",
                error       = str(exc),
            )

    # -------------------------------------------------------------------------
    # Explicit order-type helpers (used directly or by submit_order)
    # -------------------------------------------------------------------------

    def submit_market_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        tif: TimeInForce = TimeInForce.DAY,
        trade_id: Optional[str] = None,
    ) -> OrderResult:
        """Place a plain market order."""
        trade_id = trade_id or _generate_trade_id()
        try:
            order_request = MarketOrderRequest(
                symbol        = symbol,
                qty           = qty,
                side          = _alpaca_side(side),
                time_in_force = _alpaca_tif(tif),
            )
            order = self._client._trading_client.submit_order(order_request)
            result = _alpaca_order_to_result(order, trade_id, OrderType.MARKET)
            logger.info(
                "Market order submitted  symbol=%-6s  side=%s  qty=%d  trade_id=%s",
                symbol, side.value, qty, trade_id,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Market order FAILED  symbol=%s  side=%s  error=%s",
                symbol, side.value, exc,
            )
            return OrderResult(
                trade_id    = trade_id,
                order_id    = "",
                client_order_id = "",
                symbol      = symbol,
                side        = side,
                order_type  = OrderType.MARKET,
                qty         = qty,
                limit_price = None,
                stop_price  = None,
                status      = "error",
                error       = str(exc),
            )

    def submit_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        limit_price: float,
        tif: TimeInForce = TimeInForce.GTC,
        trade_id: Optional[str] = None,
    ) -> OrderResult:
        """Place a limit order."""
        trade_id = trade_id or _generate_trade_id()
        return self._submit_limit_raw(
            trade_id    = trade_id,
            symbol      = symbol,
            side        = side,
            qty         = qty,
            limit_price = limit_price,
            tif         = tif,
        )

    def _submit_limit_raw(
        self,
        trade_id:    str,
        symbol:      str,
        side:        OrderSide,
        qty:         int,
        limit_price: float,
        tif:         TimeInForce,
    ) -> OrderResult:
        """Internal limit-order helper used by both submit_order and submit_limit_order."""
        try:
            order_request = LimitOrderRequest(
                symbol        = symbol,
                qty           = qty,
                side          = _alpaca_side(side),
                type          = _AlpacaType.LIMIT,
                time_in_force = _alpaca_tif(tif),
                limit_price   = round(limit_price, 2),
            )
            order = self._client._trading_client.submit_order(order_request)
            result = _alpaca_order_to_result(order, trade_id, OrderType.LIMIT)
            logger.info(
                "Limit order submitted  symbol=%-6s  side=%s  qty=%d  "
                "limit=%.2f  trade_id=%s",
                symbol, side.value, qty, limit_price, trade_id,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Limit order FAILED  symbol=%s  side=%s  error=%s  trade_id=%s",
                symbol, side.value, exc, trade_id,
            )
            return OrderResult(
                trade_id    = trade_id,
                order_id    = "",
                client_order_id = "",
                symbol      = symbol,
                side        = side,
                order_type  = OrderType.LIMIT,
                qty         = qty,
                limit_price = limit_price,
                stop_price  = None,
                status      = "error",
                error       = str(exc),
            )

    # -------------------------------------------------------------------------
    # Stop modification
    # -------------------------------------------------------------------------

    def modify_stop(
        self,
        symbol: str,
        new_stop: float,
        current_stop: float,
    ) -> bool:
        """
        Adjust the stop-loss price for an open position.

        **Only tightening is permitted.**  For long positions (the only
        direction this system trades) that means ``new_stop > current_stop``.
        Attempting to widen the stop raises ``ValueError``.

        The stop is modified by replacing the open stop-loss order via the
        Alpaca ``replace_order`` endpoint.

        Parameters
        ----------
        symbol:
            Ticker of the position whose stop should be moved.
        new_stop:
            The new stop price.  Must be higher than ``current_stop``.
        current_stop:
            The currently active stop price (used for the tighten check).

        Returns
        -------
        ``True`` if the replacement was accepted, ``False`` on API error.

        Raises
        ------
        ``ValueError``
            When ``new_stop <= current_stop`` (would widen the stop).
        """
        if new_stop <= current_stop:
            raise ValueError(
                f"modify_stop: new_stop ({new_stop:.4f}) must be above "
                f"current_stop ({current_stop:.4f}) for long positions. "
                "Widening stops is not permitted."
            )

        # Locate the open stop-loss order for this symbol
        open_orders = self.get_open_orders(symbol=symbol)
        stop_orders = [
            o for o in open_orders
            if o.order_type in (OrderType.STOP, OrderType.STOP_LIMIT)
            and o.side == OrderSide.SELL
        ]

        if not stop_orders:
            logger.warning(
                "modify_stop: no open stop order found for %s "
                "(may be part of a bracket — modify via broker UI)",
                symbol,
            )
            return False

        for stop_order in stop_orders:
            try:
                self._client._trading_client.replace_order_by_id(
                    order_id    = stop_order.order_id,
                    order_data  = ReplaceOrderRequest(stop_price=round(new_stop, 2)),
                )
                logger.info(
                    "Stop tightened  symbol=%-6s  %.4f → %.4f",
                    symbol, current_stop, new_stop,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "modify_stop FAILED  symbol=%s  error=%s", symbol, exc
                )
                return False
        return True

    # -------------------------------------------------------------------------
    # Order management
    # -------------------------------------------------------------------------

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a specific order by ID.

        Returns ``True`` if the cancellation request was accepted.
        """
        # Disarm the pending-cancel timer if it exists
        with self._lock:
            if order_id in self._pending_limits:
                _, timer = self._pending_limits.pop(order_id)
                timer.cancel()

        try:
            self._client._trading_client.cancel_order_by_id(order_id)
            logger.info("Order cancelled  order_id=%s", order_id)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("cancel_order failed  order_id=%s  error=%s", order_id, exc)
            return False

    def cancel_all_orders(self) -> list[str]:
        """
        Cancel all open orders.

        Returns a list of cancelled order IDs.
        """
        try:
            cancel_responses = self._client._trading_client.cancel_orders()
            cancelled = [str(r.id) for r in (cancel_responses or [])]
            # Disarm all pending timers
            with self._lock:
                for oid in list(self._pending_limits.keys()):
                    _, timer = self._pending_limits.pop(oid)
                    timer.cancel()
            logger.info("cancel_all_orders: cancelled %d order(s)", len(cancelled))
            return cancelled
        except Exception as exc:  # noqa: BLE001
            logger.error("cancel_all_orders failed: %s", exc)
            return []

    def close_position(
        self,
        symbol: str,
        qty: Optional[int] = None,
        percentage: Optional[float] = None,
    ) -> Optional[OrderResult]:
        """
        Close an open position (fully or partially).

        Parameters
        ----------
        symbol:
            Ticker of the position to close.
        qty:
            Number of shares to close.  Mutually exclusive with ``percentage``.
        percentage:
            Fraction of the position to close (0.0–1.0).
            Mutually exclusive with ``qty``.

        Returns
        -------
        :class:`OrderResult` on success, or ``None`` if the API call failed.
        """
        trade_id = _generate_trade_id()
        try:
            kwargs: dict = {}
            if qty is not None:
                kwargs["close_options"] = ClosePositionRequest(qty=str(qty))
            elif percentage is not None:
                kwargs["close_options"] = ClosePositionRequest(
                    percentage=str(round(percentage * 100, 4))
                )
            order = self._client._trading_client.close_position(symbol, **kwargs)
            result = _alpaca_order_to_result(order, trade_id, OrderType.MARKET)
            logger.info(
                "close_position  symbol=%-6s  qty=%s  pct=%s  trade_id=%s",
                symbol, qty, percentage, trade_id,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            logger.error("close_position FAILED  symbol=%s  error=%s", symbol, exc)
            return None

    def close_all_positions(self, cancel_orders_first: bool = True) -> list[OrderResult]:
        """
        Close every open position via Alpaca's bulk-close endpoint.

        Parameters
        ----------
        cancel_orders_first:
            Cancel all open orders before closing positions (default ``True``).
            Prevents bracket legs from reopening closed positions.

        Returns
        -------
        List of :class:`OrderResult` — one per position closed.
        """
        if cancel_orders_first:
            self.cancel_all_orders()

        try:
            close_responses = self._client._trading_client.close_all_positions(
                cancel_orders=cancel_orders_first
            )
            results = []
            for resp in (close_responses or []):
                # close_all_positions returns CloseAllPositionsResponse objects
                # Each has a .body that is an Order
                try:
                    order = resp.body if hasattr(resp, "body") else resp
                    results.append(
                        _alpaca_order_to_result(order, _generate_trade_id(), OrderType.MARKET)
                    )
                except Exception:  # noqa: BLE001
                    pass
            logger.info("close_all_positions: closed %d position(s)", len(results))
            return results
        except Exception as exc:  # noqa: BLE001
            logger.error("close_all_positions failed: %s", exc)
            return []

    def replace_order(
        self,
        order_id: str,
        qty: Optional[int] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Optional[OrderResult]:
        """
        Amend an existing unfilled or partially filled order.

        Returns ``None`` on failure.
        """
        try:
            req_kwargs: dict = {}
            if qty is not None:
                req_kwargs["qty"] = qty
            if limit_price is not None:
                req_kwargs["limit_price"] = round(limit_price, 2)
            if stop_price is not None:
                req_kwargs["stop_price"] = round(stop_price, 2)

            order = self._client._trading_client.replace_order_by_id(
                order_id   = order_id,
                order_data = ReplaceOrderRequest(**req_kwargs),
            )
            # The original trade_id is not available here; generate a suffix ID
            return _alpaca_order_to_result(order, _generate_trade_id(), OrderType.LIMIT)
        except Exception as exc:  # noqa: BLE001
            logger.error("replace_order FAILED  order_id=%s  error=%s", order_id, exc)
            return None

    def get_order(self, order_id: str) -> Optional[OrderResult]:
        """Fetch the current state of an order by ID. Returns ``None`` on error."""
        try:
            order = self._client._trading_client.get_order_by_id(order_id)
            return _alpaca_order_to_result(order, _generate_trade_id(), OrderType.LIMIT)
        except Exception as exc:  # noqa: BLE001
            logger.error("get_order failed  order_id=%s  error=%s", order_id, exc)
            return None

    def get_open_orders(self, symbol: Optional[str] = None) -> list[OrderResult]:
        """
        Return all open (working / pending) orders.

        Parameters
        ----------
        symbol:
            When provided, filter to orders for this ticker only.
        """
        try:
            from alpaca.trading.requests import GetOrdersRequest as _Req
            from alpaca.trading.enums import QueryOrderStatus as _QOS
            req = _Req(status=_QOS.OPEN, symbols=[symbol] if symbol else None)
            orders = self._client._trading_client.get_orders(filter=req)
            return [
                _alpaca_order_to_result(o, _generate_trade_id(), OrderType.LIMIT)
                for o in orders
            ]
        except Exception as exc:  # noqa: BLE001
            logger.error("get_open_orders failed: %s", exc)
            return []

    # -------------------------------------------------------------------------
    # Convenience alias (matches signal_generator.py stub naming)
    # -------------------------------------------------------------------------

    def submit_signal(
        self,
        signal: TradeSignal,
        order_type: "OrderType" = OrderType.LIMIT,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> OrderResult:
        """
        Alias for :meth:`submit_order` that accepts an explicit ``order_type``.

        Used by the backtest harness and existing stubs.
        """
        if order_type == OrderType.MARKET:
            trade_id = _generate_trade_id()
            signal.metadata["trade_id"] = trade_id
            side = OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL
            return self.submit_market_order(
                symbol   = signal.symbol,
                side     = side,
                qty      = signal.shares,
                tif      = time_in_force,
                trade_id = trade_id,
            )
        if order_type == OrderType.BRACKET:
            return self.submit_bracket_order(signal)
        return self.submit_order(signal)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _schedule_cancel(
        self,
        order_id:        str,
        trade_id:        str,
        signal:          TradeSignal,
        retry_at_market: bool,
    ) -> None:
        """
        Schedule an automatic cancel after ``_LIMIT_CANCEL_TIMEOUT`` seconds.

        When ``retry_at_market=True``, a market order is submitted after the
        limit order has been cancelled.
        """
        def _cancel_and_maybe_retry():
            logger.info(
                "Limit order timeout (%ds)  order_id=%s  symbol=%s — cancelling …",
                _LIMIT_CANCEL_TIMEOUT, order_id, signal.symbol,
            )
            cancelled = self.cancel_order(order_id)

            if cancelled and retry_at_market:
                time.sleep(_MARKET_RETRY_DELAY)
                side = (
                    OrderSide.BUY
                    if signal.signal_type == SignalType.BUY
                    else OrderSide.SELL
                )
                logger.info(
                    "Retrying at market  symbol=%s  qty=%d  trade_id=%s",
                    signal.symbol, signal.shares, trade_id,
                )
                self.submit_market_order(
                    symbol   = signal.symbol,
                    side     = side,
                    qty      = signal.shares,
                    trade_id = trade_id,      # reuse trade_id for traceability
                )

        timer = threading.Timer(_LIMIT_CANCEL_TIMEOUT, _cancel_and_maybe_retry)
        timer.daemon = True
        timer.start()

        with self._lock:
            self._pending_limits[order_id] = (trade_id, timer)
