"""
position_tracker.py — Real-time position tracking with WebSocket fill notifications.

Responsibilities:
  - Subscribe to Alpaca's TradingStream for instant fill / partial-fill
    notifications (no polling delay).
  - On every fill: update PortfolioState and re-evaluate CircuitBreaker.
  - Per-position tracking: entry time/price, current price, unrealised P&L,
    stop level, holding period (bars), regime at entry vs. current.
  - Sync with Alpaca REST on startup to reconcile any pre-existing positions.
  - Maintain a closed-trade ledger and drift map for rebalancing decisions.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Callable

import pandas as pd

from broker.alpaca_client import AlpacaClient
from core.risk_manager import PortfolioState, CircuitBreaker, Position as RiskPosition

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# alpaca-py lazy import
# ---------------------------------------------------------------------------
try:
    from alpaca.trading.stream import TradingStream
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Extended position dataclass (richer than RiskManager's Position)
# ---------------------------------------------------------------------------

@dataclass
class TrackedPosition:
    """
    Full runtime snapshot of a single open position.

    Extends the risk-manager's ``Position`` with live-trading metadata needed
    for performance attribution and stop management.
    """
    # Core identity
    symbol:            str
    trade_id:          str               # UUID linking signal → order → fill
    side:              str               # "long" only in this system

    # Size & price
    qty:               int               # Number of shares held
    avg_entry_price:   float             # Volume-weighted average fill price
    current_price:     float             # Most recent market price
    stop_level:        float             # Active stop-loss price

    # Cost & value
    cost_basis:        float             # avg_entry_price × qty
    market_value:      float             # current_price × qty
    unrealised_pnl:    float             # market_value − cost_basis
    unrealised_pnl_pct: float            # unrealised_pnl / cost_basis

    # Timing
    entry_time:        datetime          # UTC timestamp of the first fill
    bars_held:         int = 0           # Number of bars since entry (updated each bar)

    # Regime context
    regime_at_entry:   str = "UNKNOWN"   # HMM label when the position was opened
    regime_current:    str = "UNKNOWN"   # Most recent HMM label

    # Exit management (populated after entry fill)
    r_value:           float = 0.0       # Per-share risk in $ at entry (entry − stop); used for BE trigger
    tp1_level:         float = 0.0       # First partial-exit target price (0.0 = none)
    tp1_hit:           bool  = False     # True once TP1 has been triggered; prevents duplicate close
    tp2_level:         float = 0.0       # Second partial-exit target price (0.0 = none)
    tp2_hit:           bool  = False     # True once TP2 has been triggered
    tp3_level:         float = 0.0       # Third partial-exit target price (0.0 = none / not activated)
    tp3_hit:           bool  = False     # True once TP3 has been triggered (tightens trailing)
    vol_tier:          str   = "UNKNOWN" # Strategy tier at entry: "LowVol", "MidVol", "HighVol"

    # Fill accumulation (for partial fills)
    fills:             list[dict] = field(default_factory=list)

    def update_price(self, new_price: float) -> None:
        """Refresh market-price-derived fields."""
        self.current_price      = new_price
        self.market_value       = new_price * self.qty
        self.unrealised_pnl     = self.market_value - self.cost_basis
        self.unrealised_pnl_pct = (
            self.unrealised_pnl / self.cost_basis if self.cost_basis else 0.0
        )

    def to_risk_position(self) -> RiskPosition:
        """Return a :class:`core.risk_manager.Position` view for the risk layer."""
        return RiskPosition(
            symbol      = self.symbol,
            shares      = self.qty,
            entry_price = self.avg_entry_price,
            stop_loss   = self.stop_level,
            notional    = self.market_value,
        )


@dataclass
class ClosedTrade:
    """Immutable record of a fully closed trade, stored in the ledger."""

    symbol:           str
    trade_id:         str
    entry_price:      float
    exit_price:       float
    qty:              int
    realised_pnl:     float
    realised_pnl_pct: float
    holding_bars:     int
    opened_at:        datetime
    closed_at:        datetime
    regime_at_entry:  str = ""
    regime_at_exit:   str = ""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PositionTracker:
    """
    Real-time position tracker backed by Alpaca's ``TradingStream`` WebSocket.

    On startup the tracker reconciles its internal state against live Alpaca
    positions so no position is ever missed, even across process restarts.

    Every fill event updates both the internal ``TrackedPosition`` cache and
    the shared :class:`PortfolioState` / :class:`CircuitBreaker` objects owned
    by the risk manager, ensuring the risk layer always has an accurate view.

    Usage::

        tracker = PositionTracker(client, portfolio_state, circuit_breaker)
        tracker.sync_from_broker()      # reconcile on startup
        tracker.start_stream()          # begin WebSocket fill notifications
        pos = tracker.get_position("AAPL")
    """

    def __init__(
        self,
        client:          AlpacaClient,
        portfolio_state: PortfolioState,
        circuit_breaker: CircuitBreaker,
        on_fill:         Optional[Callable[[str, dict], None]] = None,
    ) -> None:
        """
        Parameters
        ----------
        client:
            Authenticated :class:`AlpacaClient` instance.
        portfolio_state:
            Shared :class:`PortfolioState` instance owned by ``RiskManager``.
            Updated in-place on every fill.
        circuit_breaker:
            Shared :class:`CircuitBreaker` instance owned by ``RiskManager``.
            Re-evaluated after every equity change.
        on_fill:
            Optional external callback invoked after a fill is processed.
            Signature: ``on_fill(symbol: str, fill_event: dict) -> None``.
        """
        self._client          = client
        self._portfolio       = portfolio_state
        self._breaker         = circuit_breaker
        self._on_fill         = on_fill

        # symbol → TrackedPosition (open positions only)
        self._positions: dict[str, TrackedPosition] = {}
        self._lock = threading.Lock()

        # Closed trade ledger
        self._ledger: list[ClosedTrade] = []

        # TradingStream instance (lazy)
        self._stream: Optional[TradingStream] = None
        self._stream_thread: Optional[threading.Thread] = None

    # -------------------------------------------------------------------------
    # Startup reconciliation
    # -------------------------------------------------------------------------

    def sync_from_broker(self) -> None:
        """
        Fetch live positions from the Alpaca REST API and reconcile against
        the internal cache.

        - Positions that exist at the broker but not locally are imported.
        - Positions tracked locally but absent from the broker are removed.

        Call this once on startup (before starting the WebSocket stream).
        """
        broker_positions = self._client.get_positions()
        broker_symbols   = {p["symbol"] for p in broker_positions}

        with self._lock:
            # Import unknown positions
            for bp in broker_positions:
                symbol = bp["symbol"]
                if symbol not in self._positions:
                    qty   = int(float(bp["qty"]))
                    ep    = float(bp["avg_entry_price"])
                    cp    = float(bp["current_price"])
                    cost  = ep * abs(qty)
                    mv    = cp * abs(qty)
                    pnl   = mv - cost
                    tracked = TrackedPosition(
                        symbol            = symbol,
                        trade_id          = "reconciled",
                        side              = "long",
                        qty               = qty,
                        avg_entry_price   = ep,
                        current_price     = cp,
                        stop_level        = 0.0,  # unknown; caller should update
                        cost_basis        = cost,
                        market_value      = mv,
                        unrealised_pnl    = pnl,
                        unrealised_pnl_pct= pnl / cost if cost else 0.0,
                        entry_time        = datetime.now(timezone.utc),
                    )
                    self._positions[symbol] = tracked
                    logger.info("Reconciled position imported: %s  qty=%d", symbol, qty)

            # Remove stale local positions
            stale = [s for s in self._positions if s not in broker_symbols]
            for sym in stale:
                del self._positions[sym]
                logger.info("Stale local position removed: %s", sym)

        # Sync portfolio state
        self._push_to_portfolio()
        logger.info(
            "sync_from_broker complete — %d position(s) reconciled",
            len(self._positions),
        )

    # -------------------------------------------------------------------------
    # WebSocket fill stream
    # -------------------------------------------------------------------------

    def start_stream(self) -> None:
        """
        Start the ``TradingStream`` WebSocket in a daemon background thread.

        Trade update events (fills, partial fills, cancellations) are processed
        asynchronously and update the position cache in real time.
        """
        if not _ALPACA_AVAILABLE:
            raise RuntimeError("alpaca-py is not installed.  Run: pip install alpaca-py")

        if self._stream_thread and self._stream_thread.is_alive():
            logger.warning("Position tracker stream is already running")
            return

        self._stream = TradingStream(
            api_key    = self._client._api_key,
            secret_key = self._client._secret_key,
            paper      = self._client.paper,
        )
        self._stream.subscribe_trade_updates(self._handle_trade_update)

        def _run_stream():
            logger.info("TradingStream started")
            self._stream.run()

        self._stream_thread = threading.Thread(
            target  = _run_stream,
            name    = "position-tracker-stream",
            daemon  = True,
        )
        self._stream_thread.start()
        logger.info("Position tracker WebSocket thread launched")

    def stop_stream(self) -> None:
        """Gracefully stop the TradingStream WebSocket."""
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None
            logger.info("Position tracker stream stopped")

    async def _handle_trade_update(self, update) -> None:
        """
        Async callback invoked by TradingStream on every trade event.

        Handles: ``fill``, ``partial_fill``, ``canceled``, ``expired``.
        All other events are logged and ignored.
        """
        event_type = str(getattr(update, "event", "")).lower()
        order      = getattr(update, "order", None)
        if order is None:
            return

        symbol = getattr(order, "symbol", "")
        fill_price = float(getattr(update, "price", 0) or 0)
        fill_qty   = int(float(getattr(update, "qty", 0) or 0))

        logger.debug(
            "Trade update  event=%s  symbol=%s  qty=%d  price=%.4f",
            event_type, symbol, fill_qty, fill_price,
        )

        if event_type in ("fill", "partial_fill"):
            self._process_fill(update, order, symbol, fill_price, fill_qty)
        elif event_type in ("canceled", "expired", "replaced"):
            logger.info(
                "Order %s  symbol=%s  event=%s", order.id, symbol, event_type
            )

    def _process_fill(self, update, order, symbol: str, fill_price: float, fill_qty: int) -> None:
        """Update position cache and portfolio state for a fill event."""
        side_raw  = str(getattr(order, "side", "buy")).lower()
        is_buy    = (side_raw == "buy")
        client_oid = str(getattr(order, "client_order_id", ""))

        fill_record = {
            "event":      str(getattr(update, "event", "")),
            "price":      fill_price,
            "qty":        fill_qty,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        }

        with self._lock:
            if is_buy:
                if symbol in self._positions:
                    # Scale in / partial fill accumulation
                    pos = self._positions[symbol]
                    total_cost       = pos.avg_entry_price * pos.qty + fill_price * fill_qty
                    pos.qty         += fill_qty
                    pos.avg_entry_price = total_cost / pos.qty if pos.qty else fill_price
                    pos.cost_basis   = pos.avg_entry_price * pos.qty
                    pos.update_price(fill_price)
                    pos.fills.append(fill_record)
                else:
                    # New position
                    cost = fill_price * fill_qty
                    self._positions[symbol] = TrackedPosition(
                        symbol             = symbol,
                        trade_id           = client_oid,
                        side               = "long",
                        qty                = fill_qty,
                        avg_entry_price    = fill_price,
                        current_price      = fill_price,
                        stop_level         = 0.0,
                        cost_basis         = cost,
                        market_value       = cost,
                        unrealised_pnl     = 0.0,
                        unrealised_pnl_pct = 0.0,
                        entry_time         = datetime.now(timezone.utc),
                        fills              = [fill_record],
                    )
                logger.info(
                    "FILL  BUY   %-6s  qty=%d  price=%.4f", symbol, fill_qty, fill_price
                )

            else:  # SELL / exit
                if symbol in self._positions:
                    pos = self._positions[symbol]
                    pos.qty -= fill_qty

                    if pos.qty <= 0:
                        # Position fully closed — record in ledger
                        realised     = (fill_price - pos.avg_entry_price) * abs(fill_qty)
                        realised_pct = realised / pos.cost_basis if pos.cost_basis else 0.0
                        self._ledger.append(ClosedTrade(
                            symbol           = symbol,
                            trade_id         = pos.trade_id,
                            entry_price      = pos.avg_entry_price,
                            exit_price       = fill_price,
                            qty              = fill_qty,
                            realised_pnl     = realised,
                            realised_pnl_pct = realised_pct,
                            holding_bars     = pos.bars_held,
                            opened_at        = pos.entry_time,
                            closed_at        = datetime.now(timezone.utc),
                            regime_at_entry  = pos.regime_at_entry,
                            regime_at_exit   = pos.regime_current,
                        ))
                        del self._positions[symbol]
                        logger.info(
                            "FILL  SELL  %-6s  qty=%d  price=%.4f  "
                            "realised_pnl=%.2f  (position closed)",
                            symbol, fill_qty, fill_price, realised,
                        )
                    else:
                        pos.cost_basis = pos.avg_entry_price * pos.qty
                        pos.update_price(fill_price)
                        logger.info(
                            "FILL  SELL  %-6s  qty=%d  price=%.4f  "
                            "remaining_qty=%d  (partial exit)",
                            symbol, fill_qty, fill_price, pos.qty,
                        )

        # Update shared risk-layer state outside the tracker lock
        self._push_to_portfolio()
        self._evaluate_circuit_breaker()

        if self._on_fill:
            try:
                self._on_fill(symbol, fill_record)
            except Exception as exc:  # noqa: BLE001
                logger.warning("on_fill callback raised: %s", exc)

    # -------------------------------------------------------------------------
    # Live position data
    # -------------------------------------------------------------------------

    def refresh(self) -> None:
        """
        Poll Alpaca for the latest position snapshot and refresh price data.

        Use for periodic reconciliation; fill events are the primary update
        mechanism.
        """
        broker_positions = self._client.get_positions()
        with self._lock:
            for bp in broker_positions:
                sym = bp["symbol"]
                if sym in self._positions:
                    self._positions[sym].update_price(float(bp["current_price"]))
        self._push_to_portfolio()

    def update_bar(self, symbol: str, current_price: float, regime: str = "") -> None:
        """
        Called once per bar by the live trading loop to refresh price and
        increment the holding-period counter.

        Parameters
        ----------
        symbol:
            Ticker to update.
        current_price:
            Latest close price.
        regime:
            Current HMM regime label (stored for attribution).
        """
        with self._lock:
            if symbol in self._positions:
                pos = self._positions[symbol]
                pos.update_price(current_price)
                pos.bars_held    += 1
                pos.regime_current = regime

    def set_stop(self, symbol: str, stop_level: float) -> None:
        """Update the recorded stop level for a position (does NOT send to broker)."""
        with self._lock:
            if symbol in self._positions:
                self._positions[symbol].stop_level = stop_level

    def set_regime_at_entry(self, symbol: str, regime: str) -> None:
        """Record the HMM regime label at the time of entry."""
        with self._lock:
            if symbol in self._positions:
                self._positions[symbol].regime_at_entry = regime

    def set_r_value(self, symbol: str, r_value: float) -> None:
        """Record the per-share risk (entry − stop) at the time of entry."""
        with self._lock:
            if symbol in self._positions:
                self._positions[symbol].r_value = r_value

    def set_tp1_level(self, symbol: str, tp1_level: float) -> None:
        """Record the first partial-exit target price for a position."""
        with self._lock:
            if symbol in self._positions:
                self._positions[symbol].tp1_level = tp1_level

    def set_tp1_hit(self, symbol: str) -> None:
        """Mark TP1 as triggered to prevent duplicate partial exits."""
        with self._lock:
            if symbol in self._positions:
                self._positions[symbol].tp1_hit = True

    def set_tp2_level(self, symbol: str, tp2_level: float) -> None:
        """Record the second partial-exit target price for a position."""
        with self._lock:
            if symbol in self._positions:
                self._positions[symbol].tp2_level = tp2_level

    def set_tp2_hit(self, symbol: str) -> None:
        """Mark TP2 as triggered to prevent duplicate partial exits."""
        with self._lock:
            if symbol in self._positions:
                self._positions[symbol].tp2_hit = True

    def set_tp3_level(self, symbol: str, tp3_level: float) -> None:
        """Record the third partial-exit target price for a position (optional / confidence-gated)."""
        with self._lock:
            if symbol in self._positions:
                self._positions[symbol].tp3_level = tp3_level

    def set_tp3_hit(self, symbol: str) -> None:
        """Mark TP3 as triggered (tightens trailing stop on remainder)."""
        with self._lock:
            if symbol in self._positions:
                self._positions[symbol].tp3_hit = True

    def set_vol_tier(self, symbol: str, vol_tier: str) -> None:
        """Record the strategy vol tier ("LowVol", "MidVol", "HighVol") at entry."""
        with self._lock:
            if symbol in self._positions:
                self._positions[symbol].vol_tier = vol_tier

    def get_all_positions(self) -> dict[str, TrackedPosition]:
        """Return a snapshot of all currently open ``TrackedPosition`` objects."""
        with self._lock:
            return dict(self._positions)

    def get_position(self, symbol: str) -> Optional[TrackedPosition]:
        """Return the ``TrackedPosition`` for a symbol, or ``None``."""
        with self._lock:
            return self._positions.get(symbol)

    def get_notional_map(self) -> dict[str, float]:
        """Return ``symbol → market_value`` for all open positions."""
        with self._lock:
            return {s: p.market_value for s, p in self._positions.items()}

    def get_weight_map(self, total_equity: float) -> dict[str, float]:
        """Return ``symbol → weight`` (market_value / total_equity)."""
        if total_equity <= 0:
            return {}
        with self._lock:
            return {
                s: p.market_value / total_equity
                for s, p in self._positions.items()
            }

    # -------------------------------------------------------------------------
    # P&L metrics
    # -------------------------------------------------------------------------

    def get_portfolio_pnl(self) -> dict:
        """
        Return a summary dict with keys:
          - ``total_unrealised_pnl``
          - ``total_unrealised_pnl_pct``
          - ``total_cost_basis``
          - ``total_market_value``
          - ``position_count``
        """
        with self._lock:
            total_cost = sum(p.cost_basis    for p in self._positions.values())
            total_mv   = sum(p.market_value  for p in self._positions.values())
            total_pnl  = total_mv - total_cost
            return {
                "total_unrealised_pnl":     total_pnl,
                "total_unrealised_pnl_pct": total_pnl / total_cost if total_cost else 0.0,
                "total_cost_basis":         total_cost,
                "total_market_value":       total_mv,
                "position_count":           len(self._positions),
            }

    def get_position_pnl(self, symbol: str) -> Optional[dict]:
        """Return the P&L breakdown for a single symbol, or ``None``."""
        with self._lock:
            pos = self._positions.get(symbol)
            if pos is None:
                return None
            return {
                "symbol":              pos.symbol,
                "qty":                 pos.qty,
                "avg_entry_price":     pos.avg_entry_price,
                "current_price":       pos.current_price,
                "cost_basis":          pos.cost_basis,
                "market_value":        pos.market_value,
                "unrealised_pnl":      pos.unrealised_pnl,
                "unrealised_pnl_pct":  pos.unrealised_pnl_pct,
                "stop_level":          pos.stop_level,
                "bars_held":           pos.bars_held,
                "regime_at_entry":     pos.regime_at_entry,
                "regime_current":      pos.regime_current,
            }

    # -------------------------------------------------------------------------
    # Closed trade ledger
    # -------------------------------------------------------------------------

    def record_closed_trade(self, trade: ClosedTrade) -> None:
        """Append a ``ClosedTrade`` to the internal ledger."""
        self._ledger.append(trade)

    def get_trade_ledger(self) -> pd.DataFrame:
        """Return the full closed-trade ledger as a :class:`pandas.DataFrame`."""
        if not self._ledger:
            return pd.DataFrame()
        return pd.DataFrame([vars(t) for t in self._ledger])

    def get_realised_pnl(self, since: Optional[datetime] = None) -> float:
        """
        Return total realised P&L, optionally filtered to trades closed after
        ``since``.
        """
        trades = self._ledger
        if since is not None:
            trades = [t for t in trades if t.closed_at >= since]
        return sum(t.realised_pnl for t in trades)

    # -------------------------------------------------------------------------
    # Drift detection
    # -------------------------------------------------------------------------

    def get_drift_map(
        self,
        target_weights: dict[str, float],
        total_equity: float,
    ) -> dict[str, float]:
        """
        Return ``symbol → drift`` (current_weight − target_weight) for all
        symbols in ``target_weights``.

        Positive drift = overweight; negative = underweight.
        """
        current = self.get_weight_map(total_equity)
        return {
            sym: current.get(sym, 0.0) - target_weights[sym]
            for sym in target_weights
        }

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _push_to_portfolio(self) -> None:
        """
        Sync the internal position cache into the shared ``PortfolioState``.

        Called after every fill and every refresh() so the risk manager always
        operates on current data.
        """
        with self._lock:
            risk_positions = {
                sym: pos.to_risk_position()
                for sym, pos in self._positions.items()
            }

        self._portfolio.positions = risk_positions

        # Refresh equity from broker periodically (best-effort)
        try:
            acct = self._client.get_account()
            self._portfolio.equity       = acct["equity"]
            self._portfolio.cash         = acct["cash"]
            self._portfolio.buying_power = acct["buying_power"]
            if self._portfolio.peak_equity < acct["equity"]:
                self._portfolio.peak_equity = acct["equity"]
        except Exception as exc:  # noqa: BLE001
            logger.debug("_push_to_portfolio: equity refresh failed (%s)", exc)

    def _evaluate_circuit_breaker(self) -> None:
        """Re-evaluate circuit breakers after an equity-changing fill."""
        try:
            state, fired = self._breaker.check(self._portfolio)
            self._portfolio.circuit_breaker_status = fired
            if fired.value != "NONE":
                logger.warning(
                    "Circuit breaker fired: %s  equity=%.2f  "
                    "daily_dd=%.2f%%  weekly_dd=%.2f%%  peak_dd=%.2f%%",
                    fired.value,
                    self._portfolio.equity,
                    self._portfolio.daily_drawdown  * 100,
                    self._portfolio.weekly_drawdown * 100,
                    self._portfolio.peak_drawdown   * 100,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("_evaluate_circuit_breaker raised: %s", exc)
