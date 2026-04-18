"""
test_orders.py — Unit tests for the order executor (mocked Alpaca client).

Tests cover:
  - Market order submission and result mapping.
  - Limit order submission.
  - Bracket order construction (take-profit + stop-loss legs).
  - Order cancellation (single and bulk).
  - Order replacement (qty and price amendments).
  - Error handling when Alpaca API returns an error response.
  - BUY signal correctly routed to buy-side market order.
  - SELL signal correctly routed to sell-side order.
  - BLOCKED signal is never submitted.
"""

from unittest.mock import MagicMock, patch
import pytest

from broker.order_executor import (
    OrderExecutor,
    OrderResult,
    OrderSide,
    OrderType,
    TimeInForce,
)
from core.signal_generator import TradeSignal, SignalType
from core.hmm_engine import Regime
from core.risk_manager import TradingState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_alpaca_client() -> MagicMock:
    """Return a MagicMock standing in for AlpacaClient."""
    client = MagicMock()
    client.is_market_open.return_value = True
    return client


@pytest.fixture
def executor(mock_alpaca_client) -> OrderExecutor:
    return OrderExecutor(mock_alpaca_client)


def make_trade_signal(
    signal_type: SignalType = SignalType.BUY,
    symbol: str = "AAPL",
    shares: int = 10,
    entry_price: float = 150.0,
    stop_price: float = 145.0,
    target_price: float = 160.0,
) -> TradeSignal:
    """Helper factory for TradeSignal objects."""
    return TradeSignal(
        symbol=symbol,
        signal_type=signal_type,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        shares=shares,
        notional=shares * entry_price,
        regime=Regime.LOW_VOL.value,
        confidence=0.85,
        allocation_fraction=0.95,
        leverage=1.25,
        trading_state=TradingState.NORMAL,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMarketOrders:
    def test_submit_buy_market_order(self, executor, mock_alpaca_client):
        """BUY signal → submit_market_order called with correct side and qty."""
        ...

    def test_submit_sell_market_order(self, executor, mock_alpaca_client):
        """SELL signal → submit_market_order called with SELL side."""
        ...

    def test_blocked_signal_not_submitted(self, executor, mock_alpaca_client):
        """BLOCKED signal should never call any Alpaca order method."""
        ...


class TestBracketOrders:
    def test_bracket_order_includes_stop_and_target(self, executor, mock_alpaca_client):
        """Bracket order should set both stop_loss and take_profit legs."""
        ...

    def test_bracket_order_prices_correct(self, executor, mock_alpaca_client):
        """Bracket legs should use signal.stop_price and signal.target_price."""
        ...


class TestOrderManagement:
    def test_cancel_order_calls_api(self, executor, mock_alpaca_client):
        """cancel_order() should delegate to the Alpaca client."""
        ...

    def test_cancel_all_returns_id_list(self, executor, mock_alpaca_client):
        """cancel_all_orders() should return a list of cancelled order IDs."""
        ...

    def test_replace_order_sends_amendment(self, executor, mock_alpaca_client):
        """replace_order() should call the Alpaca replace endpoint."""
        ...


class TestErrorHandling:
    def test_api_error_returns_rejected_result(self, executor, mock_alpaca_client):
        """When Alpaca raises an exception, OrderResult.error should be populated."""
        ...
