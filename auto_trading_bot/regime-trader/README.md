1`1# Regime Trader

A production-grade algorithmic trading bot that uses a Hidden Markov Model (HMM) to detect
market volatility regimes and adjusts capital allocation accordingly.

---

## Philosophy

> **Risk management > signal generation.**

Most trading systems fail not because they cannot identify opportunities, but because they
take on too much risk when they are wrong. Regime Trader is designed around a single insight:
volatility clusters. Drawdowns happen almost exclusively during high-volatility regimes.
The correct response is not to go short — it is to **reduce exposure**.

The system is always long-only. Shorting was tested extensively in walk-forward backtests and
consistently destroyed returns:

- Equity markets have a long-term upward drift.
- V-shaped recoveries happen faster than the HMM can detect them (2–3 bar lag).
- Shorting during rebounds wipes out crash gains.

The edge is in position sizing, not direction.

---

## Architecture

```
 Market Data (Alpaca REST/WebSocket)
         │
         ▼
 ┌───────────────────┐
 │  Feature Engineer │  14 causal features: returns, vol, RSI, ATR, …
 └───────────────────┘
         │
         ▼
 ┌───────────────────┐
 │    HMM Engine     │  Forward algorithm only (causal)
 │  (hmmlearn/BIC)   │  BIC model selection: 3–7 states
 └───────────────────┘
         │ regime_state (label + probability + flicker_rate)
         ▼
 ┌───────────────────┐
 │  Vol Rank Mapper  │  Sorts regimes by expected_volatility (independent of label)
 └───────────────────┘
         │ vol_rank ∈ [0, 1]
         ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                  Strategy Orchestrator                       │
 │  vol_rank ≤ 0.33 → LowVolBull      95% alloc, 1.25× lev   │
 │  vol_rank 0.33–0.67 → MidVolCautious  60–95% alloc, 1.0×  │
 │  vol_rank ≥ 0.67 → HighVolDefensive  60% alloc, 1.0×      │
 └─────────────────────────────────────────────────────────────┘
         │ Signal (direction, size_pct, stop, take_profit)
         ▼
 ┌───────────────────┐
 │   Risk Manager    │  10-gate pipeline: stop, sizing, exposure, correlation,
 │                   │  duplicate guard, circuit breakers
 └───────────────────┘
         │ RiskDecision (approved / rejected / modified)
         ▼
 ┌───────────────────┐
 │  Order Executor   │  LIMIT ±0.1%, bracket orders, 30 s cancel timer,
 │  (alpaca-py)      │  tighten-only stop modification
 └───────────────────┘
         │
         ▼
 ┌─────────────────────────────────────────────────────────────┐
 │            Position Tracker + Portfolio State                │
 │  WebSocket fills → update equity → circuit breaker check    │
 └─────────────────────────────────────────────────────────────┘
         │
         ▼
 ┌───────────────────┐
 │   Dashboard /     │  Rich terminal dashboard (6 panels, 5 s refresh)
 │   Alerts / Logs   │  Email + webhook alerts, 4 rotating JSON log files
 └───────────────────┘
```

---

## Quick Start

### 1. Prerequisites

| Requirement | Version |
|---|---|
| Python | ≤ 3.12 recommended (hmmlearn has no wheel for 3.14) |
| Alpaca account | Paper account (free at [alpaca.markets](https://alpaca.markets)) |

> **Python 3.14 users:** hmmlearn requires either Visual C++ Build Tools 14+ or
> `conda install -c conda-forge hmmlearn`.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure credentials

Copy the credentials template and fill in your Alpaca API keys:

```bash
cp config/credentials.yaml.example config/credentials.yaml
```

Or set environment variables (recommended for production):

```bash
# .env file in project root
ALPACA_API_KEY=your_paper_api_key
ALPACA_SECRET_KEY=your_paper_secret_key
ALPACA_PAPER=true
```

### 4. Backtest first

Always validate the strategy on historical data before paper trading:

```bash
python main.py backtest --symbols SPY AAPL MSFT --compare --stress-test
```

### 5. Train the HMM model

```bash
python main.py train-only --symbols SPY
```

This fetches two years of daily bars, fits the HMM (BIC-selected state count),
and saves `models/hmm_model.pkl`.

### 6. Paper trade

```bash
python main.py live --symbols SPY AAPL MSFT NVDA --dry-run
```

The `--dry-run` flag generates signals and logs them, but never submits orders.
Remove it when you are satisfied with signal quality.

---

## CLI Reference

```
python main.py <subcommand> [options]
```

| Subcommand | Description |
|---|---|
| `live` | Run the live trading loop (paper or live account) |
| `backtest` | Walk-forward backtest with optional benchmarks and stress tests |
| `train-only` | Fetch data, train HMM, save model, then exit |
| `dashboard` | Read `state_snapshot.json` and display a one-shot dashboard |

### `live` options

| Flag | Default | Description |
|---|---|---|
| `--symbols SYM …` | From `settings.yaml` | Symbols to trade |
| `--dry-run` | Off | Log signals but never submit orders |

### `backtest` options

| Flag | Default | Description |
|---|---|---|
| `--symbols SYM …` | `SPY` | Symbols to backtest |
| `--compare` | Off | Also run buy-and-hold and SMA-200 benchmarks |
| `--stress-test` | Off | Run crash MC, gap MC, and regime-misclassification tests |
| `--data-file PATH` | None | Load OHLCV from CSV instead of fetching from Alpaca |
| `--output-dir DIR` | `backtest_results/` | Save equity curves and trade logs here |

### `train-only` options

| Flag | Default | Description |
|---|---|---|
| `--symbols SYM …` | `SPY` | Proxy symbol(s) for HMM training |

---

## Configuration Guide

All parameters live in `config/settings.yaml`. The sections below describe the
most important knobs.

### HMM

```yaml
hmm:
  n_candidates: [3, 4, 5, 6, 7]  # State counts evaluated; best by BIC is selected
  n_init: 10                      # Random restarts per candidate (avoids local optima)
  stability_bars: 3               # Consecutive bars in same raw state before confirming
  flicker_window: 20              # Lookback for flicker detection
  flicker_threshold: 4            # Max regime changes in window before uncertainty mode
  min_confidence: 0.55            # Posterior probability below this → uncertainty mode
```

Increasing `stability_bars` reduces false transitions but adds latency to regime detection.
Increasing `n_init` improves model quality at the cost of longer training time.

### Strategy

```yaml
strategy:
  low_vol_allocation: 0.95        # LowVolBull: 95 % of equity
  mid_vol_allocation_trend: 0.95  # MidVolCautious with trend: 95 %
  mid_vol_allocation_no_trend: 0.60  # MidVolCautious without trend: 60 %
  high_vol_allocation: 0.60       # HighVolDefensive: 60 %
  low_vol_leverage: 1.25          # Only tier that uses leverage
  rebalance_threshold: 0.10       # Drift > 10 % triggers rebalance
  uncertainty_size_mult: 0.50     # Halve size when regime is uncertain
```

### Risk Management

```yaml
risk:
  max_risk_per_trade: 0.01        # Max 1 % of equity at risk per trade
  max_single_position: 0.15       # Max 15 % of equity in one symbol
  max_exposure: 0.80              # Max 80 % of equity deployed at once
  max_concurrent: 5               # Max simultaneous open positions
  daily_dd_reduce: 0.02           # Daily DD > 2 % → halve new position sizes
  daily_dd_halt: 0.03             # Daily DD > 3 % → no new entries rest of day
  weekly_dd_reduce: 0.05          # Weekly DD > 5 % → reduce sizes
  weekly_dd_halt: 0.07            # Weekly DD > 7 % → halt until next week
  max_dd_from_peak: 0.10          # Rolling DD > 10 % → full halt (manual restart)
```

The peak-drawdown halt (`max_dd_from_peak`) writes a `trading_halted.lock` file.
Trading resumes only after you manually delete that file, which forces you to
review the situation before re-entering.

---

## Project Layout

```
regime-trader/
├── config/
│   ├── settings.yaml             # All runtime parameters
│   └── credentials.yaml.example  # API key template
├── core/
│   ├── hmm_engine.py             # HMM training, BIC selection, forward algorithm
│   ├── feature_engineering.py    # 14 causal features
│   ├── regime_strategies.py      # Three strategy tiers + StrategyOrchestrator
│   ├── risk_manager.py           # 10-gate validation pipeline + circuit breakers
│   └── signal_generator.py       # HMM → Strategy → Risk pipeline bridge
├── broker/
│   ├── alpaca_client.py          # alpaca-py wrapper (paper/live, health check)
│   ├── order_executor.py         # LIMIT/bracket orders, stop modification
│   └── position_tracker.py       # WebSocket fills → PortfolioState sync
├── data/
│   ├── market_data.py            # Historical REST + live streaming bars
│   └── feature_engineering.py   # (see core/)
├── backtest/
│   ├── backtester.py             # Walk-forward backtester
│   ├── performance.py            # Sharpe, Sortino, Calmar, confidence breakdown
│   └── stress_test.py            # Crash MC, gap MC, regime-misclassification
├── monitoring/
│   ├── logger.py                 # Structured JSON logs, 4 rotating files
│   ├── alerts.py                 # Email + webhook alerts, rate limiting
│   └── dashboard.py              # Rich terminal dashboard (6 panels)
├── tests/
│   ├── test_hmm.py
│   ├── test_look_ahead.py        # Mandatory no-look-ahead bias tests
│   ├── test_risk.py              # Circuit breakers, sizing, validation
│   ├── test_strategies.py
│   ├── test_orders.py
│   └── test_integration.py       # End-to-end and cross-layer tests
├── main.py                       # CLI entry point + TradingSession
├── requirements.txt
└── README.md
```

---

## FAQ

### Why the forward algorithm and not Viterbi?

Viterbi finds the globally optimal state sequence — but it does so by looking
backward from the end of the sequence. That is **look-ahead bias**: the regime
assigned to bar T depends on data at bars T+1, T+2, … , T+N. In a backtest
this inflates returns; in live trading it is simply not available.

The forward algorithm assigns a probability distribution at bar T using only
data up to and including bar T. It is strictly causal. All predictions in this
system use `predict_filtered_next()`, which is the incremental one-bar-at-a-time
version of the forward pass.

### How does BIC model selection work?

At each training run, the engine fits HMMs with 3, 4, 5, 6, and 7 hidden states
(configurable via `n_candidates`). For each candidate, `n_init` random
initialisations are tried and the best log-likelihood is kept. The final model
is the one with the lowest Bayesian Information Criterion (BIC = log-likelihood
penalty − complexity penalty). BIC penalises extra states more than AIC, which
helps avoid overfitting in small samples.

### Why are my trades being rejected?

Common rejection reasons and what to do:

| Rejection code | Cause | Fix |
|---|---|---|
| `[NO_STOP_LOSS]` | Signal has no stop-loss price | Strategy bug — every signal must include a stop |
| `[ZERO_RISK]` | Stop equals entry price | Check ATR calculation; widen the stop |
| `[DUPLICATE]` | Same symbol+direction within 60 s | Normal — prevents double-entry |
| `[MAX_CONCURRENT]` | Already at 5 open positions | Close a position or increase `max_concurrent` |
| `[MAX_EXPOSURE]` | Portfolio already near 80 % deployed | Wait for a position to close |
| `[DAILY_HALT]` | Daily drawdown exceeded 3 % | Wait for next trading day |
| `[PEAK_DD_HALT]` | Peak drawdown exceeded 10 % | Delete `trading_halted.lock` after reviewing |
| `[CORR_REJECT]` | >85 % correlation with existing position | Too many similar assets; diversify |

### How do I switch to a live account?

1. **Never** paper trade and live trade with the same API key pair.
2. Generate a separate live account key pair in the Alpaca dashboard.
3. Set `ALPACA_PAPER=false` in your `.env` and update the key values.
4. When you run `python main.py live`, you will be prompted to type:

   ```
   YES I UNDERSTAND THE RISKS
   ```

   Exactly as shown. The session will not start without this confirmation.

5. **Start with `--dry-run`** to verify signals look correct before enabling
   actual order submission.

### The HMM labels keep changing after retraining — is that a bug?

No. HMM state labels are assigned by return rank after each fit:
state 0 = lowest mean return (CRASH), state N-1 = highest (EUPHORIA). Because
the new training data has a different mean-return distribution, the same market
environment may map to a different state index. The strategy layer uses
**volatility rank**, not label names, so this does not affect trading behaviour.

### What does "flickering" mean?

Flickering is rapid oscillation between two regimes. If the HMM switches regime
more than `flicker_threshold` times within `flicker_window` bars, the
`flicker_rate` exceeds the threshold and the system enters **uncertainty mode**:
position sizes are halved and leverage is forced to 1.0×. This prevents the
system from thrashing positions during ambiguous market transitions.

---

## Running Tests

```bash
# All tests (requires hmmlearn for look-ahead tests)
python -m pytest tests/ -v

# Skip hmmlearn-dependent tests
python -m pytest tests/ -v -k "not look_ahead"

# Integration tests only
python -m pytest tests/test_integration.py -v

# Risk manager tests with coverage
python -m pytest tests/test_risk.py -v --tb=short
```

---

## Disclaimer

This software is provided for **educational and research purposes only**.

- It does not constitute financial advice.
- There is **no guarantee of profits**. Past backtest performance does not
  predict future live performance.
- Algorithmic trading involves substantial risk of loss.
- **Paper trade first.** Run for at least 30 days in paper mode and review
  drawdown, signal frequency, and rejection rates before risking real capital.
- The authors accept no liability for trading losses.

Always understand a system fully before trading it with real money.
