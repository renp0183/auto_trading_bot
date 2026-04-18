"""
diagnose.py — Pipeline failure diagnosis script.

Run this INSTEAD of main.py to find exactly where the zero-activity bug is.
It exercises each layer in isolation and stops at the first failure.

Usage:
    python diagnose.py
"""

from __future__ import annotations
import os, sys, traceback
from pathlib import Path

# Load credentials exactly as main.py does
try:
    from dotenv import load_dotenv as _load_dotenv_impl
    _env_file = Path(__file__).parent / ".env"
    if _env_file.exists():
        _load_dotenv_impl(_env_file, override=False)
        print(f"[OK] .env loaded from {_env_file}")
    else:
        print(f"[WARN] .env not found at {_env_file} — using shell environment")
except ImportError:
    print("[WARN] python-dotenv not installed — using shell environment only")

api_key = os.getenv("ALPACA_API_KEY", "")
secret  = os.getenv("ALPACA_SECRET_KEY", "")
print(f"[INFO] API key present: {bool(api_key)}  (first 4 chars: {api_key[:4] if api_key else 'MISSING'})")
print(f"[INFO] Secret present:  {bool(secret)}")
if not api_key or not secret:
    print("[FATAL] Missing credentials — all Alpaca calls will fail.  Check .env")
    sys.exit(1)

# ── Layer 1: API connectivity ─────────────────────────────────────────────────
print("\n" + "="*60)
print("LAYER 1: Alpaca API connectivity")
print("="*60)
try:
    from broker.alpaca_client import AlpacaClient
    client = AlpacaClient(paper=True)
    acct = client.get_account()
    print(f"[OK] Account connected  equity=${acct['equity']:,.2f}  buying_power=${acct['buying_power']:,.2f}")
except Exception as exc:
    print(f"[FATAL] AlpacaClient failed: {exc}")
    traceback.print_exc()
    sys.exit(1)

# ── Layer 2: Historical data load ─────────────────────────────────────────────
print("\n" + "="*60)
print("LAYER 2: Historical data load (REST backfill)")
print("="*60)
try:
    import yaml
    with open("config/settings.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    symbols   = cfg["broker"]["symbols"]
    timeframe = cfg["broker"]["timeframe"]
    print(f"[INFO] Symbols: {symbols}  timeframe: {timeframe}")
except Exception as exc:
    print(f"[WARN] Could not load settings.yaml: {exc} — using defaults")
    symbols   = ["GDX", "XLK", "XLE", "TQQQ", "SPXL", "TLT"]
    timeframe = "1Day"

from data.market_data import MarketDataFeed
feed = MarketDataFeed(client, symbols=symbols, timeframe=timeframe)

try:
    feed.initialise(lookback_days=504)
except Exception as exc:
    print(f"[FATAL] feed.initialise() raised: {exc}")
    traceback.print_exc()
    sys.exit(1)

avail = feed.bars_available()
print()
all_ok = True
for sym, n in avail.items():
    status = "[OK] " if n >= 252 else ("[WARN]" if n > 0 else "[FATAL]")
    print(f"  {status} {sym}: {n} bars loaded  (need ≥252)")
    if n == 0:
        all_ok = False

if not all_ok:
    print("\n[FATAL] One or more symbols have 0 bars.")
    print("  This is why the main loop stalls: get_bars(clock_sym) returns empty.")
    print("  Possible causes:")
    print("  1. Alpaca data plan does not include this symbol on IEX feed")
    print("  2. Symbol ticker incorrect (e.g. TLT may need 'TLT' exactly)")
    print("  3. Rate limit or network error during backfill")
    print("  Fix: check logs above for 'Failed to fetch history for X'")
    sys.exit(1)

# ── Layer 3: Feature engineering ─────────────────────────────────────────────
print("\n" + "="*60)
print("LAYER 3: Feature engineering on loaded bars")
print("="*60)
from data.feature_engineering import FeatureEngineer
fe = FeatureEngineer()

for sym in symbols:
    bars = feed.get_bars(sym)
    try:
        features = fe.build_features(bars).dropna()
        print(f"  [OK] {sym}: {len(bars)} bars → {len(features)} valid feature rows")
        if len(features) < 252:
            print(f"  [WARN] {sym}: only {len(features)} feature rows (need ≥252 for HMM)")
    except Exception as exc:
        print(f"  [FATAL] {sym}: feature_engineering failed: {exc}")
        traceback.print_exc()

# ── Layer 4: HMM fitted? ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("LAYER 4: HMM model availability")
print("="*60)
MODEL_DIR = Path("models")
for sym in symbols:
    model_file = MODEL_DIR / f"{sym.upper()}_hmm.pkl"
    if model_file.exists():
        import datetime
        age_days = (datetime.datetime.now() - datetime.datetime.fromtimestamp(model_file.stat().st_mtime)).days
        print(f"  [OK] {sym}: model exists  age={age_days}d  path={model_file}")
    else:
        print(f"  [WARN] {sym}: no model file — will train on startup (slow first run)")

# ── Layer 5: WebSocket stream thread alive? ────────────────────────────────────
print("\n" + "="*60)
print("LAYER 5: WebSocket stream (subscribe + thread start)")
print("="*60)
import time, threading
try:
    feed.start_stream()
    time.sleep(2)
    t = feed._stream_thread
    alive = t is not None and t.is_alive()
    print(f"  Stream thread alive: {alive}  name={t.name if t else 'None'}")
    if not alive:
        print("  [FATAL] Stream thread died immediately after start.")
        print("  Check broker/alpaca_client.py run_stream() for exceptions.")
    else:
        print("  [OK] WebSocket thread is running")
except Exception as exc:
    print(f"  [FATAL] start_stream() raised: {exc}")
    traceback.print_exc()

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("If all layers show [OK], the bot CAN process bars.")
print("Zero activity then means: no new daily close bar has arrived yet.")
print("Daily bars from Alpaca arrive at ~4:15pm ET on trading days.")
print("The buffer pre-loads history; the NEXT bar after startup triggers the first trade cycle.")
print()
print("If any layer shows [FATAL], fix that layer first — it is the blocker.")
