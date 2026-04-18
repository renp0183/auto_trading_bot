"""
web_dashboard.py — Browser-based live trading dashboard.

Runs a lightweight Flask server inside a daemon thread alongside the main
trading loop.  The browser receives real-time updates via Server-Sent Events
(SSE) so the page stays current without polling.

Access:  http://<server-ip>:8080

Layout sections
---------------
  1. Header bar          — bot name, mode badge, connection dot, UTC clock
  2. Portfolio cards      — equity, daily P&L, weekly P&L, drawdown ring
  3. Circuit-breaker strip — NORMAL / REDUCED / HALTED banner
  4. Per-asset HMM cards  — one card per symbol: label, probability bar, strategy
  5. Open positions table — qty, entry, current price, unrealised P&L, stop, bars
  6. Risk bars            — daily / weekly / peak drawdown gauges with limits
  7. Session stats row    — bars, signals, orders, uptime, feed status
  8. Recent signals table — last 20 signals with full detail

Usage (from TradingSession)
---------------------------
    from monitoring.web_dashboard import WebDashboard

    self.web_dashboard = WebDashboard(port=8080)
    self.web_dashboard.start()

    # each bar:
    self.web_dashboard.push(state_dict)

    # on shutdown:
    self.web_dashboard.stop()
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import queue
import secrets
import threading
from datetime import datetime, timezone
from functools import wraps
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Flask import
# ---------------------------------------------------------------------------
try:
    from flask import Flask, Response, request
    _HAS_FLASK = True
except ImportError:
    _HAS_FLASK = False


# ---------------------------------------------------------------------------
# HTTP Basic Auth helpers (no extra packages — stdlib only)
# ---------------------------------------------------------------------------

def _make_auth_decorator(password: str):
    """
    Return a decorator that enforces HTTP Basic Auth on a Flask route.
    Username is fixed to ``admin``.  Password is the supplied string.
    Uses constant-time comparison to prevent timing attacks.
    """
    def require_auth(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Basic "):
                try:
                    decoded   = base64.b64decode(auth_header[6:]).decode("utf-8", errors="replace")
                    _, _, pwd = decoded.partition(":")
                    if secrets.compare_digest(pwd.encode(), password.encode()):
                        return f(*args, **kwargs)
                except Exception:
                    pass
            return Response(
                "Authentication required.",
                401,
                {"WWW-Authenticate": 'Basic realm="Regime Trader Dashboard"'},
            )
        return decorated
    return require_auth

# ---------------------------------------------------------------------------
# HTML template (single-file, no CDN, no build step)
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Regime Trader — Live Dashboard</title>
<style>
/* ── Reset & base ─────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:        #0d1117;
  --surf:      #161b22;
  --surf2:     #1c2128;
  --border:    #30363d;
  --text:      #e6edf3;
  --muted:     #8b949e;
  --green:     #3fb950;
  --green-dim: #1a4a22;
  --red:       #f85149;
  --red-dim:   #4a1a1a;
  --yellow:    #d29922;
  --yellow-dim:#4a3a0a;
  --blue:      #58a6ff;
  --purple:    #bc8cff;
  --orange:    #e3b341;
  --font:      'SF Mono', 'Fira Code', 'Consolas', monospace;
}
html, body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  font-size: 13px;
  min-height: 100vh;
}
a { color: var(--blue); text-decoration: none; }

/* ── Layout ───────────────────────────────────────────────────────────── */
.container { max-width: 1400px; margin: 0 auto; padding: 0 16px 32px; }

/* ── Header ───────────────────────────────────────────────────────────── */
.header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 0 12px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 16px;
}
.header-left { display: flex; align-items: center; gap: 12px; }
.logo { font-size: 16px; font-weight: 700; letter-spacing: 0.5px; color: var(--text); }
.mode-badge {
  font-size: 11px; font-weight: 700; padding: 2px 8px;
  border-radius: 4px; letter-spacing: 1px;
}
.mode-paper  { background: #1a3a5c; color: var(--blue); border: 1px solid #2a5a8c; }
.mode-live   { background: var(--red-dim); color: var(--red); border: 1px solid #8c2a2a; }
.mode-dryrun { background: #2a2a1a; color: var(--yellow); border: 1px solid #6c5a1a; }
.header-right { display: flex; align-items: center; gap: 16px; color: var(--muted); }
.conn-dot {
  display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  background: var(--green); animation: pulse 2s infinite;
}
.conn-dot.disconnected { background: var(--red); animation: none; }
@keyframes pulse {
  0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
}
#clock { color: var(--text); font-size: 13px; }

/* ── Section titles ───────────────────────────────────────────────────── */
.section-title {
  font-size: 11px; font-weight: 700; letter-spacing: 1.5px;
  color: var(--muted); text-transform: uppercase;
  margin-bottom: 10px; padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
}

/* ── Cards grid ───────────────────────────────────────────────────────── */
.cards-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-bottom: 16px;
}
.card {
  background: var(--surf);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
}
.card-label {
  font-size: 11px; color: var(--muted); font-weight: 600;
  letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 8px;
}
.card-value {
  font-size: 22px; font-weight: 700; line-height: 1.2;
}
.card-sub { font-size: 12px; color: var(--muted); margin-top: 4px; }
.green { color: var(--green); }
.red   { color: var(--red); }
.yellow{ color: var(--yellow); }
.blue  { color: var(--blue); }
.muted { color: var(--muted); }

/* ── Circuit breaker strip ────────────────────────────────────────────── */
.breaker-strip {
  border-radius: 6px;
  padding: 10px 16px;
  margin-bottom: 16px;
  display: flex; align-items: center; gap: 10px;
  font-size: 13px; font-weight: 700; letter-spacing: 0.5px;
}
.breaker-normal  { background: var(--green-dim); border: 1px solid #2a6a32; color: var(--green); }
.breaker-reduced { background: var(--yellow-dim); border: 1px solid #6c5a1a; color: var(--yellow); }
.breaker-halted  { background: var(--red-dim); border: 1px solid #6c2a2a; color: var(--red);
                   animation: flash 1s step-start infinite; }
@keyframes flash { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }

/* ── Regime cards ─────────────────────────────────────────────────────── */
.regime-grid {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 10px;
  margin-bottom: 16px;
}
.regime-card {
  background: var(--surf);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
  text-align: center;
  position: relative;
  overflow: hidden;
}
.regime-card.confirmed { border-color: var(--border); }
.regime-card.unconfirmed { border-style: dashed; }
.regime-sym {
  font-size: 14px; font-weight: 700; margin-bottom: 6px; color: var(--text);
}
.regime-label {
  font-size: 11px; font-weight: 700; padding: 2px 6px;
  border-radius: 4px; display: inline-block; margin-bottom: 6px;
  letter-spacing: 0.5px;
}
.rl-bull       { background: var(--green-dim); color: var(--green); }
.rl-bear       { background: var(--red-dim);   color: var(--red);   }
.rl-neutral    { background: var(--yellow-dim);color: var(--yellow);}
.rl-unknown    { background: #1a1a1a; color: var(--muted); }
.rl-euphoria   { background: #2a1a3a; color: var(--purple); }
.regime-prob {
  font-size: 18px; font-weight: 700; margin-bottom: 4px;
}
.prob-bar-bg {
  background: var(--border); border-radius: 2px; height: 4px; margin-bottom: 6px;
}
.prob-bar { height: 4px; border-radius: 2px; transition: width 0.5s ease; }
.regime-strategy { font-size: 10px; color: var(--muted); }
.regime-tag {
  position: absolute; top: 6px; right: 6px;
  font-size: 9px; padding: 1px 4px; border-radius: 3px;
}
.tag-transition { background: var(--yellow-dim); color: var(--yellow); }
.tag-flicker    { background: var(--red-dim);    color: var(--red); }

/* ── Table shared ─────────────────────────────────────────────────────── */
.table-wrap {
  background: var(--surf);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 16px;
}
.table-wrap .section-title { padding: 12px 16px 10px; margin-bottom: 0; border-bottom: 1px solid var(--border); }
table { width: 100%; border-collapse: collapse; }
thead tr { background: var(--surf2); }
th {
  text-align: left; padding: 8px 12px;
  font-size: 10px; font-weight: 700; color: var(--muted);
  letter-spacing: 0.8px; text-transform: uppercase;
  border-bottom: 1px solid var(--border);
}
td { padding: 9px 12px; border-bottom: 1px solid var(--border); font-size: 12px; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: var(--surf2); }
.empty-row td { text-align: center; color: var(--muted); padding: 24px; }

/* ── Position table ───────────────────────────────────────────────────── */
.sym-badge {
  font-weight: 700; font-size: 12px;
  display: inline-block; padding: 1px 6px;
  border-radius: 4px; background: var(--surf2);
  border: 1px solid var(--border);
}
.dir-long  { color: var(--green); font-weight: 700; }
.dir-flat  { color: var(--muted); }
.pnl-pos   { color: var(--green); font-weight: 600; }
.pnl-neg   { color: var(--red);   font-weight: 600; }
.stop-price { color: var(--orange); font-size: 11px; }

/* ── Risk bars ────────────────────────────────────────────────────────── */
.risk-section {
  background: var(--surf);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin-bottom: 16px;
  overflow: hidden;
}
.risk-section .section-title { padding: 12px 16px 10px; margin-bottom: 0; }
.risk-rows { padding: 12px 16px 16px; display: flex; flex-direction: column; gap: 12px; }
.risk-row { display: grid; grid-template-columns: 100px 1fr 70px 80px; align-items: center; gap: 12px; }
.risk-name { font-size: 12px; color: var(--muted); }
.risk-track { background: var(--border); border-radius: 3px; height: 8px; overflow: hidden; }
.risk-fill  { height: 8px; border-radius: 3px; transition: width 0.5s ease, background 0.5s ease; }
.risk-fill.safe    { background: var(--green); }
.risk-fill.warning { background: var(--yellow); }
.risk-fill.danger  { background: var(--red); }
.risk-pct   { font-size: 12px; font-weight: 700; text-align: right; }
.risk-limit { font-size: 11px; color: var(--muted); text-align: right; }

/* ── Stats row ────────────────────────────────────────────────────────── */
.stats-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-bottom: 16px;
}
.stat-card {
  background: var(--surf);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px 16px;
  display: flex; align-items: center; gap: 12px;
}
.stat-icon { font-size: 20px; }
.stat-body .stat-label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 2px; }
.stat-body .stat-val   { font-size: 16px; font-weight: 700; }

/* ── Signals table ────────────────────────────────────────────────────── */
.signal-dir-long { color: var(--green); font-weight: 700; font-size: 11px; letter-spacing: 0.5px; }
.signal-dir-flat { color: var(--muted); font-size: 11px; }
.reasoning-cell  { max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--muted); font-size: 11px; }
.strategy-chip {
  font-size: 10px; padding: 1px 5px; border-radius: 3px;
  background: var(--surf2); border: 1px solid var(--border); color: var(--muted);
}

/* ── Footer ───────────────────────────────────────────────────────────── */
.footer {
  border-top: 1px solid var(--border);
  padding: 12px 0;
  color: var(--muted); font-size: 11px;
  display: flex; justify-content: space-between; align-items: center;
}

/* ── Responsive ───────────────────────────────────────────────────────── */
@media (max-width: 1100px) {
  .regime-grid    { grid-template-columns: repeat(3, 1fr); }
  .cards-row      { grid-template-columns: repeat(2, 1fr); }
  .stats-row      { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 700px) {
  .regime-grid  { grid-template-columns: repeat(2, 1fr); }
  .cards-row    { grid-template-columns: 1fr 1fr; }
  .stats-row    { grid-template-columns: 1fr 1fr; }
}
</style>
</head>
<body>
<div class="container">

  <!-- ── Header ─────────────────────────────────────────────────────── -->
  <div class="header">
    <div class="header-left">
      <span class="logo">⬡ REGIME TRADER</span>
      <span id="mode-badge" class="mode-badge mode-paper">PAPER</span>
    </div>
    <div class="header-right">
      <span id="conn-dot" class="conn-dot disconnected" title="Waiting for data…"></span>
      <span id="conn-label" style="font-size:11px;">connecting…</span>
      <span id="clock">--:--:-- UTC</span>
    </div>
  </div>

  <!-- ── Portfolio cards ───────────────────────────────────────────── -->
  <div class="cards-row">
    <div class="card">
      <div class="card-label">Equity</div>
      <div class="card-value" id="equity">—</div>
      <div class="card-sub" id="buying-power">Buying power: —</div>
    </div>
    <div class="card">
      <div class="card-label">Daily P&amp;L</div>
      <div class="card-value" id="daily-pnl">—</div>
      <div class="card-sub" id="daily-pnl-pct" style="font-size:13px;">—</div>
    </div>
    <div class="card">
      <div class="card-label">Weekly P&amp;L</div>
      <div class="card-value" id="weekly-pnl">—</div>
      <div class="card-sub" id="weekly-pnl-pct" style="font-size:13px;">—</div>
    </div>
    <div class="card">
      <div class="card-label">Exposure / Positions</div>
      <div class="card-value" id="exposure">—</div>
      <div class="card-sub" id="n-positions">— open positions</div>
    </div>
  </div>

  <!-- ── Circuit breaker ───────────────────────────────────────────── -->
  <div id="breaker-strip" class="breaker-strip breaker-normal">
    <span id="breaker-icon">●</span>
    <span id="breaker-text">CIRCUIT BREAKER: NORMAL</span>
  </div>

  <!-- ── Per-asset HMM regime cards ────────────────────────────────── -->
  <div class="section-title">Per-Asset HMM Regimes</div>
  <div class="regime-grid" id="regime-grid">
    <!-- injected by JS -->
  </div>

  <!-- ── Open positions ────────────────────────────────────────────── -->
  <div class="table-wrap">
    <div class="section-title">Open Positions</div>
    <table>
      <thead>
        <tr>
          <th>Symbol</th><th>Qty</th><th>Entry</th><th>Current</th>
          <th>Unrealised P&amp;L</th><th>P&amp;L %</th><th>Stop</th>
          <th>Regime @ Entry</th><th>Current Regime</th><th>Bars Held</th>
        </tr>
      </thead>
      <tbody id="positions-body">
        <tr class="empty-row"><td colspan="10">No open positions</td></tr>
      </tbody>
    </table>
  </div>

  <!-- ── Risk drawdown bars ─────────────────────────────────────────── -->
  <div class="risk-section">
    <div class="section-title">Drawdown Risk</div>
    <div class="risk-rows">
      <div class="risk-row">
        <span class="risk-name">Daily DD</span>
        <div class="risk-track"><div class="risk-fill safe" id="dd-daily-bar" style="width:0%"></div></div>
        <span class="risk-pct" id="dd-daily-pct">0.00%</span>
        <span class="risk-limit">limit 3.0%</span>
      </div>
      <div class="risk-row">
        <span class="risk-name">Weekly DD</span>
        <div class="risk-track"><div class="risk-fill safe" id="dd-weekly-bar" style="width:0%"></div></div>
        <span class="risk-pct" id="dd-weekly-pct">0.00%</span>
        <span class="risk-limit">limit 7.0%</span>
      </div>
      <div class="risk-row">
        <span class="risk-name">Peak DD</span>
        <div class="risk-track"><div class="risk-fill safe" id="dd-peak-bar" style="width:0%"></div></div>
        <span class="risk-pct" id="dd-peak-pct">0.00%</span>
        <span class="risk-limit">limit 10.0%</span>
      </div>
    </div>
  </div>

  <!-- ── Session stats ─────────────────────────────────────────────── -->
  <div class="stats-row">
    <div class="stat-card">
      <div class="stat-icon">📊</div>
      <div class="stat-body">
        <div class="stat-label">Bars Processed</div>
        <div class="stat-val" id="stat-bars">—</div>
      </div>
    </div>
    <div class="stat-card">
      <div class="stat-icon">⚡</div>
      <div class="stat-body">
        <div class="stat-label">Signals / Orders</div>
        <div class="stat-val" id="stat-orders">—</div>
      </div>
    </div>
    <div class="stat-card">
      <div class="stat-icon">🕒</div>
      <div class="stat-body">
        <div class="stat-label">Uptime</div>
        <div class="stat-val" id="stat-uptime">—</div>
      </div>
    </div>
    <div class="stat-card">
      <div class="stat-icon" id="feed-icon">📡</div>
      <div class="stat-body">
        <div class="stat-label">Data Feed</div>
        <div class="stat-val" id="stat-feed">—</div>
      </div>
    </div>
  </div>

  <!-- ── Recent signals ────────────────────────────────────────────── -->
  <div class="table-wrap">
    <div class="section-title">Recent Signals (last 20)</div>
    <table>
      <thead>
        <tr>
          <th>Time</th><th>Symbol</th><th>Direction</th><th>Size %</th>
          <th>Entry</th><th>Stop</th><th>Take Profit</th>
          <th>Regime</th><th>Confidence</th><th>Strategy</th><th>Reasoning</th>
        </tr>
      </thead>
      <tbody id="signals-body">
        <tr class="empty-row"><td colspan="11">No signals yet</td></tr>
      </tbody>
    </table>
  </div>

  <!-- ── Footer ────────────────────────────────────────────────────── -->
  <div class="footer">
    <span>Regime Trader — HMM Algorithmic Trading Bot</span>
    <span id="last-update">Last update: —</span>
  </div>

</div><!-- /container -->

<script>
// ── Helpers ─────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const fmt  = (n, d=2) => n == null ? '—' : n.toLocaleString('en-US', {minimumFractionDigits:d, maximumFractionDigits:d});
const fmtD = (n, d=2) => n == null ? '—' : (n >= 0 ? '+' : '') + fmt(n, d);
const fmtUSD = n => n == null ? '—' : '$' + fmt(n, 2);
const fmtPct = n => n == null ? '—' : (n >= 0 ? '+' : '') + fmt(n, 2) + '%';

function colorClass(n) { return n > 0 ? 'green' : n < 0 ? 'red' : 'muted'; }

function regimeLabelClass(label) {
  if (!label) return 'rl-unknown';
  const l = label.toUpperCase();
  if (['BULL','STRONG_BULL','WEAK_BULL'].some(x => l.includes(x))) return 'rl-bull';
  if (['BEAR','STRONG_BEAR','WEAK_BEAR','CRASH'].some(x => l.includes(x))) return 'rl-bear';
  if (['EUPHORIA','CAUTIOUS_GROWTH'].some(x => l.includes(x))) return 'rl-euphoria';
  if (['NEUTRAL'].some(x => l.includes(x))) return 'rl-neutral';
  return 'rl-unknown';
}

function regimeColor(label) {
  const cls = regimeLabelClass(label);
  return {
    'rl-bull':     '#3fb950',
    'rl-bear':     '#f85149',
    'rl-euphoria': '#bc8cff',
    'rl-neutral':  '#d29922',
    'rl-unknown':  '#8b949e',
  }[cls] || '#8b949e';
}

function uptime(secs) {
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = secs % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function shortTime(ts) {
  if (!ts) return '—';
  try {
    const d = new Date(ts);
    return d.toISOString().substring(11, 19) + ' UTC';
  } catch { return ts; }
}

// ── Clock ────────────────────────────────────────────────────────────────
function tickClock() {
  const now = new Date();
  $('clock').textContent = now.toISOString().substring(11, 19) + ' UTC';
}
tickClock();
setInterval(tickClock, 1000);

// ── Render ───────────────────────────────────────────────────────────────
function render(d) {
  // ── Mode badge
  const modeBadge = $('mode-badge');
  const mode = (d.mode || 'PAPER').toUpperCase();
  modeBadge.textContent = mode;
  modeBadge.className = 'mode-badge ' + (
    mode === 'LIVE' ? 'mode-live' :
    mode === 'DRY-RUN' ? 'mode-dryrun' : 'mode-paper'
  );

  // ── Portfolio cards
  const pf = d.portfolio || {};
  const eq = $('equity');
  eq.textContent = fmtUSD(pf.equity);
  eq.className   = 'card-value';

  $('buying-power').textContent = 'Buying power: ' + fmtUSD(pf.buying_power);

  const dp = $('daily-pnl');
  dp.textContent = fmtD(pf.daily_pnl);
  dp.className   = 'card-value ' + colorClass(pf.daily_pnl);
  $('daily-pnl-pct').textContent = fmtPct(pf.daily_pnl_pct);
  $('daily-pnl-pct').className = colorClass(pf.daily_pnl);

  const wp = $('weekly-pnl');
  wp.textContent = fmtD(pf.weekly_pnl);
  wp.className   = 'card-value ' + colorClass(pf.weekly_pnl);
  $('weekly-pnl-pct').textContent = fmtPct(pf.weekly_pnl_pct);
  $('weekly-pnl-pct').className = colorClass(pf.weekly_pnl);

  $('exposure').textContent = fmt(pf.total_exposure_pct, 1) + '% deployed';
  $('n-positions').textContent = (pf.n_positions || 0) + ' open position(s)';

  // ── Circuit breaker
  const breaker = (d.circuit_breaker || 'NORMAL').toUpperCase();
  const btype   = (d.breaker_type   || 'NONE').toUpperCase();
  const strip   = $('breaker-strip');
  strip.className = 'breaker-strip breaker-' + breaker.toLowerCase();
  $('breaker-text').textContent =
    'CIRCUIT BREAKER: ' + breaker + (btype !== 'NONE' ? '  (' + btype + ')' : '');

  // ── Regime cards
  const grid     = $('regime-grid');
  const regimes  = d.regime_labels || {};
  const symbols  = Object.keys(regimes);
  if (symbols.length === 0) {
    grid.innerHTML = '<div style="color:var(--muted);grid-column:1/-1;padding:16px;">Waiting for first bar…</div>';
  } else {
    grid.innerHTML = symbols.map(sym => {
      const r      = regimes[sym];
      const label  = r.label || 'UNKNOWN';
      const prob   = r.probability || 0;
      const cls    = regimeLabelClass(label);
      const color  = regimeColor(label);
      const strat  = (r.strategy || '').replace(/([A-Z])/g, ' $1').trim();
      const tags   = [
        r.in_transition ? '<span class="regime-tag tag-transition">TRANS</span>' : '',
        (r.flicker_rate > 0.2) ? '<span class="regime-tag tag-flicker">FLICKER</span>' : '',
      ].join('');
      return `
        <div class="regime-card ${r.confirmed ? 'confirmed' : 'unconfirmed'}">
          ${tags}
          <div class="regime-sym">${sym}</div>
          <span class="regime-label ${cls}">${label}</span>
          <div class="regime-prob" style="color:${color}">${Math.round(prob * 100)}%</div>
          <div class="prob-bar-bg">
            <div class="prob-bar" style="width:${Math.round(prob*100)}%;background:${color}"></div>
          </div>
          <div class="regime-strategy">${strat}</div>
        </div>`;
    }).join('');
  }

  // ── Positions table
  const positions = d.positions || [];
  const posBody   = $('positions-body');
  if (positions.length === 0) {
    posBody.innerHTML = '<tr class="empty-row"><td colspan="10">No open positions</td></tr>';
  } else {
    posBody.innerHTML = positions.map(p => {
      const pnl     = p.unrealised_pnl || 0;
      const pnlPct  = p.unrealised_pnl_pct || 0;
      const pnlCls  = pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
      const rCls    = regimeLabelClass(p.regime_current);
      return `<tr>
        <td><span class="sym-badge">${p.symbol}</span></td>
        <td>${(p.qty||0).toLocaleString()}</td>
        <td>$${fmt(p.avg_entry_price)}</td>
        <td>$${fmt(p.current_price)}</td>
        <td class="${pnlCls}">${fmtD(pnl)}</td>
        <td class="${pnlCls}">${fmtD(pnlPct, 2)}%</td>
        <td class="stop-price">$${fmt(p.stop_level)}</td>
        <td><span class="regime-label ${regimeLabelClass(p.regime_at_entry)}">${p.regime_at_entry||'?'}</span></td>
        <td><span class="regime-label ${rCls}">${p.regime_current||'?'}</span></td>
        <td>${p.bars_held||0}</td>
      </tr>`;
    }).join('');
  }

  // ── Risk bars
  function setBar(barId, pctId, value, limit) {
    const fill   = Math.min((value / limit) * 100, 100);
    const el     = $(barId);
    el.style.width = fill + '%';
    el.className = 'risk-fill ' + (fill >= 80 ? 'danger' : fill >= 50 ? 'warning' : 'safe');
    $(pctId).textContent = fmt(value, 2) + '%';
    $(pctId).className   = 'risk-pct ' + (fill >= 80 ? 'red' : fill >= 50 ? 'yellow' : 'green');
  }
  setBar('dd-daily-bar',  'dd-daily-pct',  pf.daily_drawdown  || 0, 3.0);
  setBar('dd-weekly-bar', 'dd-weekly-pct', pf.weekly_drawdown || 0, 7.0);
  setBar('dd-peak-bar',   'dd-peak-pct',   pf.peak_drawdown   || 0, 10.0);

  // ── Session stats
  const ss = d.session_stats || {};
  $('stat-bars').textContent   = (ss.bars_processed || 0).toLocaleString();
  $('stat-orders').textContent = `${ss.signals_generated||0} sig / ${ss.orders_submitted||0} ord`;
  $('stat-uptime').textContent = ss.uptime_seconds != null ? uptime(ss.uptime_seconds) : '—';

  const feedOk = ss.feed_ok !== false;
  $('stat-feed').textContent = feedOk ? 'OK  ' + shortTime(ss.last_bar_time) : 'DOWN';
  $('stat-feed').className   = 'stat-val ' + (feedOk ? 'green' : 'red');
  $('feed-icon').textContent = feedOk ? '📡' : '⚠️';

  // ── Signals table
  const signals  = (d.recent_signals || []).slice().reverse();
  const sigBody  = $('signals-body');
  if (signals.length === 0) {
    sigBody.innerHTML = '<tr class="empty-row"><td colspan="11">No signals yet this session</td></tr>';
  } else {
    sigBody.innerHTML = signals.map(s => {
      const dirCls = s.direction === 'LONG' ? 'signal-dir-long' : 'signal-dir-flat';
      const tp     = s.take_profit ? '$' + fmt(s.take_profit) : '—';
      return `<tr>
        <td style="color:var(--muted);font-size:11px">${shortTime(s.timestamp)}</td>
        <td><span class="sym-badge">${s.symbol}</span></td>
        <td class="${dirCls}">${s.direction}</td>
        <td>${fmt(s.position_size_pct, 1)}%</td>
        <td>$${fmt(s.entry_price)}</td>
        <td class="stop-price">$${fmt(s.stop_loss)}</td>
        <td>${tp}</td>
        <td><span class="regime-label ${regimeLabelClass(s.regime_name)}">${s.regime_name||'?'}</span></td>
        <td>${Math.round((s.regime_probability||0)*100)}%</td>
        <td><span class="strategy-chip">${s.strategy_name||'—'}</span></td>
        <td class="reasoning-cell" title="${(s.reasoning||'').replace(/"/g,"'")}">${s.reasoning||'—'}</td>
      </tr>`;
    }).join('');
  }

  // ── Last update timestamp
  $('last-update').textContent = 'Last update: ' + shortTime(d.timestamp);
}

// ── SSE connection ────────────────────────────────────────────────────────
function connect() {
  const dot   = $('conn-dot');
  const label = $('conn-label');
  const es    = new EventSource('/events');

  es.onopen = () => {
    dot.className   = 'conn-dot';
    label.textContent = 'live';
  };

  es.onmessage = e => {
    try {
      const data = JSON.parse(e.data);
      if (data && data.timestamp) render(data);
    } catch (err) {
      console.warn('parse error:', err);
    }
  };

  es.onerror = () => {
    dot.className     = 'conn-dot disconnected';
    label.textContent = 'reconnecting…';
    es.close();
    setTimeout(connect, 3000);
  };
}

connect();
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
# WebDashboard class
# ---------------------------------------------------------------------------

class WebDashboard:
    """
    Lightweight Flask web server that serves the live trading dashboard.

    Runs in a daemon thread — the main trading loop calls ``push()`` after
    every bar and the browser receives the update instantly via SSE.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        if not _HAS_FLASK:
            raise ImportError(
                "Flask is required for the web dashboard.  "
                "Install it with:  pip install flask"
            )

        self._host  = host
        self._port  = port
        self._state: dict = {}
        self._lock  = threading.Lock()
        self._subscribers: list[queue.Queue] = []
        self._sub_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._sys_state: dict = {}  # set via set_system_state()

        # ── Auth: read password from env, generate a random one if absent ──
        password = os.environ.get("DASHBOARD_PASSWORD", "")
        if not password:
            password = secrets.token_urlsafe(16)
            log.warning(
                "DASHBOARD_PASSWORD not set — generated a one-time password: %s  "
                "(set DASHBOARD_PASSWORD in .env to make it permanent)",
                password,
            )
        self._password = password
        self._app = self._build_app()

    # ── Public API ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the Flask server in a background daemon thread."""
        self._thread = threading.Thread(
            target=self._run,
            name="web-dashboard",
            daemon=True,
        )
        self._thread.start()
        log.info(
            "Web dashboard started — SSH tunnel to access:  "
            "ssh -i <key> -L 8080:localhost:%d ubuntu@<server-ip> -N  "
            "then open  http://localhost:%d  (user: admin)",
            self._port, self._port,
        )

    def stop(self) -> None:
        """No-op — daemon thread exits automatically with the process."""
        pass

    def set_system_state(self, state: dict) -> None:
        """
        Inject a SystemState snapshot for the /health endpoint.

        Call after startup() completes and after every bar so /health always
        reflects the live system state.  No auth required on /health.
        """
        with self._lock:
            self._sys_state = state

    def push(self, state: dict) -> None:
        """
        Push a new state snapshot to all connected browsers.

        Parameters
        ----------
        state : dict
            JSON-serialisable dict produced by ``TradingSession._build_web_state()``.
        """
        with self._lock:
            self._state = state
        payload = json.dumps(state, default=str)
        dead = []
        with self._sub_lock:
            for q in self._subscribers:
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._subscribers.remove(q)

    # ── Internal ──────────────────────────────────────────────────────────

    def _run(self) -> None:
        import logging as _logging
        wl = _logging.getLogger("werkzeug")
        wl.setLevel(_logging.ERROR)          # suppress Flask request logs
        self._app.run(
            host        = self._host,
            port        = self._port,
            threaded    = True,
            use_reloader= False,
            debug       = False,
        )

    def _build_app(self) -> "Flask":
        app = Flask(__name__)
        app.config["SECRET_KEY"] = secrets.token_hex(32)

        protected = _make_auth_decorator(self._password)

        @app.route("/")
        @protected
        def index():
            return _HTML, 200, {"Content-Type": "text/html"}

        @app.route("/api/state")
        @protected
        def api_state():
            with self._lock:
                snapshot = dict(self._state)
            return app.response_class(
                response    = json.dumps(snapshot, default=str),
                status      = 200,
                mimetype    = "application/json",
            )

        @app.route("/events")
        @protected
        def events():
            """Server-Sent Events endpoint — streams state to the browser."""
            sub_q: queue.Queue = queue.Queue(maxsize=4)

            with self._sub_lock:
                self._subscribers.append(sub_q)

            # Immediately send the current state so the page populates on load
            with self._lock:
                initial = json.dumps(self._state, default=str) if self._state else None

            def stream():
                try:
                    if initial:
                        yield f"data: {initial}\n\n"
                    while True:
                        try:
                            payload = sub_q.get(timeout=25)
                            yield f"data: {payload}\n\n"
                        except queue.Empty:
                            # Keepalive ping so the connection stays open through
                            # proxies / load balancers that close idle connections
                            yield ": keepalive\n\n"
                except GeneratorExit:
                    pass
                finally:
                    with self._sub_lock:
                        if sub_q in self._subscribers:
                            self._subscribers.remove(sub_q)

            return Response(
                stream(),
                content_type = "text/event-stream",
                headers      = {
                    "Cache-Control":    "no-cache",
                    "X-Accel-Buffering":"no",
                    "Connection":       "keep-alive",
                },
            )

        @app.route("/health")
        def health():
            with self._lock:
                payload = dict(self._sys_state) if self._sys_state else {}
            payload["status"]    = "ok"
            payload["timestamp"] = datetime.now(timezone.utc).isoformat()
            return app.response_class(
                response  = json.dumps(payload, default=str),
                status    = 200,
                mimetype  = "application/json",
            )

        return app
