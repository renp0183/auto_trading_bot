"""
alerts.py — Structured alerting for critical trading events.

Triggers (one enum value per distinct event class):
  REGIME_CHANGE      — HMM regime transitions between states
  CIRCUIT_BREAKER    — daily/weekly/peak drawdown halt or reduce
  LARGE_PNL          — single-session P&L exceeds ±threshold
  FEED_DOWN          — market data WebSocket disconnected / no bars
  API_LOST           — Alpaca REST connectivity lost
  HMM_RETRAINED      — model retrained with new data
  FLICKER_EXCEEDED   — regime change rate above flicker_threshold

Delivery channels (all optional except console):
  Console  — always logged via the alerts logger
  File     — logs/alerts.log (via get_alert_logger)
  Email    — SMTP with TLS (requires smtp_config dict)
  Webhook  — HTTP POST as JSON (Slack, Discord, generic)

Rate limiting:
  Each ``(trigger, symbol)`` key is suppressed for 15 minutes after the
  first dispatch (configurable).  A manual ``reset_rate_limit()`` call
  clears the cache — useful in tests.

Usage::

    mgr = AlertManager(
        smtp_config        = {"host": "smtp.gmail.com", "port": 587,
                               "username": "…", "password": "…",
                               "recipient": "trader@example.com"},
        webhook_url        = "https://hooks.slack.com/services/…",
        rate_limit_minutes = 15,
    )

    mgr.send_regime_change_alert("NEUTRAL", "BULL", confidence=0.82)
    mgr.send_drawdown_alert("daily_halt", current_dd=0.032,
                            threshold=0.03, equity=97_000)
"""

from __future__ import annotations

import json
import logging
import smtplib
import ssl
import threading
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Optional

from monitoring.logger import get_alert_logger

_log = get_alert_logger()


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AlertSeverity(str, Enum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"


class AlertTrigger(str, Enum):
    """Distinct event classes that can produce an alert."""
    REGIME_CHANGE    = "REGIME_CHANGE"
    CIRCUIT_BREAKER  = "CIRCUIT_BREAKER"
    LARGE_PNL        = "LARGE_PNL"
    FEED_DOWN        = "FEED_DOWN"
    API_LOST         = "API_LOST"
    HMM_RETRAINED    = "HMM_RETRAINED"
    FLICKER_EXCEEDED = "FLICKER_EXCEEDED"
    GENERAL_ERROR    = "GENERAL_ERROR"


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    """Immutable record of a single alert event."""

    severity:  AlertSeverity
    trigger:   AlertTrigger
    title:     str
    body:      str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    symbol:    Optional[str]  = None
    metadata:  dict           = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict for webhook payloads."""
        return {
            "severity":  self.severity.value,
            "trigger":   self.trigger.value,
            "title":     self.title,
            "body":      self.body,
            "timestamp": self.timestamp.isoformat(),
            "symbol":    self.symbol,
            "metadata":  self.metadata,
        }


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------

class AlertManager:
    """
    Dispatches alerts to configured channels with per-trigger rate limiting.

    Parameters
    ----------
    smtp_config:
        Dict with keys: ``host``, ``port``, ``username``, ``password``,
        ``recipient``.  If ``None``, email alerts are disabled.
    webhook_url:
        HTTP(S) POST endpoint.  ``None`` disables webhook delivery.
    rate_limit_minutes:
        Minimum minutes between repeated alerts of the same
        ``(trigger, symbol)`` pair (default 15).
    large_pnl_threshold_pct:
        Absolute daily P&L threshold (as fraction of equity) that fires a
        LARGE_PNL alert (default 0.02 = 2 %).
    """

    def __init__(
        self,
        smtp_config:             Optional[dict] = None,
        webhook_url:             Optional[str]  = None,
        rate_limit_minutes:      int            = 15,
        large_pnl_threshold_pct: float          = 0.02,
    ) -> None:
        self._smtp           = smtp_config
        self._webhook_url    = webhook_url
        self._rate_minutes   = rate_limit_minutes
        self._large_pnl_pct  = large_pnl_threshold_pct

        # (trigger_value, symbol_or_None) → last dispatch time
        self._rate_cache: dict[tuple, datetime] = {}
        self._lock = threading.Lock()

    # =========================================================================
    # Primary dispatch
    # =========================================================================

    def send(self, alert: Alert) -> None:
        """
        Dispatch an alert to all configured channels, subject to rate limiting.

        Always logs to the alerts logger.  Email and webhook are optional and
        best-effort (failures are logged but not re-raised).
        """
        if self._is_rate_limited(alert):
            _log.debug(
                "Alert suppressed (rate-limited): %s", alert.title
            )
            return

        self._record_dispatch(alert)

        # ── Console / file (always) ───────────────────────────────────────────
        self._send_console(alert)

        # ── Email ────────────────────────────────────────────────────────────
        if self._smtp:
            try:
                self._send_email(alert)
            except Exception as exc:
                _log.warning("Email alert failed: %s", exc)

        # ── Webhook ───────────────────────────────────────────────────────────
        if self._webhook_url:
            try:
                self._send_webhook(alert)
            except Exception as exc:
                _log.warning("Webhook alert failed: %s", exc)

    # =========================================================================
    # Convenience builders
    # =========================================================================

    def send_regime_change_alert(
        self,
        previous_regime: str,
        new_regime:      str,
        confidence:      float,
        symbol:          Optional[str] = None,
    ) -> None:
        """Fire a REGIME_CHANGE alert when the HMM transitions states."""
        # Severity depends on the severity of the destination regime
        _high_vol = {"CRASH", "STRONG_BEAR", "BEAR"}
        sev = AlertSeverity.WARNING if new_regime in _high_vol else AlertSeverity.INFO
        self.send(Alert(
            severity  = sev,
            trigger   = AlertTrigger.REGIME_CHANGE,
            title     = f"Regime change: {previous_regime} → {new_regime}",
            body      = (
                f"HMM regime transitioned from {previous_regime} to {new_regime}.\n"
                f"Posterior confidence: {confidence:.1%}\n"
                f"Strategy allocations and stop levels will adjust."
            ),
            symbol    = symbol,
            metadata  = {
                "previous_regime": previous_regime,
                "new_regime":      new_regime,
                "confidence":      confidence,
            },
        ))

    def send_drawdown_alert(
        self,
        level:      str,
        current_dd: float,
        threshold:  float,
        equity:     float,
    ) -> None:
        """
        Fire a CIRCUIT_BREAKER alert on drawdown breach.

        Parameters
        ----------
        level:
            One of ``"daily_reduce"``, ``"daily_halt"``, ``"weekly_reduce"``,
            ``"weekly_halt"``, ``"peak"``.
        current_dd:
            Current drawdown fraction (e.g. ``0.031``).
        threshold:
            The breached threshold fraction.
        equity:
            Current portfolio equity in USD.
        """
        halt_levels  = {"daily_halt", "weekly_halt", "peak"}
        sev = AlertSeverity.CRITICAL if level in halt_levels else AlertSeverity.WARNING
        self.send(Alert(
            severity = sev,
            trigger  = AlertTrigger.CIRCUIT_BREAKER,
            title    = f"Circuit breaker fired: {level.upper()}",
            body     = (
                f"Drawdown threshold breached.\n"
                f"Level:    {level}\n"
                f"Current:  {current_dd:.2%}\n"
                f"Limit:    {threshold:.2%}\n"
                f"Equity:   ${equity:,.2f}\n"
                + ("Trading HALTED — manual review required."
                   if level in halt_levels else
                   "Position sizes reduced for rest of session.")
            ),
            metadata = {
                "level":      level,
                "current_dd": current_dd,
                "threshold":  threshold,
                "equity":     equity,
            },
        ))

    def send_large_pnl_alert(
        self,
        daily_pnl:    float,
        equity:       float,
        direction:    str = "loss",
    ) -> None:
        """
        Fire a LARGE_PNL alert when daily P&L exceeds the configured threshold.

        Parameters
        ----------
        daily_pnl:
            Today's P&L in USD (negative for a loss).
        equity:
            Current equity.
        direction:
            ``"gain"`` or ``"loss"``.
        """
        pct = abs(daily_pnl) / max(equity, 1)
        sev = AlertSeverity.CRITICAL if direction == "loss" else AlertSeverity.INFO
        self.send(Alert(
            severity = sev,
            trigger  = AlertTrigger.LARGE_PNL,
            title    = f"Large daily P&L {direction}: {daily_pnl:+,.2f} ({pct:.2%})",
            body     = (
                f"Today's P&L exceeded the {self._large_pnl_pct:.0%} threshold.\n"
                f"P&L:    ${daily_pnl:+,.2f}\n"
                f"Change: {pct:.2%}\n"
                f"Equity: ${equity:,.2f}"
            ),
            metadata = {
                "daily_pnl": daily_pnl,
                "equity":    equity,
                "pct":       pct,
                "direction": direction,
            },
        ))

    def send_feed_down_alert(
        self,
        seconds_silent: float,
        last_bar_time:  Optional[str] = None,
    ) -> None:
        """Fire a FEED_DOWN alert when the data WebSocket stops sending bars."""
        self.send(Alert(
            severity = AlertSeverity.CRITICAL,
            trigger  = AlertTrigger.FEED_DOWN,
            title    = "Market data feed DOWN",
            body     = (
                f"No bar received for {seconds_silent:.0f} seconds.\n"
                f"Last bar: {last_bar_time or 'unknown'}\n"
                "Signal generation paused; open-position stops remain active.\n"
                "Check Alpaca WebSocket connectivity."
            ),
            metadata = {
                "seconds_silent": seconds_silent,
                "last_bar_time":  last_bar_time,
            },
        ))

    def send_api_lost_alert(self, error: str) -> None:
        """Fire an API_LOST alert when the Alpaca REST API becomes unreachable."""
        self.send(Alert(
            severity = AlertSeverity.CRITICAL,
            trigger  = AlertTrigger.API_LOST,
            title    = "Alpaca API connection lost",
            body     = (
                f"REST API unreachable after retries.\n"
                f"Error: {error}\n"
                "Order submission suspended. Attempting auto-reconnect."
            ),
            metadata = {"error": error},
        ))

    def send_hmm_retrained_alert(
        self,
        n_states:      int,
        bic:           float,
        train_bars:    int,
        model_path:    str,
    ) -> None:
        """Fire an HMM_RETRAINED informational alert after a weekly retrain."""
        self.send(Alert(
            severity = AlertSeverity.INFO,
            trigger  = AlertTrigger.HMM_RETRAINED,
            title    = f"HMM retrained: {n_states} states  BIC={bic:.1f}",
            body     = (
                f"Weekly HMM retrain completed.\n"
                f"States:      {n_states}\n"
                f"BIC:         {bic:.2f}\n"
                f"Train bars:  {train_bars:,}\n"
                f"Model saved: {model_path}"
            ),
            metadata = {
                "n_states":   n_states,
                "bic":        bic,
                "train_bars": train_bars,
                "model_path": model_path,
            },
        ))

    def send_flicker_exceeded_alert(
        self,
        flicker_rate:      float,
        flicker_threshold: int,
        flicker_window:    int,
        regime:            str,
    ) -> None:
        """Fire a FLICKER_EXCEEDED alert when regime instability is high."""
        self.send(Alert(
            severity = AlertSeverity.WARNING,
            trigger  = AlertTrigger.FLICKER_EXCEEDED,
            title    = (
                f"Regime flickering: {flicker_rate:.1f}/{flicker_window} bars"
            ),
            body     = (
                f"Regime change rate exceeds threshold.\n"
                f"Rate:      {flicker_rate:.1f} changes / {flicker_window} bars\n"
                f"Threshold: {flicker_threshold} changes\n"
                f"Regime:    {regime}\n"
                "Entering uncertainty mode — position sizes halved, leverage 1×."
            ),
            metadata = {
                "flicker_rate":      flicker_rate,
                "flicker_threshold": flicker_threshold,
                "flicker_window":    flicker_window,
                "regime":            regime,
            },
        ))

    def send_error_alert(
        self,
        error:   Exception,
        context: str = "",
    ) -> None:
        """Fire a GENERAL_ERROR alert on unhandled exceptions."""
        self.send(Alert(
            severity = AlertSeverity.CRITICAL,
            trigger  = AlertTrigger.GENERAL_ERROR,
            title    = f"Unhandled error: {type(error).__name__}",
            body     = (
                f"An unhandled exception occurred.\n"
                f"Context: {context}\n"
                f"Error:   {error}"
            ),
            metadata = {
                "error_type":    type(error).__name__,
                "error_message": str(error),
                "context":       context,
            },
        ))

    # =========================================================================
    # Rate limiting
    # =========================================================================

    def _is_rate_limited(self, alert: Alert) -> bool:
        """
        Return ``True`` if this ``(trigger, symbol)`` was sent within the
        rate-limit window.
        """
        key = (alert.trigger.value, alert.symbol)
        with self._lock:
            last = self._rate_cache.get(key)
        if last is None:
            return False
        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
        return elapsed < self._rate_minutes * 60

    def _record_dispatch(self, alert: Alert) -> None:
        """Mark this trigger as dispatched now."""
        key = (alert.trigger.value, alert.symbol)
        with self._lock:
            self._rate_cache[key] = datetime.now(timezone.utc)

    def reset_rate_limit(
        self,
        trigger: Optional[AlertTrigger] = None,
        symbol:  Optional[str]          = None,
    ) -> None:
        """
        Clear rate-limit entries.

        Parameters
        ----------
        trigger:
            If given, clear only this trigger type.
        symbol:
            If given, further filter to this symbol.
            Pass ``None`` for entries with no associated symbol.
        """
        with self._lock:
            if trigger is None:
                self._rate_cache.clear()
            else:
                keys_to_remove = [
                    k for k in self._rate_cache
                    if k[0] == trigger.value
                    and (symbol is None or k[1] == symbol)
                ]
                for k in keys_to_remove:
                    del self._rate_cache[k]

    # =========================================================================
    # Delivery channels
    # =========================================================================

    def _send_console(self, alert: Alert) -> None:
        """Log the alert to the alerts logger (always fires)."""
        level_map = {
            AlertSeverity.INFO:     logging.INFO,
            AlertSeverity.WARNING:  logging.WARNING,
            AlertSeverity.CRITICAL: logging.CRITICAL,
        }
        level = level_map.get(alert.severity, logging.WARNING)
        _log.log(
            level,
            "[%s] %s — %s",
            alert.trigger.value,
            alert.title,
            alert.body.replace("\n", " | "),
            extra=alert.metadata,
        )

    def _send_email(self, alert: Alert) -> None:
        """
        Dispatch via SMTP with STARTTLS.

        Requires ``smtp_config`` dict with keys:
        ``host``, ``port``, ``username``, ``password``, ``recipient``.
        """
        cfg = self._smtp
        if not cfg:
            return

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[{alert.severity.value}] {alert.title}"
        msg["From"]    = cfg["username"]
        msg["To"]      = cfg["recipient"]

        # Plain-text body
        plain_body = (
            f"Trigger:   {alert.trigger.value}\n"
            f"Severity:  {alert.severity.value}\n"
            f"Time:      {alert.timestamp.isoformat()}\n"
            f"Symbol:    {alert.symbol or 'N/A'}\n\n"
            f"{alert.body}\n\n"
            + (f"Metadata:\n{json.dumps(alert.metadata, indent=2, default=str)}"
               if alert.metadata else "")
        )
        msg.attach(MIMEText(plain_body, "plain"))

        context = ssl.create_default_context()
        with smtplib.SMTP(cfg["host"], int(cfg.get("port", 587))) as server:
            server.ehlo()
            server.starttls(context=context)
            server.login(cfg["username"], cfg["password"])
            server.sendmail(cfg["username"], cfg["recipient"], msg.as_string())

        _log.debug("Email alert sent: %s → %s", alert.title, cfg["recipient"])

    def _send_webhook(self, alert: Alert) -> None:
        """
        POST the alert payload as JSON to the configured webhook URL.

        Compatible with Slack, Discord incoming webhooks, and generic
        HTTP endpoints.
        """
        payload = self._format_payload(alert)

        # Slack / Discord expect a ``text`` or ``content`` key at the top level
        if "slack" in (self._webhook_url or "").lower():
            payload = {
                "text": f"*[{alert.severity.value}]* {alert.title}\n{alert.body}"
            }
        elif "discord" in (self._webhook_url or "").lower():
            payload = {
                "content": f"**[{alert.severity.value}]** {alert.title}\n{alert.body}"
            }

        data    = json.dumps(payload, default=str).encode("utf-8")
        request = urllib.request.Request(
            url     = self._webhook_url,
            data    = data,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )

        with urllib.request.urlopen(request, timeout=10) as resp:
            status = resp.getcode()

        if status not in (200, 204):
            raise RuntimeError(f"Webhook returned HTTP {status}")

        _log.debug("Webhook alert sent: %s (HTTP %s)", alert.title, status)

    def _format_payload(self, alert: Alert) -> dict:
        """Serialise an Alert to a JSON-serialisable dict for webhook POST."""
        return alert.to_dict()
