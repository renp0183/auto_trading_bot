"""
monitoring — Structured logging, live dashboard, and alerting.
"""

from monitoring.logger import get_logger
from monitoring.dashboard import Dashboard
from monitoring.alerts import AlertManager

__all__ = ["get_logger", "Dashboard", "AlertManager"]
