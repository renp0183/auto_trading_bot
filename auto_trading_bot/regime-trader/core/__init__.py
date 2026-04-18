"""
core — HMM engine, regime strategies, risk management, and signal generation.
"""

from core.hmm_engine import HMMEngine, HMMConfig, RegimeState, RegimeInfo, RegimeLabel, Regime

try:
    from core.regime_strategies import RegimeStrategy
    from core.risk_manager import RiskManager
    from core.signal_generator import SignalGenerator
    __all__ = [
        "HMMEngine", "HMMConfig", "RegimeState", "RegimeInfo",
        "RegimeLabel", "Regime",
        "RegimeStrategy", "RiskManager", "SignalGenerator",
    ]
except ImportError:
    __all__ = ["HMMEngine", "HMMConfig", "RegimeState", "RegimeInfo", "RegimeLabel", "Regime"]
