"""
data — Market data fetching and feature engineering.
"""

from data.feature_engineering import FeatureEngineer

# MarketDataFeed depends on the broker (alpaca) package — import lazily so that
# feature_engineering and hmm_engine can be used in tests without the full
# broker stack installed.
try:
    from data.market_data import MarketDataFeed
    __all__ = ["MarketDataFeed", "FeatureEngineer"]
except ImportError:
    __all__ = ["FeatureEngineer"]
