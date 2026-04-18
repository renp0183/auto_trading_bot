"""
feature_engineering.py — Technical indicator computation and HMM feature matrix construction.

All features are computed with STRICTLY CAUSAL logic — only past observations are used
at each time step.  Rolling z-score normalisation uses only data available up to bar t.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import ta
from ta.trend import ADXIndicator, SMAIndicator
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import AverageTrueRange

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Canonical feature column names — HMMEngine references these by name
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_NAMES: list[str] = [
    "log_return_1",      # 1-bar log return  (primary return signal for regime labelling)
    "log_return_5",      # 5-bar log return
    "log_return_20",     # 20-bar log return
    "realized_vol_20",   # 20-period annualised realised vol
    "vol_ratio_5_20",    # Short/long vol ratio (momentum of vol)
    "volume_zscore",     # Volume z-score vs 50-period mean
    "volume_trend",      # Slope of 10-period volume SMA
    "adx_14",            # Average Directional Index (trend strength)
    "sma50_slope",       # 50-period SMA slope (pct change)
    "rsi_14_zscore",     # RSI(14) rolling z-score
    "dist_from_200sma",  # (close − SMA200) / close
    "roc_10",            # Rate of change over 10 bars
    "roc_20",            # Rate of change over 20 bars
    "norm_atr_14",       # ATR(14) / close  (normalised range)
    # ── Distributional shape features (added for better HMM state separation) ─
    "skew_14",           # 14-bar rolling skewness of log_return_1
    "kurt_14",           # 14-bar rolling excess kurtosis of log_return_1
    "autocorr_lag1_20",  # Lag-1 autocorrelation of log_return_1 over 20 bars
    "ret_q05_20",        # 5th-percentile log return over 20 bars (left-tail risk)
]


class FeatureEngineer:
    """
    Transforms raw OHLCV bars into a standardised feature matrix for the HMM engine.

    All indicators are causal:  at bar t, only data from bars 0 … t−1 (plus t itself
    for contemporaneous values) is used.  Rolling z-scores use a 252-bar lookback,
    which is also causal.

    Parameters
    ----------
    zscore_window : int
        Lookback for the rolling z-score standardisation step.  Default 252.
    min_periods_frac : float
        Minimum fraction of ``zscore_window`` required before z-score is valid.
        Default 0.5 → 126 bars.

    Usage
    -----
    ::

        engineer = FeatureEngineer()
        features = engineer.build_features(ohlcv_df)
        # features is a DataFrame with columns == FEATURE_NAMES (NaN rows dropped)
    """

    def __init__(
        self,
        zscore_window: int = 252,
        min_periods_frac: float = 0.5,
    ) -> None:
        self.zscore_window = zscore_window
        self._min_periods = max(1, int(zscore_window * min_periods_frac))

    # ─────────────────────────────────────────────────────────────────────────
    # Primary interface
    # ─────────────────────────────────────────────────────────────────────────

    def build_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all 18 features and return a clean, z-scored feature matrix.

        Features 1-14: existing price/vol/momentum indicators.
        Features 15-18: distributional shape (skew, kurtosis, autocorrelation,
        left-tail quantile) that improve HMM state separation between regimes
        with similar realized vol but different return distributions.

        Parameters
        ----------
        ohlcv : DataFrame
            Must contain columns ``open``, ``high``, ``low``, ``close``, ``volume``.
            Index should be a DatetimeIndex.

        Returns
        -------
        DataFrame
            Shape (n_valid_bars, 18) with columns matching ``FEATURE_NAMES``.
            Rows that contain any NaN (insufficient history) are dropped.
        """
        ohlcv = self._validate_ohlcv(ohlcv)
        high  = ohlcv["high"]
        low   = ohlcv["low"]
        close = ohlcv["close"]
        vol   = ohlcv["volume"].astype(float)

        # ── Raw (unstandardised) features ────────────────────────────────────
        lr1 = self._log_return(close, 1)
        lr5 = self._log_return(close, 5)
        lr20 = self._log_return(close, 20)

        rv20   = self._realized_vol(lr1, window=20)
        vr5_20 = self._vol_ratio(lr1, short=5, long=20)

        vol_z = self._volume_zscore(vol, window=50)
        vol_t = self._volume_trend(vol, sma_window=10)

        adx = self._adx(high, low, close, window=14)
        sma50_sl = self._sma_slope(close, window=50)

        rsi = self._rsi(close, window=14)
        rsi_z = self._rolling_zscore(rsi, window=self.zscore_window)

        dist200 = self._dist_from_sma(close, window=200)

        roc10 = self._roc(close, period=10)
        roc20 = self._roc(close, period=20)

        norm_atr = self._norm_atr(high, low, close, window=14)

        # ── Distributional shape features ─────────────────────────────────────
        skew14        = self._rolling_skew(lr1, window=14)
        kurt14        = self._rolling_kurt(lr1, window=14)
        autocorr_lag1 = self._autocorr_lag1(lr1, window=20)
        ret_q05       = self._ret_quantile(lr1, window=20, quantile=0.05)

        # ── Assemble raw frame ────────────────────────────────────────────────
        raw = pd.DataFrame(
            {
                "log_return_1":     lr1,
                "log_return_5":     lr5,
                "log_return_20":    lr20,
                "realized_vol_20":  rv20,
                "vol_ratio_5_20":   vr5_20,
                "volume_zscore":    vol_z,
                "volume_trend":     vol_t,
                "adx_14":           adx,
                "sma50_slope":      sma50_sl,
                "rsi_14_zscore":    rsi_z,
                "dist_from_200sma": dist200,
                "roc_10":           roc10,
                "roc_20":           roc20,
                "norm_atr_14":      norm_atr,
                "skew_14":          skew14,
                "kurt_14":          kurt14,
                "autocorr_lag1_20": autocorr_lag1,
                "ret_q05_20":       ret_q05,
            },
            index=ohlcv.index,
        )

        # ── Apply rolling z-score to raw features (except RSI which is already) ──
        # RSI was already z-scored above; z-score the remaining features.
        already_zscored = {"rsi_14_zscore"}
        for col in raw.columns:
            if col not in already_zscored:
                raw[col] = self._rolling_zscore(raw[col], window=self.zscore_window)

        # ── Drop rows that still have NaN (warm-up period) ────────────────────
        features = raw.dropna()

        if features.empty:
            logger.warning(
                "build_features: all rows are NaN after standardisation. "
                "Supply more historical data (min ~%d bars recommended).",
                self.zscore_window + 200,
            )

        # Reorder columns to match FEATURE_NAMES
        features = features[FEATURE_NAMES]

        logger.debug(
            "build_features: %d bars in → %d valid feature rows",
            len(ohlcv),
            len(features),
        )
        return features

    # ─────────────────────────────────────────────────────────────────────────
    # Individual feature computations  (all causal, tested independently)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _log_return(close: pd.Series, period: int) -> pd.Series:
        """Log return over ``period`` bars: ln(close_t / close_{t-period})."""
        return np.log(close / close.shift(period))

    @staticmethod
    def _realized_vol(log_ret: pd.Series, window: int = 20) -> pd.Series:
        """
        Rolling standard deviation of log returns, annualised by √252.

        Strictly causal: each value uses bars t-window … t.
        """
        return log_ret.rolling(window, min_periods=window // 2).std() * np.sqrt(252)

    @staticmethod
    def _vol_ratio(log_ret: pd.Series, short: int = 5, long: int = 20) -> pd.Series:
        """Short-window / long-window realised vol ratio."""
        short_vol = log_ret.rolling(short, min_periods=short // 2).std()
        long_vol  = log_ret.rolling(long,  min_periods=long  // 2).std()
        return short_vol / (long_vol + 1e-10)

    @staticmethod
    def _volume_zscore(volume: pd.Series, window: int = 50) -> pd.Series:
        """Z-score of volume relative to its rolling mean and std."""
        roll = volume.rolling(window, min_periods=window // 2)
        return (volume - roll.mean()) / (roll.std() + 1e-10)

    @staticmethod
    def _volume_trend(volume: pd.Series, sma_window: int = 10) -> pd.Series:
        """
        Percentage change of the 10-period volume SMA — measures whether
        volume is expanding or contracting.
        """
        sma = volume.rolling(sma_window, min_periods=sma_window // 2).mean()
        return sma.pct_change().fillna(0.0)

    @staticmethod
    def _adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
    ) -> pd.Series:
        """Average Directional Index — measures trend strength (0–100 scale)."""
        return ADXIndicator(
            high=high, low=low, close=close, window=window, fillna=False
        ).adx()

    @staticmethod
    def _sma_slope(close: pd.Series, window: int = 50) -> pd.Series:
        """
        Percentage change of the SMA — positive = uptrend, negative = downtrend.
        """
        sma = SMAIndicator(close=close, window=window, fillna=False).sma_indicator()
        return sma.pct_change().fillna(0.0)

    @staticmethod
    def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index (0–100 scale)."""
        return RSIIndicator(close=close, window=window, fillna=False).rsi()

    @staticmethod
    def _dist_from_sma(close: pd.Series, window: int = 200) -> pd.Series:
        """
        Normalised distance from the long-term SMA:
        (close − SMA) / close.

        Positive = above SMA (bullish), negative = below (bearish).
        """
        sma = SMAIndicator(close=close, window=window, fillna=False).sma_indicator()
        return (close - sma) / close

    @staticmethod
    def _roc(close: pd.Series, period: int = 10) -> pd.Series:
        """Rate of change: (close_t / close_{t-period}) − 1."""
        return ROCIndicator(close=close, window=period, fillna=False).roc() / 100.0

    @staticmethod
    def _norm_atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
    ) -> pd.Series:
        """
        ATR(14) divided by close price — normalised range / volatility proxy.
        """
        atr = AverageTrueRange(
            high=high, low=low, close=close, window=window, fillna=False
        ).average_true_range()
        return atr / close

    @staticmethod
    def _rolling_skew(log_ret: pd.Series, window: int = 14) -> pd.Series:
        """
        Rolling skewness of log returns.

        Negative in crash regimes (left-tail dominance); near-zero in calm bull.
        min_periods = window // 2 for an earlier valid start; sparse early values
        are removed when the full feature matrix is dropna()'d.
        """
        return log_ret.rolling(window, min_periods=window // 2).skew()

    @staticmethod
    def _rolling_kurt(log_ret: pd.Series, window: int = 14) -> pd.Series:
        """
        Rolling excess kurtosis of log returns.

        High in panic / crash regimes (fat tails); near-zero in normal markets.
        pandas rolling().kurt() returns Fisher's excess kurtosis (normal = 0).
        """
        return log_ret.rolling(window, min_periods=window // 2).kurt()

    @staticmethod
    def _autocorr_lag1(log_ret: pd.Series, window: int = 20) -> pd.Series:
        """
        Rolling lag-1 autocorrelation of log returns over ``window`` bars.

        Positive → momentum (trending low-vol bull).
        Negative → mean-reversion (high-vol bounce / mean-reverting regime).
        """
        def _acf1(x: np.ndarray) -> float:
            if len(x) < 3:
                return float("nan")
            # Pearson correlation between x[:-1] and x[1:]
            x0, x1 = x[:-1], x[1:]
            mu0, mu1 = x0.mean(), x1.mean()
            num = ((x0 - mu0) * (x1 - mu1)).sum()
            denom = np.sqrt(((x0 - mu0) ** 2).sum() * ((x1 - mu1) ** 2).sum())
            return float(num / denom) if denom > 1e-10 else 0.0

        return log_ret.rolling(window, min_periods=window // 2).apply(_acf1, raw=True)

    @staticmethod
    def _ret_quantile(log_ret: pd.Series, window: int = 20, quantile: float = 0.05) -> pd.Series:
        """
        Rolling 5th-percentile of log returns (left-tail risk indicator).

        More negative values signal heavier left tails (crash / bear regimes).
        """
        return log_ret.rolling(window, min_periods=window // 2).quantile(quantile)

    def _rolling_zscore(self, series: pd.Series, window: Optional[int] = None) -> pd.Series:
        """
        Rolling z-score normalisation: (x − μ_rolling) / σ_rolling.

        Uses only past data — causal by definition of rolling windows.

        Parameters
        ----------
        series : pd.Series
            Input series.
        window : int, optional
            Lookback window.  Defaults to ``self.zscore_window``.

        Returns
        -------
        pd.Series with NaN where insufficient history exists.
        """
        w = window or self.zscore_window
        roll = series.rolling(w, min_periods=self._min_periods)
        return (series - roll.mean()) / (roll.std() + 1e-10)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Lowercase column names and verify required columns are present."""
        ohlcv = ohlcv.copy()
        ohlcv.columns = [c.lower() for c in ohlcv.columns]
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(ohlcv.columns)
        if missing:
            raise ValueError(f"OHLCV DataFrame missing columns: {missing}")
        return ohlcv
