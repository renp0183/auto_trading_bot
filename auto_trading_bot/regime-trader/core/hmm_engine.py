"""
hmm_engine.py — Hidden Markov Model regime detection engine.

DESIGN PHILOSOPHY
-----------------
The HMM is a VOLATILITY CLASSIFIER.  It detects whether the market is in a
calm, moderate, or turbulent volatility environment.  It does NOT predict price
direction.  Regime labels are assigned by sorting states on mean log-return for
human readability — but the STRATEGY layer makes its own volatility-sorted
allocation decisions independently.

CRITICAL: NO LOOK-AHEAD BIAS
-----------------------------
This engine uses ONLY the forward (filtering) algorithm.  ``model.predict()``
(Viterbi) and ``model.predict_proba()`` (forward-backward / smoothing) both
revise past states using future observations — that is look-ahead bias.

The forward algorithm implemented here computes:

    P(state_t | obs_{1:t})

using ONLY observations up to time t.  The state at bar t is identical
whether we supply data[0:T] or data[0:T+N].  This is verified by the
mandatory test in tests/test_look_ahead.py.
"""

from __future__ import annotations

import logging
import pickle
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

try:
    from hmmlearn import hmm as _hmmlearn_hmm
    _HAS_HMMLEARN = True
except ImportError:
    _hmmlearn_hmm = None  # type: ignore[assignment]
    _HAS_HMMLEARN = False

from data.feature_engineering import FEATURE_NAMES

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Regime label tables (sorted ascending by expected return)
# ─────────────────────────────────────────────────────────────────────────────

#: Maps n_states → ordered list of label strings (lowest return first)
LABEL_MAPS: dict[int, list[str]] = {
    3: ["BEAR", "NEUTRAL", "BULL"],
    4: ["CRASH", "BEAR", "BULL", "EUPHORIA"],
    5: ["CRASH", "BEAR", "NEUTRAL", "BULL", "EUPHORIA"],
    6: ["CRASH", "STRONG_BEAR", "WEAK_BEAR", "WEAK_BULL", "STRONG_BULL", "EUPHORIA"],
    7: ["CRASH", "STRONG_BEAR", "WEAK_BEAR", "NEUTRAL", "WEAK_BULL", "STRONG_BULL", "EUPHORIA"],
}

#: Strategy guidance per label — (type, max_leverage, max_position_pct, min_confidence)
_STRATEGY_PARAMS: dict[str, tuple[str, float, float, float]] = {
    "CRASH":         ("DEFENSIVE",         0.00, 0.05, 0.70),
    "STRONG_BEAR":   ("DEFENSIVE",         0.00, 0.08, 0.65),
    "BEAR":          ("DEFENSIVE",         0.50, 0.10, 0.60),
    "WEAK_BEAR":     ("CAUTIOUS",          0.75, 0.12, 0.58),
    "NEUTRAL":       ("BALANCED",          1.00, 0.15, 0.55),
    "WEAK_BULL":     ("GROWTH",            1.00, 0.15, 0.55),
    "BULL":          ("GROWTH",            1.25, 0.15, 0.50),
    "STRONG_BULL":   ("AGGRESSIVE_GROWTH", 1.25, 0.15, 0.50),
    "EUPHORIA":      ("CAUTIOUS_GROWTH",   1.00, 0.10, 0.60),
    "UNKNOWN":       ("NEUTRAL",           0.50, 0.08, 0.70),
}

# ─────────────────────────────────────────────────────────────────────────────
# Public enums and dataclasses
# ─────────────────────────────────────────────────────────────────────────────


class RegimeLabel(str, Enum):
    """All possible regime labels across n_states configurations."""

    CRASH        = "CRASH"
    STRONG_BEAR  = "STRONG_BEAR"
    BEAR         = "BEAR"
    WEAK_BEAR    = "WEAK_BEAR"
    NEUTRAL      = "NEUTRAL"
    WEAK_BULL    = "WEAK_BULL"
    BULL         = "BULL"
    STRONG_BULL  = "STRONG_BULL"
    EUPHORIA     = "EUPHORIA"
    UNKNOWN      = "UNKNOWN"


# Backward-compatible alias used in Phase-1 skeleton files
Regime = RegimeLabel


@dataclass
class RegimeInfo:
    """
    Per-regime metadata derived from the fitted HMM.

    ``expected_return`` and ``expected_volatility`` are expressed in z-score
    units (relative to the training sample) — higher values indicate higher
    return / volatility relative to other regimes in the model.
    """

    regime_id:               int
    regime_name:             str
    expected_return:         float   # Mean log-return feature value (z-score units)
    expected_volatility:     float   # Mean realised-vol feature value (z-score units)
    recommended_strategy_type: str
    max_leverage_allowed:    float
    max_position_size_pct:   float
    min_confidence_to_act:   float


@dataclass
class RegimeState:
    """
    Complete regime state snapshot at a single point in time.

    ``label`` and ``state_id`` always reflect the CONFIRMED regime (i.e. the
    regime that has been stable for at least ``stability_bars`` bars).  When a
    new candidate is building up but not yet confirmed, ``in_transition=True``
    and ``candidate_label`` carries the pending label.  The strategy layer
    should reduce size by 25 % whenever ``in_transition`` is True.
    """

    label:              str               # Confirmed regime label string
    state_id:           int               # HMM state index of confirmed regime
    probability:        float             # Posterior P(confirmed state | obs_{1:t})
    state_probabilities: np.ndarray       # Full (n_states,) posterior distribution
    timestamp:          Optional[pd.Timestamp]
    is_confirmed:       bool              # True once candidate held ≥ stability_bars
    consecutive_bars:   int               # Bars in current confirmed regime
    in_transition:      bool              # True when candidate ≠ confirmed
    candidate_label:    Optional[str]     # Label of pending candidate (if transitioning)
    flicker_rate:       float             # Regime changes per flicker_window bars
    regime_info:        Optional[RegimeInfo] = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RegimeState):
            return NotImplemented
        return self.label == other.label and self.state_id == other.state_id

    def __hash__(self) -> int:
        return hash((self.label, self.state_id))

    def __repr__(self) -> str:
        return (
            f"RegimeState(label={self.label!r}, p={self.probability:.3f}, "
            f"confirmed={self.is_confirmed}, bars={self.consecutive_bars}, "
            f"flickering={self.flicker_rate:.2f})"
        )


@dataclass
class HMMConfig:
    """
    Configuration mirroring the ``[hmm]`` section of ``settings.yaml``.

    All fields map 1-to-1 to the YAML keys so the caller can do::

        cfg = HMMConfig(**settings["hmm"])
    """

    n_candidates:      list[int] = field(default_factory=lambda: [3, 4, 5, 6, 7])
    n_init:            int   = 10
    covariance_type:   str   = "full"
    min_train_bars:    int   = 252
    stability_bars:    int   = 3
    flicker_window:    int   = 20
    flicker_threshold: int   = 4
    min_confidence:    float = 0.55


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper dataclass for stability tracking
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _StabilityState:
    confirmed_id:   int   # last confirmed HMM state index
    candidate_id:   int   # pending (not-yet-confirmed) state index
    candidate_run:  int   # consecutive bars in candidate
    confirmed_bars: int   # total bars since confirmed was last set


def _step_stability(
    s: _StabilityState,
    raw_id: int,
    stability_bars: int,
) -> _StabilityState:
    """
    Advance stability state by one bar.

    A new regime is **confirmed** only after ``stability_bars`` consecutive
    observations in the same raw state.  Until then, the previous confirmed
    state is preserved to avoid false transitions.
    """
    if raw_id == s.candidate_id:
        new_cand_run = s.candidate_run + 1
    else:
        new_cand_run = 1

    new_cand_id = raw_id

    # Confirm the candidate if it has been stable long enough
    if new_cand_run >= stability_bars:
        new_conf_id = new_cand_id
    else:
        new_conf_id = s.confirmed_id

    # Reset or increment confirmed-bars counter
    if new_conf_id != s.confirmed_id:
        new_conf_bars = new_cand_run  # carry over the run length
    else:
        new_conf_bars = s.confirmed_bars + 1

    return _StabilityState(
        confirmed_id=new_conf_id,
        candidate_id=new_cand_id,
        candidate_run=new_cand_run,
        confirmed_bars=new_conf_bars,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main engine
# ─────────────────────────────────────────────────────────────────────────────


class HMMEngine:
    """
    Gaussian HMM regime detection engine.

    Model selection
    ~~~~~~~~~~~~~~~
    Trains one GaussianHMM per candidate state count with ``n_init`` random
    restarts.  Selects the model with the lowest BIC score.

    Regime labelling
    ~~~~~~~~~~~~~~~~
    States are sorted by their mean log-return feature value (ascending), then
    labelled with human-readable names (CRASH … EUPHORIA) based on n_states.
    Labels are for human readability only — the strategy layer independently
    sorts by volatility for allocation decisions.

    Causal inference
    ~~~~~~~~~~~~~~~~
    ``predict_regime_filtered()`` runs the **forward algorithm** only::

        alpha_t = P(state_t | obs_{1:t})

    This uses only past and present data.  ``model.predict()`` (Viterbi) and
    ``model.predict_proba()`` (forward-backward) are intentionally NOT used
    for inference — they introduce look-ahead bias.

    Usage
    ~~~~~
    ::

        engine = HMMEngine(HMMConfig())
        engine.fit(feature_df)
        states = engine.predict_regime_filtered(feature_df)
        current = states[-1]
        print(current.label, current.probability)
    """

    def __init__(self, config: HMMConfig) -> None:
        self.config = config

        # Fitted model artefacts
        self._model:         Optional[Any] = None
        self._n_states:      int   = 0
        self._bic:           float = float("inf")
        self._bic_scores:    dict[int, float] = {}
        self._training_date: Optional[datetime] = None
        self._feature_cols:  list[str] = []

        # Label and metadata maps
        self._state_to_label: dict[int, str]      = {}   # state_idx → label string
        self._regime_infos:   dict[str, RegimeInfo] = {}

        # Incremental (live) forward algorithm state
        self._last_log_alpha:  Optional[np.ndarray] = None   # cached log-alpha vector
        self._live_stability:  Optional[_StabilityState] = None
        self._live_history:    deque = deque(maxlen=config.flicker_window)

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────

    def fit(self, features: pd.DataFrame) -> "HMMEngine":
        """
        Fit a Gaussian HMM to the supplied feature matrix.

        Iterates over ``config.n_candidates``, trains ``config.n_init`` random
        initialisations for each candidate, selects the lowest-BIC model, then
        labels states by mean log-return rank.

        Parameters
        ----------
        features : DataFrame
            Shape (n_bars, n_features).  Must have ≥ ``config.min_train_bars`` rows.
            Columns should match ``FEATURE_NAMES`` from feature_engineering.

        Returns
        -------
        self — for method chaining.

        Raises
        ------
        ValueError
            If fewer than ``config.min_train_bars`` valid rows are supplied.
        RuntimeError
            If all candidate models fail to converge.
        """
        if not _HAS_HMMLEARN:
            raise RuntimeError(
                "hmmlearn is required for HMMEngine.fit(). "
                "Install it: pip install hmmlearn  (requires Python ≤ 3.12 or C++ build tools)"
            )
        if len(features) < self.config.min_train_bars:
            raise ValueError(
                f"HMMEngine.fit requires ≥ {self.config.min_train_bars} bars, "
                f"got {len(features)}."
            )

        self._feature_cols = list(features.columns)
        X = features.values.astype(float)

        # ── Select best model by BIC ──────────────────────────────────────────
        best_model, best_n, bic_scores = self._select_best_model(X)
        self._model        = best_model
        self._n_states     = best_n
        self._bic          = bic_scores[best_n]
        self._bic_scores   = bic_scores
        self._training_date = datetime.utcnow()

        logger.info(
            "HMMEngine.fit: selected n_states=%d  BIC=%.2f  (all: %s)",
            best_n,
            self._bic,
            {k: f"{v:.1f}" for k, v in bic_scores.items()},
        )

        # ── Label states by return rank ───────────────────────────────────────
        self._state_to_label = self._label_states_by_return(self._model)
        self._regime_infos   = self._build_regime_infos(self._model)

        logger.info(
            "HMMEngine.fit: state labels = %s",
            {k: v for k, v in sorted(self._state_to_label.items())},
        )

        # ── Reset live inference state ────────────────────────────────────────
        self._last_log_alpha = None
        self._live_stability = None
        self._live_history.clear()

        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Causal batch inference  (primary method — use for backtesting & testing)
    # ─────────────────────────────────────────────────────────────────────────

    def predict_regime_filtered(self, features: pd.DataFrame) -> list[RegimeState]:
        """
        Run the forward (filtering) algorithm on a full feature sequence.

        **This is the only correct method for backtesting and the no-look-ahead
        bias test.**  At bar ``t`` the result uses only ``obs_{1:t}``, regardless
        of how many additional bars follow in ``features``.

        Parameters
        ----------
        features : DataFrame
            Feature matrix (same columns as used during ``fit()``).

        Returns
        -------
        list[RegimeState]
            One entry per row in ``features``.  ``states[t].label`` reflects
            the confirmed regime at bar ``t`` using only data up to bar ``t``.
        """
        self._assert_fitted()
        X, timestamps = self._prepare_input(features)

        log_emissions = self._compute_log_emissions(X)         # (T, n_states)
        alphas        = self._forward_pass(log_emissions)       # (T, n_states)

        return self._build_regime_states_batch(alphas, timestamps)

    def predict_regime_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Return the filtered posterior probability distribution P(state_t | obs_{1:t})
        as a DataFrame.

        Parameters
        ----------
        features : DataFrame

        Returns
        -------
        DataFrame
            Shape (len(features), n_states).  Columns are state indices 0 … n_states−1.
            Index matches ``features.index``.
        """
        self._assert_fitted()
        X, timestamps = self._prepare_input(features)
        log_emissions = self._compute_log_emissions(X)
        alphas        = self._forward_pass(log_emissions)
        cols = [self._state_to_label.get(k, str(k)) for k in range(self._n_states)]
        return pd.DataFrame(alphas, index=features.index, columns=cols)

    # ─────────────────────────────────────────────────────────────────────────
    # Incremental (live-trading) inference
    # ─────────────────────────────────────────────────────────────────────────

    def predict_filtered_next(
        self,
        new_obs: np.ndarray,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> RegimeState:
        """
        Advance the forward algorithm by one observation (live trading).

        Caches the previous alpha vector so the full history need not be
        reprocessed on each bar.  Call ``reset_live_state()`` to restart.

        Parameters
        ----------
        new_obs : array-like, shape (n_features,)
            Feature vector for the new bar.
        timestamp : pd.Timestamp, optional

        Returns
        -------
        RegimeState for the current bar.
        """
        self._assert_fitted()
        obs = np.asarray(new_obs, dtype=float).reshape(1, -1)

        log_em = self._compute_log_emissions(obs)[0]  # (n_states,)

        if self._last_log_alpha is None:
            # Initialise with start probabilities
            log_alpha = np.log(self._model.startprob_ + 1e-300) + log_em
        else:
            log_transmat = np.log(self._model.transmat_ + 1e-300)
            log_alpha_pred = logsumexp(
                self._last_log_alpha[:, None] + log_transmat, axis=0
            )
            log_alpha = log_alpha_pred + log_em

        log_alpha -= logsumexp(log_alpha)          # normalise
        self._last_log_alpha = log_alpha

        probs = np.exp(log_alpha)
        raw_state_id = int(np.argmax(probs))

        # ── Stability update ──────────────────────────────────────────────────
        if self._live_stability is None:
            self._live_stability = _StabilityState(
                confirmed_id=raw_state_id,
                candidate_id=raw_state_id,
                candidate_run=1,
                confirmed_bars=1,
            )
        else:
            self._live_stability = _step_stability(
                self._live_stability, raw_state_id, self.config.stability_bars
            )

        self._live_history.append(self._live_stability.confirmed_id)

        return self._make_regime_state(
            probs=probs,
            stab=self._live_stability,
            history=self._live_history,
            timestamp=timestamp,
        )

    def reset_live_state(self) -> None:
        """Clear cached alpha and stability state (e.g. after a model retrain)."""
        self._last_log_alpha = None
        self._live_stability = None
        self._live_history.clear()

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience wrappers  (backward-compat with Phase-1 stubs)
    # ─────────────────────────────────────────────────────────────────────────

    def predict(self, features: pd.DataFrame) -> list[RegimeState]:
        """Alias for ``predict_regime_filtered()``."""
        return self.predict_regime_filtered(features)

    def predict_current(self, features: pd.DataFrame) -> RegimeState:
        """Return only the last RegimeState in the sequence."""
        return self.predict_regime_filtered(features)[-1]

    # ─────────────────────────────────────────────────────────────────────────
    # Inspection helpers
    # ─────────────────────────────────────────────────────────────────────────

    def is_fitted(self) -> bool:
        """Return True if a model has been successfully fitted."""
        return self._model is not None

    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Return the learned transition probability matrix as a labelled DataFrame.

        Rows = previous state, columns = next state.
        """
        self._assert_fitted()
        labels = [self._state_to_label.get(k, str(k)) for k in range(self._n_states)]
        return pd.DataFrame(
            self._model.transmat_,
            index=labels,
            columns=labels,
        )

    def get_regime_stability(
        self, states: list[RegimeState]
    ) -> int:
        """
        Return the number of consecutive bars the *last* regime in ``states``
        has been confirmed.
        """
        if not states:
            return 0
        return states[-1].consecutive_bars

    def detect_regime_change(
        self, states: list[RegimeState], lookback: int = 1
    ) -> bool:
        """
        Return True if a regime change was confirmed in the last ``lookback`` bars.
        """
        if len(states) < lookback + 1:
            return False
        recent = states[-lookback - 1:]
        return any(
            recent[i].state_id != recent[i - 1].state_id
            and recent[i].is_confirmed
            for i in range(1, len(recent))
        )

    def get_regime_flicker_rate(self, states: list[RegimeState]) -> float:
        """Return the flicker rate of the most recent state."""
        if not states:
            return 0.0
        return states[-1].flicker_rate

    def is_flickering(self, states: list[RegimeState]) -> bool:
        """Return True if the current flicker rate exceeds ``config.flicker_threshold``."""
        if not states:
            return False
        last = states[-1]
        # flicker_rate is changes/window; threshold comparison on raw count
        changes_in_window = last.flicker_rate * self.config.flicker_window
        return changes_in_window > self.config.flicker_threshold

    def get_regime_info(self, label: str) -> Optional[RegimeInfo]:
        """Return the RegimeInfo for a label string, or None."""
        return self._regime_infos.get(label)

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """
        Serialise the engine to a pickle file.

        Saves the fitted model plus metadata (n_states, BIC, training date,
        feature columns, labels, regime infos).
        """
        self._assert_fitted()
        payload = {
            "model":          self._model,
            "n_states":       self._n_states,
            "bic":            self._bic,
            "bic_scores":     self._bic_scores,
            "training_date":  self._training_date,
            "feature_cols":   self._feature_cols,
            "state_to_label": self._state_to_label,
            "regime_infos":   self._regime_infos,
            "config":         self.config,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("HMMEngine saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "HMMEngine":
        """
        Deserialise an engine previously saved with ``save()``.

        Returns a fully initialised HMMEngine ready for inference.
        """
        with open(path, "rb") as fh:
            payload = pickle.load(fh)

        engine = cls(payload["config"])
        engine._model          = payload["model"]
        engine._n_states       = payload["n_states"]
        engine._bic            = payload["bic"]
        engine._bic_scores     = payload.get("bic_scores", {})
        engine._training_date  = payload["training_date"]
        engine._feature_cols   = payload["feature_cols"]
        engine._state_to_label = payload["state_to_label"]
        engine._regime_infos   = payload["regime_infos"]

        logger.info(
            "HMMEngine loaded from %s  (n_states=%d, trained=%s)",
            path,
            engine._n_states,
            engine._training_date,
        )
        return engine

    # ─────────────────────────────────────────────────────────────────────────
    # Private: model selection
    # ─────────────────────────────────────────────────────────────────────────

    def _select_best_model(
        self, X: np.ndarray
    ) -> tuple[Any, int, dict[int, float]]:
        """
        Train one GaussianHMM per candidate state count with ``n_init`` random
        restarts each.  Return the (model, n_states, bic_dict) triple where
        n_states minimises BIC.
        """
        best_model: Optional[Any] = None
        best_n     = self.config.n_candidates[0]
        best_bic   = float("inf")
        bic_scores: dict[int, float] = {}

        for n in self.config.n_candidates:
            candidate_model, candidate_bic = self._fit_with_restarts(X, n)
            if candidate_model is None:
                logger.warning("All %d restarts failed for n_states=%d", self.config.n_init, n)
                continue

            bic_scores[n] = candidate_bic
            logger.debug("  n_states=%d  BIC=%.2f", n, candidate_bic)

            if candidate_bic < best_bic:
                best_bic   = candidate_bic
                best_n     = n
                best_model = candidate_model

        if best_model is None:
            raise RuntimeError(
                "HMMEngine: all candidate models failed to converge. "
                "Check your feature matrix for NaN / constant columns."
            )

        return best_model, best_n, bic_scores

    def _fit_with_restarts(
        self, X: np.ndarray, n_states: int
    ) -> tuple[Optional[Any], float]:
        """
        Fit ``n_init`` GaussianHMMs with random seeds and return the one with
        the best (highest) log-likelihood, together with its BIC.
        """
        best_model:  Optional[Any] = None
        best_ll:     float = -float("inf")

        for seed in range(self.config.n_init):
            try:
                model = _hmmlearn_hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type=self.config.covariance_type,
                    n_iter=100,
                    tol=1e-4,
                    random_state=seed,
                    verbose=False,
                )
                model.fit(X)
                ll = model.score(X) * len(X)
                if ll > best_ll:
                    best_ll    = ll
                    best_model = model
            except Exception as exc:
                logger.debug("HMM fit failed n_states=%d seed=%d: %s", n_states, seed, exc)

        if best_model is None:
            return None, float("inf")

        bic = self._compute_bic(best_model, X)
        return best_model, bic

    def _compute_bic(
        self, model: Any, X: np.ndarray
    ) -> float:
        """
        Bayesian Information Criterion:

            BIC = −2 · ℓ + n_params · ln(n_samples)

        where ℓ = total log-likelihood (model.score(X) * len(X)).

        Free parameters for covariance_type="full":
          - Initial distribution: n_states − 1
          - Transition matrix:    n_states · (n_states − 1)
          - Means:                n_states · n_features
          - Full covariances:     n_states · n_features · (n_features + 1) / 2
        """
        n, d   = X.shape
        K      = model.n_components
        n_params = (
            (K - 1)
            + K * (K - 1)
            + K * d
            + K * d * (d + 1) // 2
        )
        log_likelihood = model.score(X) * n
        return -2.0 * log_likelihood + n_params * np.log(n)

    # ─────────────────────────────────────────────────────────────────────────
    # Private: state labelling
    # ─────────────────────────────────────────────────────────────────────────

    def _label_states_by_return(
        self, model: Any
    ) -> dict[int, str]:
        """
        Sort HMM states by mean log-return (ascending) and assign label strings.

        The return feature index is located by searching ``_feature_cols`` for
        ``"log_return_1"``; falls back to column 0.
        """
        n_states = model.n_components

        # Locate the return feature column
        if "log_return_1" in self._feature_cols:
            ret_idx = self._feature_cols.index("log_return_1")
        else:
            ret_idx = 0
            logger.warning(
                "Could not find 'log_return_1' in feature columns %s; "
                "using column 0 for regime labelling.",
                self._feature_cols,
            )

        mean_returns = model.means_[:, ret_idx]           # (n_states,)
        sorted_state_idxs = list(np.argsort(mean_returns))  # ascending

        # Use label map for the fitted n_states; fall back to 5-state labels
        label_names = LABEL_MAPS.get(n_states, LABEL_MAPS[5])
        # Pad/trim if n_states is outside [3,7]
        while len(label_names) < n_states:
            label_names = ["UNKNOWN"] + label_names
        label_names = label_names[:n_states]

        return {
            state_idx: label_names[rank]
            for rank, state_idx in enumerate(sorted_state_idxs)
        }

    def _build_regime_infos(
        self, model: Any
    ) -> dict[str, RegimeInfo]:
        """
        Construct a ``RegimeInfo`` for every label using mean feature values
        from the fitted model and the static ``_STRATEGY_PARAMS`` table.
        """
        # Locate vol feature
        vol_idx = (
            self._feature_cols.index("realized_vol_20")
            if "realized_vol_20" in self._feature_cols
            else (
                self._feature_cols.index("norm_atr_14")
                if "norm_atr_14" in self._feature_cols
                else 0
            )
        )
        ret_idx = (
            self._feature_cols.index("log_return_1")
            if "log_return_1" in self._feature_cols
            else 0
        )

        infos: dict[str, RegimeInfo] = {}
        for state_id, label in self._state_to_label.items():
            params = _STRATEGY_PARAMS.get(label, _STRATEGY_PARAMS["UNKNOWN"])
            infos[label] = RegimeInfo(
                regime_id=state_id,
                regime_name=label,
                expected_return=float(model.means_[state_id, ret_idx]),
                expected_volatility=float(model.means_[state_id, vol_idx]),
                recommended_strategy_type=params[0],
                max_leverage_allowed=params[1],
                max_position_size_pct=params[2],
                min_confidence_to_act=params[3],
            )
        return infos

    # ─────────────────────────────────────────────────────────────────────────
    # Private: forward algorithm (the causal core)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_log_emissions(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log P(obs_t | state = k) for all t and k.

        Uses ``scipy.stats.multivariate_normal`` with the model's learned
        means and covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (T, n_features)

        Returns
        -------
        ndarray, shape (T, n_states)
        """
        T        = len(X)
        n_states = self._model.n_components
        log_em   = np.full((T, n_states), -1e300, dtype=float)

        for k in range(n_states):
            mean = self._model.means_[k]
            cov  = self._model.covars_[k]   # shape (d, d) for "full"
            try:
                rv = multivariate_normal(mean=mean, cov=cov, allow_singular=True)
                log_em[:, k] = rv.logpdf(X)
            except Exception as exc:
                logger.warning(
                    "_compute_log_emissions: state %d failed (%s); using -inf.", k, exc
                )

        return log_em

    def _forward_pass(self, log_emissions: np.ndarray) -> np.ndarray:
        """
        Causal forward algorithm.

        Computes P(state_t = k | obs_{1:t}) for each t and k, using ONLY
        observations up to time t.

        Algorithm (in log space for numerical stability):

        Initialisation (t = 0):
            log α₀[k] = log π[k] + log p(obs₀ | k)
            log α₀    -= logsumexp(log α₀)        # normalise

        Recursion (t = 1 … T−1):
            log α̂[k] = logsumexp_j( log α_{t-1}[j] + log A[j,k] )
            log αₜ[k] = log α̂[k] + log p(obsₜ | k)
            log αₜ    -= logsumexp(log αₜ)         # normalise

        where A[j,k] = P(sₜ = k | sₜ₋₁ = j) = model.transmat_[j, k].

        Returns
        -------
        ndarray, shape (T, n_states)
            Filtered posterior probabilities (NOT log-space).
        """
        T, n_states  = log_emissions.shape
        log_transmat = np.log(self._model.transmat_ + 1e-300)  # (K, K)
        log_alphas   = np.zeros((T, n_states), dtype=float)

        # ── t = 0 ─────────────────────────────────────────────────────────────
        log_alpha = np.log(self._model.startprob_ + 1e-300) + log_emissions[0]
        log_alpha -= logsumexp(log_alpha)
        log_alphas[0] = log_alpha

        # ── t = 1 … T−1 ───────────────────────────────────────────────────────
        for t in range(1, T):
            # log_alpha[:, None] broadcasts to (K, K); sum over axis 0 (prev states)
            # log_transmat[j, k] = log P(s_t=k | s_{t-1}=j)
            log_alpha_pred = logsumexp(
                log_alpha[:, None] + log_transmat, axis=0
            )  # (K,)
            log_alpha = log_alpha_pred + log_emissions[t]
            log_alpha -= logsumexp(log_alpha)   # normalise
            log_alphas[t] = log_alpha

        return np.exp(log_alphas)   # convert from log-probs to probs

    # ─────────────────────────────────────────────────────────────────────────
    # Private: building RegimeState objects
    # ─────────────────────────────────────────────────────────────────────────

    def _build_regime_states_batch(
        self,
        alphas: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex],
    ) -> list[RegimeState]:
        """
        Convert a (T, n_states) alpha matrix to a list of RegimeStates,
        applying the stability filter and computing flicker rates.
        """
        T       = len(alphas)
        states: list[RegimeState] = []

        # Initialise stability tracking
        raw_id0 = int(np.argmax(alphas[0]))
        stab = _StabilityState(
            confirmed_id=raw_id0,
            candidate_id=raw_id0,
            candidate_run=1,
            confirmed_bars=1,
        )
        history: deque = deque(maxlen=self.config.flicker_window)
        history.append(raw_id0)

        for t in range(T):
            raw_id = int(np.argmax(alphas[t]))

            if t > 0:
                stab = _step_stability(stab, raw_id, self.config.stability_bars)
                history.append(stab.confirmed_id)
            # t == 0 already initialised above

            # ── Log regime changes ────────────────────────────────────────────
            if t > 0 and stab.confirmed_id != states[-1].state_id:
                old_label = states[-1].label
                new_label = self._state_to_label.get(stab.confirmed_id, "UNKNOWN")
                logger.warning(
                    "Regime change confirmed at bar %d: %s → %s",
                    t, old_label, new_label,
                )
            elif t > 0 and stab.candidate_run == self.config.stability_bars:
                logger.info(
                    "Regime confirmed at bar %d: %s (p=%.3f)",
                    t,
                    self._state_to_label.get(stab.confirmed_id, "UNKNOWN"),
                    float(alphas[t, stab.confirmed_id]),
                )

            ts = timestamps[t] if timestamps is not None and t < len(timestamps) else None
            regime_state = self._make_regime_state(alphas[t], stab, history, ts)
            states.append(regime_state)

        return states

    def _make_regime_state(
        self,
        probs: np.ndarray,
        stab: _StabilityState,
        history: deque,
        timestamp: Optional[pd.Timestamp],
    ) -> RegimeState:
        """Construct a RegimeState from probabilities and stability tracking data."""
        label          = self._state_to_label.get(stab.confirmed_id, "UNKNOWN")
        in_transition  = stab.candidate_id != stab.confirmed_id
        cand_label     = (
            self._state_to_label.get(stab.candidate_id, "UNKNOWN")
            if in_transition
            else None
        )

        # Flicker rate: fraction of transitions in the recent history window
        hist_list = list(history)
        if len(hist_list) > 1:
            changes      = sum(1 for i in range(1, len(hist_list)) if hist_list[i] != hist_list[i - 1])
            flicker_rate = changes / len(hist_list)
        else:
            flicker_rate = 0.0

        confirmed_prob = float(probs[stab.confirmed_id]) if stab.confirmed_id < len(probs) else 0.0

        return RegimeState(
            label=label,
            state_id=stab.confirmed_id,
            probability=confirmed_prob,
            state_probabilities=probs.copy(),
            timestamp=timestamp,
            is_confirmed=(stab.candidate_run >= self.config.stability_bars),
            consecutive_bars=stab.confirmed_bars,
            in_transition=in_transition,
            candidate_label=cand_label,
            flicker_rate=flicker_rate,
            regime_info=self._regime_infos.get(label),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Private: utilities
    # ─────────────────────────────────────────────────────────────────────────

    def _assert_fitted(self) -> None:
        if not self.is_fitted():
            raise RuntimeError(
                "HMMEngine has not been fitted.  Call fit() before inference."
            )

    def _prepare_input(
        self, features: pd.DataFrame
    ) -> tuple[np.ndarray, Optional[pd.DatetimeIndex]]:
        """Convert DataFrame to float ndarray; return (X, timestamps)."""
        X = features.values.astype(float)
        ts = features.index if isinstance(features.index, pd.DatetimeIndex) else None
        return X, ts
