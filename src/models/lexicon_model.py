"""
Lexicon-based Sentiment Model using VADER.

VADER (Valence Aware Dictionary and sEntiment Reasoner) is specifically
attuned to sentiments expressed in social media and works out-of-the-box
with no training required.

References:
    Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based
    Model for Sentiment Analysis of Social Media Text. ICWSM.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .base_model import BaseSentimentModel, SentimentResult

logger = logging.getLogger(__name__)


class LexiconSentimentModel(BaseSentimentModel):
    """
    Rule-based sentiment analysis using VADER.

    Particularly effective for:
    - Social media posts (tweets, comments)
    - Short texts with emoticons and slang
    - Documents without requiring training data

    Sentiment Labels:
        - "positive"  → compound score ≥ threshold_positive  (default 0.05)
        - "negative"  → compound score ≤ threshold_negative  (default -0.05)
        - "neutral"   → everything in between

    Example::

        model = LexiconSentimentModel()
        result = model.predict("I absolutely love this product! 😍")
        print(result.label)   # → 'positive'
        print(result.score)   # → 0.9458  (confidence)
    """

    MODEL_NAME = "vader"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        threshold_positive: float = 0.05,
        threshold_negative: float = -0.05,
    ):
        super().__init__(model_name=self.MODEL_NAME, config=config)
        self.threshold_positive = threshold_positive
        self.threshold_negative = threshold_negative
        self._analyzer: Optional[SentimentIntensityAnalyzer] = None
        self.is_trained = True  # VADER is pre-trained
        self._load_analyzer()

    # ── Initialization ────────────────────────────────────

    def _load_analyzer(self) -> None:
        """Initialize the VADER analyzer (downloads lexicon if needed)."""
        try:
            self._analyzer = SentimentIntensityAnalyzer()
            self._logger.info("VADER analyzer initialized successfully.")
        except Exception as exc:
            self._logger.error("Failed to initialize VADER: %s", exc)
            raise

    # ── Prediction ────────────────────────────────────────

    def predict(self, text: str) -> SentimentResult:
        """
        Analyse sentiment for a single text.

        Args:
            text: The input text to analyse.

        Returns:
            SentimentResult with label, compound score, and per-polarity scores.
        """
        if not text or not text.strip():
            return SentimentResult(
                text=text,
                label="neutral",
                score=0.0,
                scores={"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                model_name=self.MODEL_NAME,
            )

        raw_scores = self._analyzer.polarity_scores(text)
        compound = raw_scores["compound"]

        label = self._compound_to_label(compound)

        # Map compound [-1, 1] to a confidence score [0, 1]
        if label == "positive":
            score = (compound + 1) / 2
        elif label == "negative":
            score = (abs(compound) + 1) / 2
        else:
            score = 1.0 - abs(compound)

        return SentimentResult(
            text=text,
            label=label,
            score=round(score, 4),
            scores={
                "positive": round(raw_scores["pos"], 4),
                "negative": round(raw_scores["neg"], 4),
                "neutral": round(raw_scores["neu"], 4),
            },
            model_name=self.MODEL_NAME,
            metadata={"compound": round(compound, 4)},
        )

    def predict_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyse sentiment for a list of texts."""
        return [self.predict(text) for text in texts]

    # ── Helpers ───────────────────────────────────────────

    def _compound_to_label(self, compound: float) -> str:
        if compound >= self.threshold_positive:
            return "positive"
        if compound <= self.threshold_negative:
            return "negative"
        return "neutral"

    def get_raw_scores(self, text: str) -> Dict[str, float]:
        """Return VADER's raw polarity scores (pos, neg, neu, compound)."""
        return self._analyzer.polarity_scores(text)
