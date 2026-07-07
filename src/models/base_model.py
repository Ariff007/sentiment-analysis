"""
Abstract Base Class for Sentiment Analysis Models.

All concrete models must implement the predict() and predict_batch() methods.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────

@dataclass
class SentimentResult:
    """A single sentiment prediction result."""

    text: str
    label: str                          # "positive", "negative", "neutral"
    score: float                        # Confidence score [0.0 – 1.0]
    scores: Dict[str, float] = field(default_factory=dict)   # Per-class probabilities
    model_name: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "score": round(self.score, 4),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "model_name": self.model_name,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"SentimentResult(label={self.label!r}, "
            f"score={self.score:.4f}, model={self.model_name!r})"
        )


# ─────────────────────────────────────────────────────────
# Abstract Base
# ─────────────────────────────────────────────────────────

class BaseSentimentModel(ABC):
    """Abstract base class for sentiment analysis models."""

    def __init__(self, model_name: str = "base", config: Optional[Dict] = None):
        self.model_name = model_name
        self.config = config or {}
        self.is_trained = False
        self._logger = logging.getLogger(self.__class__.__name__)

    # ── Abstract Interface ────────────────────────────────

    @abstractmethod
    def predict(self, text: str) -> SentimentResult:
        """Predict sentiment for a single text string."""
        ...

    @abstractmethod
    def predict_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Predict sentiment for a list of text strings."""
        ...

    # ── Optional Training Interface ───────────────────────

    def train(self, texts: List[str], labels: List[str], **kwargs) -> None:
        """Train the model. Override in trainable subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support training."
        )

    def save(self, path: str) -> None:
        """Persist the model to disk. Override in subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support saving."
        )

    def load(self, path: str) -> None:
        """Load the model from disk. Override in subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support loading."
        )

    # ── Convenience ───────────────────────────────────────

    def analyze(self, text: str) -> SentimentResult:
        """Alias for predict()."""
        return self.predict(text)

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Alias for predict_batch()."""
        return self.predict_batch(texts)

    def get_label(self, text: str) -> str:
        """Return only the sentiment label string."""
        return self.predict(text).label

    def get_score(self, text: str) -> float:
        """Return only the confidence score."""
        return self.predict(text).score

    # ── Internal Helpers ──────────────────────────────────

    @staticmethod
    def _normalize_label(raw_label: str) -> str:
        """Normalize varied label formats to positive/negative/neutral."""
        label = raw_label.lower().strip()
        positive_synonyms = {"pos", "positive", "1", "good", "happy"}
        negative_synonyms = {"neg", "negative", "-1", "bad", "sad"}
        if label in positive_synonyms:
            return "positive"
        if label in negative_synonyms:
            return "negative"
        return "neutral"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self.model_name!r})"
