"""
Training Pipeline for Sentiment Analysis.

Orchestrates the full end-to-end training workflow:
  1. Load & validate data (CSV / TSV / JSON / plain text)
  2. Pre-process text
  3. Train a model (ML or Transformer)
  4. Evaluate on held-out test set
  5. Save the model and metrics

Example::

    trainer = SentimentTrainer(model_type="ml", algorithm="logistic_regression")
    metrics = trainer.train_from_csv("data/sample_data.csv",
                                     text_col="text",
                                     label_col="sentiment")
    trainer.save_model("models/ml_model.joblib")
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..models.lexicon_model import LexiconSentimentModel
from ..models.ml_model import MLSentimentModel
from ..models.transformer_model import TransformerSentimentModel
from ..preprocessing.text_cleaner import TextCleaner

logger = logging.getLogger(__name__)


class SentimentTrainer:
    """
    High-level training orchestrator.

    Args:
        model_type: ``"ml"`` or ``"transformer"`` (``"vader"`` requires no training).
        algorithm: For ML models – one of ``logistic_regression``, ``svm``,
            ``naive_bayes``, ``random_forest``.
        transformer_model: HuggingFace model ID for transformer models.
        cleaner_kwargs: Extra keyword arguments passed to :class:`TextCleaner`.
        config: Optional model config dict.
    """

    SUPPORTED_TYPES = {"ml", "transformer"}

    def __init__(
        self,
        model_type: str = "ml",
        algorithm: str = "logistic_regression",
        transformer_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
        cleaner_kwargs: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        if model_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"model_type must be one of {self.SUPPORTED_TYPES}, got '{model_type}'"
            )

        self.model_type = model_type
        self.algorithm = algorithm
        self.transformer_model_name = transformer_model
        self.config = config or {}
        self.cleaner = TextCleaner(**(cleaner_kwargs or {}))
        self.model = None
        self._metrics: Dict[str, Any] = {}

    # ── Public API ────────────────────────────────────────

    def train_from_csv(
        self,
        path: str,
        text_col: str = "text",
        label_col: str = "sentiment",
        sep: str = ",",
        sample_size: Optional[int] = None,
        **train_kwargs,
    ) -> Dict[str, Any]:
        """
        Load a CSV/TSV file and train the model.

        Expected CSV format::

            text,sentiment
            "I love this!","positive"
            "Terrible service","negative"

        Args:
            path:        Path to the CSV file.
            text_col:    Column name containing raw text.
            label_col:   Column name containing sentiment labels.
            sep:         CSV delimiter (default ``,``).
            sample_size: Optional. Use only the first N rows.

        Returns:
            Dictionary of training metrics.
        """
        logger.info("Loading data from '%s' …", path)
        df = pd.read_csv(path, sep=sep)

        if text_col not in df.columns or label_col not in df.columns:
            raise ValueError(
                f"Expected columns '{text_col}' and '{label_col}', "
                f"found: {list(df.columns)}"
            )

        # Drop rows with missing values
        df = df[[text_col, label_col]].dropna()
        df[label_col] = df[label_col].str.strip().str.lower()

        if sample_size:
            df = df.sample(min(sample_size, len(df)), random_state=42)

        texts = df[text_col].tolist()
        labels = df[label_col].tolist()

        logger.info(
            "Dataset: %d samples | Labels: %s",
            len(texts),
            dict(pd.Series(labels).value_counts()),
        )

        return self.train(texts, labels, **train_kwargs)

    def train_from_json(
        self,
        path: str,
        text_key: str = "text",
        label_key: str = "sentiment",
        **train_kwargs,
    ) -> Dict[str, Any]:
        """Load a JSON file (list of objects) and train the model."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        texts = [item[text_key] for item in data]
        labels = [item[label_key] for item in data]
        return self.train(texts, labels, **train_kwargs)

    def train(
        self,
        texts: List[str],
        labels: List[str],
        preprocess: bool = True,
        **train_kwargs,
    ) -> Dict[str, Any]:
        """
        Train on raw text + label lists.

        Args:
            texts:       Raw text strings.
            labels:      Corresponding sentiment labels.
            preprocess:  Apply TextCleaner before training. Default True.
            **train_kwargs: Forwarded to the underlying model's ``train()``.

        Returns:
            Training metrics dictionary.
        """
        logger.info("Starting training pipeline (model_type=%s)…", self.model_type)
        t0 = time.time()

        if preprocess:
            logger.info("Pre-processing %d texts …", len(texts))
            texts = self.cleaner.clean_batch(texts)

        # Initialise model
        self.model = self._build_model()

        # Train
        self._metrics = self.model.train(texts, labels, **train_kwargs)
        elapsed = time.time() - t0
        self._metrics["training_time_seconds"] = round(elapsed, 2)

        logger.info(
            "Training complete in %.1f s | Metrics: %s", elapsed, self._metrics
        )
        return self._metrics

    def save_model(self, path: str) -> None:
        """Save the trained model to disk."""
        if self.model is None:
            raise RuntimeError("No model trained yet. Call train() first.")
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.model.save(path)
        logger.info("Model saved to '%s'.", path)

    def get_metrics(self) -> Dict[str, Any]:
        """Return the training metrics from the last training run."""
        return self._metrics

    # ── Internal ──────────────────────────────────────────

    def _build_model(self):
        """Instantiate and return the correct model object."""
        if self.model_type == "ml":
            return MLSentimentModel(
                algorithm=self.algorithm,
                config=self.config,
            )
        if self.model_type == "transformer":
            return TransformerSentimentModel(
                model_name=self.transformer_model_name,
                config=self.config,
            )
        raise ValueError(f"Unknown model_type: {self.model_type}")
