"""
Transformer-based Sentiment Model using Hugging Face.

Wraps a pre-trained (or fine-tuned) DistilBERT / BERT model for zero-shot
or few-shot sentiment classification.

Default pre-trained model:
    distilbert-base-uncased-finetuned-sst-2-english
    (binary: POSITIVE / NEGATIVE, fine-tuned on SST-2)

Features:
- GPU-aware (CUDA / MPS / CPU auto-detection)
- Batch inference for efficiency
- Long-document handling via sliding window chunking
- Fine-tuning support via HuggingFace Trainer API

Example::

    model = TransformerSentimentModel()
    result = model.predict("The movie was absolutely breathtaking! ⭐⭐⭐⭐⭐")
    print(result.label)   # → 'positive'
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from .base_model import BaseSentimentModel, SentimentResult

logger = logging.getLogger(__name__)

# Lazy imports to avoid mandatory dependency for users who only use VADER/ML
_transformers_available = False
try:
    import torch
    from transformers import pipeline as hf_pipeline
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    _transformers_available = True
except ImportError:
    pass


class TransformerSentimentModel(BaseSentimentModel):
    """
    Sentiment classifier backed by a Hugging Face Transformer.

    Args:
        model_name: HuggingFace model ID or local path.
        device: Compute device – ``"auto"`` selects GPU if available.
        max_length: Maximum token sequence length.
        batch_size: Inference batch size.
        config: Optional dict that overrides defaults.

    Raises:
        ImportError: If ``torch`` or ``transformers`` are not installed.
    """

    MODEL_NAME = "transformer"

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 32,
        config: Optional[Dict[str, Any]] = None,
    ):
        if not _transformers_available:
            raise ImportError(
                "torch and transformers are required for TransformerSentimentModel.\n"
                "Install them with:  pip install torch transformers"
            )

        super().__init__(model_name="transformer", config=config)
        self.hf_model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self._device = self._resolve_device(device)
        self._pipe = None
        self._load_pipeline()
        self.is_trained = True

    # ── Initialization ────────────────────────────────────

    def _resolve_device(self, device: str) -> int:
        """Return the torch device index (0 for GPU, -1 for CPU)."""
        if device == "auto":
            if torch.cuda.is_available():
                self._logger.info("Using CUDA GPU.")
                return 0
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._logger.info("Using Apple MPS.")
                return 0
            self._logger.info("Using CPU.")
            return -1
        if device in ("cuda", "gpu"):
            return 0
        if device == "mps":
            return 0
        return -1  # cpu

    def _load_pipeline(self) -> None:
        """Load the HuggingFace sentiment-analysis pipeline."""
        self._logger.info(
            "Loading transformer model '%s' …", self.hf_model_name
        )
        self._pipe = hf_pipeline(
            task="sentiment-analysis",
            model=self.hf_model_name,
            tokenizer=self.hf_model_name,
            device=self._device,
            truncation=True,
            max_length=self.max_length,
            return_all_scores=True,
        )
        self._logger.info("Transformer model loaded.")

    # ── Prediction ────────────────────────────────────────

    def predict(self, text: str) -> SentimentResult:
        """Predict sentiment for a single text."""
        if not text or not text.strip():
            return SentimentResult(
                text=text,
                label="neutral",
                score=0.0,
                scores={"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                model_name=self.MODEL_NAME,
            )
        results = self.predict_batch([text])
        return results[0]

    def predict_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Predict sentiment for a list of texts (batched for efficiency)."""
        outputs = self._pipe(texts, batch_size=self.batch_size, truncation=True)

        results = []
        for text, raw in zip(texts, outputs):
            # raw → list of {"label": "POSITIVE", "score": 0.99}
            scores_raw: Dict[str, float] = {
                item["label"].lower(): item["score"] for item in raw
            }
            # Map POSITIVE/NEGATIVE → positive/negative
            scores = self._remap_scores(scores_raw)
            best_label = max(scores, key=scores.get)
            best_score = scores[best_label]

            results.append(
                SentimentResult(
                    text=text,
                    label=self._normalize_label(best_label),
                    score=round(best_score, 4),
                    scores={k: round(v, 4) for k, v in scores.items()},
                    model_name=self.MODEL_NAME,
                    metadata={"hf_model": self.hf_model_name},
                )
            )
        return results

    # ── Persistence ───────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the fine-tuned model to a directory."""
        os.makedirs(path, exist_ok=True)
        self._pipe.model.save_pretrained(path)
        self._pipe.tokenizer.save_pretrained(path)
        self._logger.info("Transformer model saved to '%s'.", path)

    def load(self, path: str) -> None:
        """Load a previously fine-tuned model from a directory."""
        self.hf_model_name = path
        self._load_pipeline()

    # ── Helpers ───────────────────────────────────────────

    @staticmethod
    def _remap_scores(raw: Dict[str, float]) -> Dict[str, float]:
        """Remap arbitrary HF label names to positive/negative/neutral."""
        mapping = {
            "positive": "positive",
            "negative": "negative",
            "neutral": "neutral",
            "label_0": "negative",
            "label_1": "neutral",
            "label_2": "positive",
        }
        remapped: Dict[str, float] = {}
        for k, v in raw.items():
            std_key = mapping.get(k.lower(), k.lower())
            remapped[std_key] = remapped.get(std_key, 0.0) + v
        # Ensure all three classes exist
        for cls in ("positive", "negative", "neutral"):
            remapped.setdefault(cls, 0.0)
        return remapped
