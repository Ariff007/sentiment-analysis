"""
Model Evaluation Utilities.

Computes classification metrics, generates a rich text report, and
optionally produces confusion matrix and ROC curve visualizations.

Example::

    evaluator = SentimentEvaluator()
    metrics = evaluator.evaluate(model, test_texts, true_labels)
    evaluator.print_report()
    evaluator.plot_confusion_matrix(save_path="reports/cm.png")
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _plotting_available = True
except ImportError:
    _plotting_available = False

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..models.base_model import BaseSentimentModel


class SentimentEvaluator:
    """
    Evaluates a sentiment model against a ground-truth labelled dataset.

    After calling :meth:`evaluate`, access results via:
    - :attr:`metrics` – dict of scalar performance metrics
    - :meth:`print_report` – formatted classification report
    - :meth:`plot_confusion_matrix` – saves/shows a confusion matrix heatmap
    """

    def __init__(self, average: str = "weighted"):
        self.average = average
        self.metrics: Dict[str, Any] = {}
        self._true_labels: List[str] = []
        self._pred_labels: List[str] = []
        self._classes: List[str] = []
        self._report: str = ""

    # ── Evaluation ────────────────────────────────────────

    def evaluate(
        self,
        model: BaseSentimentModel,
        texts: List[str],
        true_labels: List[str],
        preprocess: bool = False,
    ) -> Dict[str, Any]:
        """
        Run inference on ``texts`` and compute metrics vs ``true_labels``.

        Args:
            model:        A fitted :class:`BaseSentimentModel` instance.
            texts:        List of raw input texts.
            true_labels:  Ground-truth sentiment labels.
            preprocess:   Whether to clean texts before prediction.

        Returns:
            Dict of metric names → scalar values.
        """
        logger.info("Evaluating model on %d samples …", len(texts))

        results = model.predict_batch(texts)
        pred_labels = [r.label for r in results]

        return self.evaluate_predictions(true_labels, pred_labels)

    def evaluate_predictions(
        self,
        true_labels: List[str],
        pred_labels: List[str],
    ) -> Dict[str, Any]:
        """
        Compute metrics given ground-truth and predicted label lists.

        Args:
            true_labels: Ground-truth sentiment labels.
            pred_labels: Model-predicted labels.

        Returns:
            Dict of metric names → scalar values.
        """
        # Normalise labels
        self._true_labels = [l.strip().lower() for l in true_labels]
        self._pred_labels = [l.strip().lower() for l in pred_labels]
        self._classes = sorted(set(self._true_labels + self._pred_labels))

        # Compute metrics
        acc = accuracy_score(self._true_labels, self._pred_labels)
        prec = precision_score(
            self._true_labels, self._pred_labels,
            average=self.average, zero_division=0
        )
        rec = recall_score(
            self._true_labels, self._pred_labels,
            average=self.average, zero_division=0
        )
        f1 = f1_score(
            self._true_labels, self._pred_labels,
            average=self.average, zero_division=0
        )

        self.metrics = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "num_samples": len(self._true_labels),
            "classes": self._classes,
        }

        # ROC-AUC (only for binary classification)
        if len(self._classes) == 2:
            try:
                binary_true = [1 if l == self._classes[1] else 0
                               for l in self._true_labels]
                binary_pred = [1 if l == self._classes[1] else 0
                               for l in self._pred_labels]
                roc = roc_auc_score(binary_true, binary_pred)
                self.metrics["roc_auc"] = round(roc, 4)
            except Exception:
                pass

        self._report = classification_report(
            self._true_labels,
            self._pred_labels,
            target_names=self._classes,
            zero_division=0,
        )

        logger.info("Evaluation complete: %s", self.metrics)
        return self.metrics

    # ── Reporting ─────────────────────────────────────────

    def print_report(self) -> None:
        """Print a formatted classification report to stdout."""
        if not self._report:
            print("No evaluation has been run yet. Call evaluate() first.")
            return
        print("\n" + "=" * 60)
        print(" SENTIMENT ANALYSIS – EVALUATION REPORT")
        print("=" * 60)
        for k, v in self.metrics.items():
            if k not in ("classes",):
                print(f"  {k:<20}: {v}")
        print("\nPer-class breakdown:")
        print(self._report)
        print("=" * 60)

    def get_classification_report(self) -> str:
        """Return the sklearn classification report as a string."""
        return self._report

    def plot_confusion_matrix(
        self,
        save_path: Optional[str] = None,
        title: str = "Sentiment Confusion Matrix",
        figsize: tuple = (8, 6),
    ) -> None:
        """
        Plot and optionally save a confusion matrix heatmap.

        Args:
            save_path: If provided, saves the figure to this path.
            title:     Plot title.
            figsize:   Figure size tuple.
        """
        if not _plotting_available:
            logger.warning("matplotlib/seaborn not installed – cannot plot.")
            return

        if not self._true_labels:
            raise RuntimeError("No evaluation results. Call evaluate() first.")

        cm = confusion_matrix(self._true_labels, self._pred_labels,
                              labels=self._classes)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm_norm,
            annot=cm,
            fmt="d",
            cmap="Blues",
            xticklabels=self._classes,
            yticklabels=self._classes,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Confusion matrix saved to '%s'.", save_path)

        plt.show()
        plt.close(fig)

    def save_report(self, path: str) -> None:
        """Save the full evaluation report to a text file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("SENTIMENT ANALYSIS – EVALUATION REPORT\n")
            f.write("=" * 60 + "\n")
            for k, v in self.metrics.items():
                f.write(f"{k}: {v}\n")
            f.write("\n" + self._report)
        logger.info("Report saved to '%s'.", path)
