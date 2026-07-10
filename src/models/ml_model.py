"""
Traditional Machine-Learning Sentiment Model.

Uses TF-IDF vectorization combined with one of several sklearn classifiers:
- Logistic Regression (default, best general-purpose)
- Linear SVM
- Multinomial Naive Bayes
- Random Forest

Features:
- Cross-validation
- Hyperparameter grid search
- Model serialization with joblib
- Probability calibration for reliable confidence scores

Example::

    model = MLSentimentModel(algorithm="logistic_regression")
    model.train(texts, labels)
    result = model.predict("This is fantastic!")
    model.save("models/ml_model.joblib")
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from .base_model import BaseSentimentModel, SentimentResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Supported Algorithms
# ─────────────────────────────────────────────────────────

ALGORITHM_MAP: Dict[str, Any] = {
    "logistic_regression": LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs"
    ),
    "svm": CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000)),
    "naive_bayes": MultinomialNB(alpha=0.1),
    "random_forest": RandomForestClassifier(n_estimators=200, n_jobs=-1),
}


class MLSentimentModel(BaseSentimentModel):
    """
    Traditional Machine-Learning based sentiment classifier.

    Attributes:
        algorithm: Classifier name – one of ``logistic_regression``, ``svm``,
            ``naive_bayes``, ``random_forest``.
        pipeline: The fitted sklearn Pipeline (vectorizer + classifier).
        label_encoder: Fitted LabelEncoder for string ↔ integer labels.
    """

    MODEL_NAME = "ml_classifier"

    def __init__(
        self,
        algorithm: str = "logistic_regression",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model_name=self.MODEL_NAME, config=config)

        if algorithm not in ALGORITHM_MAP:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. "
                f"Choose from: {list(ALGORITHM_MAP.keys())}"
            )

        self.algorithm = algorithm
        self.pipeline: Optional[Pipeline] = None
        self.label_encoder = LabelEncoder()
        self.classes_: List[str] = []
        self.cv_scores_: Optional[np.ndarray] = None

    # ── Build ─────────────────────────────────────────────

    def _build_pipeline(self) -> Pipeline:
        """Construct the TF-IDF → Classifier sklearn pipeline."""
        cfg = self.config.get("tfidf", {})
        vectorizer = TfidfVectorizer(
            max_features=cfg.get("max_features", 50_000),
            ngram_range=tuple(cfg.get("ngram_range", [1, 3])),
            min_df=cfg.get("min_df", 2),
            max_df=cfg.get("max_df", 0.95),
            sublinear_tf=cfg.get("sublinear_tf", True),
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b\w+\b",
        )
        classifier = ALGORITHM_MAP[self.algorithm]
        return Pipeline([("tfidf", vectorizer), ("clf", classifier)])

    # ── Training ──────────────────────────────────────────

    def train(
        self,
        texts: List[str],
        labels: List[str],
        cv_folds: int = 5,
        test_size: float = 0.2,
        random_state: int = 42,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train the ML classifier.

        Args:
            texts:        List of raw text strings.
            labels:       Corresponding sentiment labels.
            cv_folds:     Number of cross-validation folds (0 to skip).
            test_size:    Fraction of data reserved for a held-out test set.
            random_state: RNG seed for reproducibility.

        Returns:
            Dictionary with training metrics (accuracy, cv_mean, cv_std).
        """
        self._logger.info(
            "Training ML model (%s) on %d samples …", self.algorithm, len(texts)
        )

        # Encode labels
        encoded = self.label_encoder.fit_transform(labels)
        self.classes_ = list(self.label_encoder.classes_)

        # Build pipeline
        self.pipeline = self._build_pipeline()

        # Cross-validation (optional)
        metrics: Dict[str, Any] = {}
        if cv_folds > 1:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            self.cv_scores_ = cross_val_score(
                self.pipeline, texts, encoded, cv=skf, scoring="f1_weighted", n_jobs=-1
            )
            metrics["cv_f1_mean"] = float(np.mean(self.cv_scores_))
            metrics["cv_f1_std"] = float(np.std(self.cv_scores_))
            self._logger.info(
                "CV F1: %.4f ± %.4f", metrics["cv_f1_mean"], metrics["cv_f1_std"]
            )

        # Train/test split evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            texts, encoded, test_size=test_size, random_state=random_state, stratify=encoded
        )
        self.pipeline.fit(X_train, y_train)

        from sklearn.metrics import accuracy_score, f1_score
        y_pred = self.pipeline.predict(X_test)
        metrics["test_accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["test_f1"] = float(f1_score(y_test, y_pred, average="weighted"))
        self._logger.info(
            "Test accuracy: %.4f | Test F1: %.4f",
            metrics["test_accuracy"],
            metrics["test_f1"],
        )

        # Final fit on all data
        self.pipeline.fit(texts, encoded)
        self.is_trained = True
        return metrics

    # ── Prediction ────────────────────────────────────────

    def predict(self, text: str) -> SentimentResult:
        """Predict sentiment for a single text."""
        self._check_trained()
        results = self.predict_batch([text])
        return results[0]

    def predict_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Predict sentiment for a list of texts."""
        self._check_trained()

        proba_matrix = self.pipeline.predict_proba(texts)
        results = []

        for text, proba in zip(texts, proba_matrix):
            idx = int(np.argmax(proba))
            label = self.classes_[idx]
            score = float(proba[idx])
            scores = {cls: float(p) for cls, p in zip(self.classes_, proba)}

            results.append(
                SentimentResult(
                    text=text,
                    label=self._normalize_label(label),
                    score=round(score, 4),
                    scores={self._normalize_label(k): round(v, 4) for k, v in scores.items()},
                    model_name=self.MODEL_NAME,
                    metadata={"algorithm": self.algorithm},
                )
            )
        return results

    # ── Persistence ───────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the trained pipeline and label encoder to disk."""
        self._check_trained()
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        payload = {
            "pipeline": self.pipeline,
            "label_encoder": self.label_encoder,
            "classes": self.classes_,
            "algorithm": self.algorithm,
        }
        joblib.dump(payload, path)
        self._logger.info("Model saved to '%s'.", path)

    def load(self, path: str) -> None:
        """Load a previously saved model from disk."""
        payload = joblib.load(path)
        self.pipeline = payload["pipeline"]
        self.label_encoder = payload["label_encoder"]
        self.classes_ = payload["classes"]
        self.algorithm = payload["algorithm"]
        self.is_trained = True
        self._logger.info("Model loaded from '%s'.", path)

    # ── Helpers ───────────────────────────────────────────

    def _check_trained(self) -> None:
        if not self.is_trained or self.pipeline is None:
            raise RuntimeError(
                "Model is not trained yet. Call train() or load() first."
            )

    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Return the top-N most informative features (words) for each class.
        Only works when the underlying classifier exposes ``coef_``.

        Returns:
            List of (word, coefficient) sorted by descending importance.
        """
        self._check_trained()
        vectorizer: TfidfVectorizer = self.pipeline.named_steps["tfidf"]
        clf = self.pipeline.named_steps["clf"]

        # Handle CalibratedClassifierCV wrapper
        base_clf = getattr(clf, "estimator", clf)
        if not hasattr(base_clf, "coef_"):
            raise AttributeError(
                f"Algorithm '{self.algorithm}' does not expose feature coefficients."
            )

        feature_names = vectorizer.get_feature_names_out()
        coef = base_clf.coef_

        top_features: List[Tuple[str, float]] = []
        for i, cls in enumerate(self.classes_):
            indices = np.argsort(coef[i])[-top_n:][::-1]
            for idx in indices:
                top_features.append((feature_names[idx], float(coef[i][idx])))

        return top_features
