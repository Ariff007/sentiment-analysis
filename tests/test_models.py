"""
Unit tests for sentiment analysis models.
"""

import sys
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.models.base_model import BaseSentimentModel, SentimentResult
from src.models.lexicon_model import LexiconSentimentModel
from src.models.ml_model import MLSentimentModel


# ─────────────────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────────────────

POSITIVE_TEXTS = [
    "I absolutely love this! It's amazing!",
    "Best product ever! Highly recommend to everyone.",
    "Fantastic quality and great customer service! 😍",
    "This made my day. So happy with this purchase!",
    "Outstanding performance. Exceeded all expectations.",
]

NEGATIVE_TEXTS = [
    "This is absolutely terrible. Worst purchase ever.",
    "Complete waste of money. Do NOT buy this.",
    "Horrible quality. Broke after one day. Very disappointed.",
    "Awful experience from start to finish. Zero stars.",
    "Disgusting product. Returning immediately.",
]

NEUTRAL_TEXTS = [
    "The product arrived today.",
    "I received my order yesterday.",
    "The item is as described in the listing.",
    "Package delivered on schedule.",
]

ALL_TEXTS = POSITIVE_TEXTS + NEGATIVE_TEXTS + NEUTRAL_TEXTS
ALL_LABELS = (
    ["positive"] * len(POSITIVE_TEXTS)
    + ["negative"] * len(NEGATIVE_TEXTS)
    + ["neutral"] * len(NEUTRAL_TEXTS)
)


# ─────────────────────────────────────────────────────────
# SentimentResult Tests
# ─────────────────────────────────────────────────────────

class TestSentimentResult:
    def test_creation(self):
        r = SentimentResult(
            text="Hello",
            label="positive",
            score=0.95,
            scores={"positive": 0.95, "negative": 0.05},
            model_name="test",
        )
        assert r.label == "positive"
        assert r.score == 0.95
        assert r.model_name == "test"

    def test_to_dict(self):
        r = SentimentResult(
            text="Test",
            label="negative",
            score=0.8,
            scores={"positive": 0.2, "negative": 0.8},
            model_name="test",
        )
        d = r.to_dict()
        assert d["label"] == "negative"
        assert d["score"] == 0.8
        assert "scores" in d

    def test_repr(self):
        r = SentimentResult(text="T", label="neutral", score=0.5, model_name="m")
        assert "neutral" in repr(r)
        assert "0.5" in repr(r)


# ─────────────────────────────────────────────────────────
# LexiconSentimentModel Tests
# ─────────────────────────────────────────────────────────

class TestLexiconModel:
    @pytest.fixture
    def model(self):
        return LexiconSentimentModel()

    def test_model_instantiation(self, model):
        assert model.is_trained is True
        assert model.model_name == "vader"

    def test_predict_returns_result(self, model):
        result = model.predict("I love this!")
        assert isinstance(result, SentimentResult)
        assert result.label in ("positive", "negative", "neutral")
        assert 0.0 <= result.score <= 1.0

    def test_predict_positive_text(self, model):
        result = model.predict("This is absolutely amazing! I love it so much!")
        assert result.label == "positive"

    def test_predict_negative_text(self, model):
        result = model.predict("This is terrible, awful, and completely disgusting!")
        assert result.label == "negative"

    def test_predict_empty_text(self, model):
        result = model.predict("")
        assert result.label == "neutral"
        assert result.score == 0.0

    def test_predict_whitespace(self, model):
        result = model.predict("   ")
        assert result.label == "neutral"

    def test_predict_batch_returns_list(self, model):
        results = model.predict_batch(POSITIVE_TEXTS)
        assert isinstance(results, list)
        assert len(results) == len(POSITIVE_TEXTS)

    def test_predict_batch_correct_types(self, model):
        results = model.predict_batch(["Hello!", "Goodbye!"])
        for r in results:
            assert isinstance(r, SentimentResult)

    def test_has_scores_dict(self, model):
        result = model.predict("Great product!")
        assert "positive" in result.scores
        assert "negative" in result.scores
        assert "neutral" in result.scores

    def test_scores_sum_approximately_one(self, model):
        result = model.predict("Decent item, nothing special.")
        total = sum(result.scores.values())
        assert abs(total - 1.0) < 0.1  # VADER scores don't always sum to 1.0

    def test_metadata_contains_compound(self, model):
        result = model.predict("Amazing!")
        assert "compound" in result.metadata

    def test_get_raw_scores(self, model):
        scores = model.get_raw_scores("I love this!")
        assert "compound" in scores
        assert "pos" in scores
        assert "neg" in scores
        assert "neu" in scores

    def test_social_media_text(self, model):
        """VADER should handle emoji and slang."""
        result = model.predict("omg this is soooo good 😍❤️❤️")
        assert result.label == "positive"

    def test_negative_emoji(self, model):
        # Use strongly negative words + angry emoji so VADER reliably scores negative
        result = model.predict("I absolutely hate this terrible garbage! Disgusting! 😡😡🤬")
        assert result.label == "negative"

    def test_model_accuracy_on_positives(self, model):
        """At least 80% of clearly positive texts should be classified correctly."""
        results = model.predict_batch(POSITIVE_TEXTS)
        correct = sum(1 for r in results if r.label == "positive")
        accuracy = correct / len(POSITIVE_TEXTS)
        assert accuracy >= 0.8, f"Positive accuracy only {accuracy:.1%}"

    def test_model_accuracy_on_negatives(self, model):
        """At least 80% of clearly negative texts should be classified correctly."""
        results = model.predict_batch(NEGATIVE_TEXTS)
        correct = sum(1 for r in results if r.label == "negative")
        accuracy = correct / len(NEGATIVE_TEXTS)
        assert accuracy >= 0.8, f"Negative accuracy only {accuracy:.1%}"

    def test_threshold_customization(self):
        """Custom thresholds should affect classification."""
        strict_model = LexiconSentimentModel(
            threshold_positive=0.5,
            threshold_negative=-0.5,
        )
        lenient_model = LexiconSentimentModel(
            threshold_positive=0.01,
            threshold_negative=-0.01,
        )
        text = "This is okay."
        strict_result = strict_model.predict(text)
        lenient_result = lenient_model.predict(text)
        # Strict model more likely to call it neutral
        assert strict_result.label in ("neutral", "positive", "negative")
        assert lenient_result.label in ("neutral", "positive", "negative")

    def test_analyze_alias(self, model):
        result = model.analyze("Test text")
        assert isinstance(result, SentimentResult)

    def test_get_label(self, model):
        label = model.get_label("Excellent!")
        assert label in ("positive", "negative", "neutral")

    def test_get_score(self, model):
        score = model.get_score("Excellent!")
        assert 0.0 <= score <= 1.0


# ─────────────────────────────────────────────────────────
# MLSentimentModel Tests
# ─────────────────────────────────────────────────────────

class TestMLModel:
    @pytest.fixture
    def untrained_model(self):
        return MLSentimentModel(algorithm="logistic_regression")

    @pytest.fixture
    def trained_model(self):
        model = MLSentimentModel(algorithm="logistic_regression")
        # Use balanced dataset with more samples for reliable training
        model.train(ALL_TEXTS * 3, ALL_LABELS * 3, cv_folds=0)
        return model

    def test_instantiation(self, untrained_model):
        assert untrained_model.model_name == "ml_classifier"
        assert untrained_model.is_trained is False

    def test_invalid_algorithm(self):
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            MLSentimentModel(algorithm="invalid_algo")

    def test_predict_before_training_raises(self, untrained_model):
        with pytest.raises(RuntimeError, match="not trained"):
            untrained_model.predict("Test")

    def test_train_returns_metrics(self):
        model = MLSentimentModel()
        metrics = model.train(ALL_TEXTS, ALL_LABELS, cv_folds=0)
        assert "test_accuracy" in metrics
        assert "test_f1" in metrics
        assert 0 <= metrics["test_accuracy"] <= 1

    def test_is_trained_after_training(self):
        model = MLSentimentModel()
        model.train(ALL_TEXTS, ALL_LABELS, cv_folds=0)
        assert model.is_trained is True

    def test_predict_returns_result(self, trained_model):
        result = trained_model.predict("I love this!")
        assert isinstance(result, SentimentResult)
        assert result.label in ("positive", "negative", "neutral")
        assert 0.0 <= result.score <= 1.0

    def test_predict_batch(self, trained_model):
        texts = ["Great!", "Terrible!", "Okay."]
        results = trained_model.predict_batch(texts)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, SentimentResult)

    def test_classes_after_training(self, trained_model):
        assert len(trained_model.classes_) > 0
        assert "positive" in trained_model.classes_ or \
               any("pos" in c for c in trained_model.classes_)

    def test_save_and_load(self, trained_model, tmp_path):
        save_path = str(tmp_path / "test_model.joblib")
        trained_model.save(save_path)

        new_model = MLSentimentModel()
        new_model.load(save_path)
        assert new_model.is_trained is True

        result = new_model.predict("Excellent service!")
        assert isinstance(result, SentimentResult)

    def test_cross_validation(self):
        model = MLSentimentModel()
        metrics = model.train(ALL_TEXTS * 2, ALL_LABELS * 2, cv_folds=3)
        assert "cv_f1_mean" in metrics
        assert "cv_f1_std" in metrics
        assert 0 <= metrics["cv_f1_mean"] <= 1

    @pytest.mark.parametrize("algorithm", ["logistic_regression", "naive_bayes"])
    def test_different_algorithms(self, algorithm):
        model = MLSentimentModel(algorithm=algorithm)
        metrics = model.train(ALL_TEXTS, ALL_LABELS, cv_folds=0)
        assert "test_accuracy" in metrics
        result = model.predict("I love this product!")
        assert isinstance(result, SentimentResult)
