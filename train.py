"""
Training Entry Point.

Convenience script to train a sentiment model from the command line.

Usage::

    # Train default ML model
    python train.py --data data/sample_data.csv

    # Train SVM with cross-validation
    python train.py --data data/sample_data.csv --algo svm --cv-folds 5

    # Train with a custom save path
    python train.py --data data/train.csv --save-path models/my_model.joblib
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

# Force UTF-8 output on Windows (prevents UnicodeEncodeError with emoji/special chars)
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

from src.training.trainer import SentimentTrainer
from src.utils.helpers import load_config, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a sentiment analysis model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data", required=True, help="Path to training CSV file.")
    parser.add_argument("--model", default="ml", choices=["ml", "transformer"],
                        help="Model type to train. (default: ml)")
    parser.add_argument("--algo", default="logistic_regression",
                        choices=["logistic_regression", "svm", "naive_bayes", "random_forest"],
                        help="Algorithm for ML models. (default: logistic_regression)")
    parser.add_argument("--text-col", default="text",
                        help="Column name for text. (default: text)")
    parser.add_argument("--label-col", default="sentiment",
                        help="Column name for labels. (default: sentiment)")
    parser.add_argument("--save-path", default="models/ml_model.joblib",
                        help="Path to save the trained model.")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of cross-validation folds. (default: 5)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data for test set. (default: 0.2)")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Use only N training samples (for quick tests).")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity. (default: INFO)")
    parser.add_argument("--config", default=None,
                        help="Path to custom config.yaml file.")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level=args.log_level, log_file="logs/training.log")
    cfg = load_config(args.config)

    print(f"\n{'='*60}")
    print(f"  SentimentAI – Training Pipeline")
    print(f"{'='*60}")
    print(f"  Model      : {args.model} ({args.algo})")
    print(f"  Data       : {args.data}")
    print(f"  Save Path  : {args.save_path}")
    print(f"  CV Folds   : {args.cv_folds}")
    print(f"{'='*60}\n")

    trainer = SentimentTrainer(
        model_type=args.model,
        algorithm=args.algo,
        config=cfg.get("ml_model", {}),
    )

    metrics = trainer.train_from_csv(
        path=args.data,
        text_col=args.text_col,
        label_col=args.label_col,
        cv_folds=args.cv_folds,
        test_size=args.test_size,
        sample_size=args.sample_size,
    )

    trainer.save_model(args.save_path)

    print(f"\n{'─'*60}")
    print("  Training Metrics:")
    print(f"{'─'*60}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<30}: {v:.4f}")
        else:
            print(f"  {k:<30}: {v}")
    print(f"{'─'*60}")
    print(f"\n  ✓ Model saved to '{args.save_path}'")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
