"""
Prediction Entry Point.

Convenience script for quick inference from the command line.

Usage::

    # Predict from a single text string
    python predict.py --text "I love this product!"

    # Predict using ML model
    python predict.py --text "Terrible experience" --model ml --model-path models/ml_model.joblib

    # Predict from a file (one text per line)
    python predict.py --file data/reviews.txt --output results.csv

    # Predict using transformer model
    python predict.py --text "Amazing quality!" --model transformer
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

# Force UTF-8 output on Windows (prevents UnicodeEncodeError with emoji/special chars)
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.text_cleaner import TextCleaner
from src.utils.helpers import create_model, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sentiment analysis inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="Single text string to analyse.")
    input_group.add_argument("--file", help="Path to text file (one text per line).")
    input_group.add_argument("--csv", help="Path to CSV file.")

    # Model
    parser.add_argument("--model", default="vader",
                        choices=["vader", "ml", "transformer"],
                        help="Model type. (default: vader)")
    parser.add_argument("--model-path", default=None,
                        help="Path to a saved ML model file.")
    parser.add_argument("--text-col", default="text",
                        help="CSV column name for text. (default: text)")

    # Preprocessing
    parser.add_argument("--no-preprocess", action="store_true",
                        help="Skip text preprocessing.")

    # Output
    parser.add_argument("--output", default=None,
                        help="Save results to this CSV file.")
    parser.add_argument("--json-output", action="store_true",
                        help="Print results as JSON.")
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level=args.log_level)

    model = create_model(model_type=args.model, model_path=args.model_path)
    cleaner = TextCleaner()
    preprocess = not args.no_preprocess

    # ── Gather texts ──────────────────────────────────────
    texts = []

    if args.text:
        texts = [args.text]

    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [ln.strip() for ln in f if ln.strip()]

    elif args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        if args.text_col not in df.columns:
            print(f"Error: Column '{args.text_col}' not found in CSV.", file=sys.stderr)
            sys.exit(1)
        texts = df[args.text_col].dropna().tolist()

    if not texts:
        print("Error: No input text found.", file=sys.stderr)
        sys.exit(1)

    # ── Predict ───────────────────────────────────────────
    cleaned = cleaner.clean_batch(texts) if preprocess else texts
    results = model.predict_batch(cleaned)

    # ── Output ────────────────────────────────────────────
    output_data = []
    for original, result in zip(texts, results):
        row = {
            "text": original,
            "label": result.label,
            "score": result.score,
            **{f"score_{k}": v for k, v in result.scores.items()},
            "model": result.model_name,
        }
        output_data.append(row)

    if args.json_output:
        print(json.dumps(output_data, indent=2, ensure_ascii=False))
    elif len(texts) == 1:
        r = results[0]
        label_symbols = {"positive": "[+]", "negative": "[-]", "neutral": "[~]"}
        sym = label_symbols.get(r.label, "[?]")
        print(f"\n  Text       : {texts[0][:100]}")
        print(f"  Label      : {sym} {r.label.upper()}")
        print(f"  Confidence : {r.score:.1%}")
        if r.scores:
            print(f"  Scores     : {r.scores}")
        print(f"  Model      : {r.model_name}\n")
    else:
        # Print table
        print(f"\n{'-'*80}")
        print(f"{'TEXT':<45} {'LABEL':<12} {'SCORE'}")
        print(f"{'-'*80}")
        for row in output_data:
            text_trunc = row["text"][:43] + "..." if len(row["text"]) > 44 else row["text"]
            print(f"{text_trunc:<45} {row['label']:<12} {row['score']:.4f}")
        print(f"{'-'*80}")
        print(f"Total: {len(output_data)} texts analysed using '{args.model}' model.\n")

    if args.output:
        import csv as csv_mod
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv_mod.DictWriter(f, fieldnames=output_data[0].keys())
            writer.writeheader()
            writer.writerows(output_data)
        print(f"Results saved to '{args.output}'")


if __name__ == "__main__":
    main()
