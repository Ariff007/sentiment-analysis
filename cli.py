"""
SentimentAI – Command-Line Interface.

Usage examples:

    # Analyse a single text
    python cli.py analyze "I love this product!"

    # Analyse using a specific model
    python cli.py analyze "Terrible experience" --model vader

    # Analyse from a file
    python cli.py analyze-file data/sample_data.csv --model vader

    # Train an ML model
    python cli.py train data/sample_data.csv --model ml --algo logistic_regression

    # Evaluate a model
    python cli.py evaluate data/test_data.csv --model-path models/ml_model.joblib
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

# Setup sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.helpers import create_model, load_config, setup_logging

console = Console()


# ─────────────────────────────────────────────────────────
# Styling helpers
# ─────────────────────────────────────────────────────────

LABEL_STYLES = {
    "positive": "[bold green][+] POSITIVE[/bold green]",
    "negative": "[bold red][-] NEGATIVE[/bold red]",
    "neutral":  "[bold yellow][~] NEUTRAL[/bold yellow]",
}

LABEL_COLORS = {
    "positive": "green",
    "negative": "red",
    "neutral": "yellow",
}


def _label_display(label: str) -> str:
    return LABEL_STYLES.get(label, f"[dim]{label}[/dim]")


def _print_banner():
    console.print(
        Panel.fit(
            "[bold cyan]SentimentAI[/bold cyan] [dim]v1.0.0[/dim]\n"
            "[italic]Multi-model Sentiment Analysis System[/italic]",
            border_style="cyan",
        )
    )


def _print_result(result, show_scores: bool = False):
    label_str = _label_display(result.label)
    score_bar = "█" * int(result.score * 20) + "░" * (20 - int(result.score * 20))
    color = LABEL_COLORS.get(result.label, "white")

    console.print(f"\n  Sentiment : {label_str}")
    console.print(f"  Confidence: [{color}]{score_bar}[/{color}] {result.score:.1%}")
    console.print(f"  Model     : [dim]{result.model_name}[/dim]")

    if show_scores and result.scores:
        console.print("\n  Per-class scores:")
        for cls, sc in result.scores.items():
            bar = "▓" * int(sc * 15) + "░" * (15 - int(sc * 15))
            console.print(f"    {cls:<12} {bar} {sc:.4f}")


# ─────────────────────────────────────────────────────────
# CLI Group
# ─────────────────────────────────────────────────────────

@click.group()
@click.option("--log-level", default="WARNING", help="Logging level.")
@click.pass_context
def cli(ctx, log_level):
    """SentimentAI – Command-Line Sentiment Analysis Tool."""
    setup_logging(level=log_level)
    ctx.ensure_object(dict)


# ─────────────────────────────────────────────────────────
# analyze
# ─────────────────────────────────────────────────────────

@cli.command()
@click.argument("text", required=False)
@click.option("--model", "-m", default="vader",
              type=click.Choice(["vader", "ml", "transformer"]),
              help="Sentiment model to use.", show_default=True)
@click.option("--model-path", default=None,
              help="Path to a saved ML model (.joblib).")
@click.option("--preprocess/--no-preprocess", default=True,
              help="Apply text cleaning before inference.", show_default=True)
@click.option("--scores", is_flag=True,
              help="Show per-class probability scores.")
@click.option("--json", "output_json", is_flag=True,
              help="Output results as JSON.")
@click.option("--stdin", is_flag=True,
              help="Read text from stdin.")
def analyze(text, model, model_path, preprocess, scores, output_json, stdin):
    """Analyse the sentiment of TEXT.

    If --stdin is set, reads from standard input.
    """
    _print_banner()

    # Resolve input text
    if stdin or not text:
        if sys.stdin.isatty():
            console.print("[yellow]Enter text (Ctrl+D to finish):[/yellow]")
        text = sys.stdin.read().strip()

    if not text:
        console.print("[red]Error: no text provided.[/red]")
        sys.exit(1)

    # Load model
    with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"),
                  transient=True, console=console) as progress:
        progress.add_task(f"Loading {model} model …")
        m = create_model(model_type=model, model_path=model_path)

    # Pre-process
    from src.preprocessing.text_cleaner import TextCleaner
    clean_text = TextCleaner().clean(text) if preprocess else text

    # Predict
    t0 = time.perf_counter()
    result = m.predict(clean_text)
    elapsed = (time.perf_counter() - t0) * 1000

    if output_json:
        data = result.to_dict()
        data["processing_time_ms"] = round(elapsed, 2)
        console.print(json.dumps(data, indent=2))
        return

    console.print(f"\n  [bold]Input[/bold] : {text[:120]}{'…' if len(text) > 120 else ''}")
    _print_result(result, show_scores=scores)
    console.print(f"\n  [dim]Processed in {elapsed:.1f} ms[/dim]\n")


# ─────────────────────────────────────────────────────────
# analyze-file
# ─────────────────────────────────────────────────────────

@cli.command("analyze-file")
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--model", "-m", default="vader",
              type=click.Choice(["vader", "ml", "transformer"]),
              show_default=True)
@click.option("--model-path", default=None)
@click.option("--text-col", default="text",
              help="Column name for text (CSV files).", show_default=True)
@click.option("--output", "-o", default=None,
              help="Save results to this CSV file.")
@click.option("--preprocess/--no-preprocess", default=True)
@click.option("--limit", default=None, type=int,
              help="Maximum number of lines to process.")
def analyze_file(filepath, model, model_path, text_col, output, preprocess, limit):
    """Analyse sentiment for every line / row in FILEPATH."""
    import pandas as pd
    from src.preprocessing.text_cleaner import TextCleaner

    _print_banner()

    fp = Path(filepath)
    if fp.suffix == ".csv":
        df = pd.read_csv(fp)
        if text_col not in df.columns:
            console.print(f"[red]Column '{text_col}' not found. Available: {list(df.columns)}[/red]")
            sys.exit(1)
        texts = df[text_col].dropna().tolist()
    else:
        with open(fp, "r", encoding="utf-8") as f:
            texts = [ln.strip() for ln in f if ln.strip()]

    if limit:
        texts = texts[:limit]

    m = create_model(model_type=model, model_path=model_path)
    cleaner = TextCleaner()

    results_data = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Analysing {len(texts)} texts …", total=len(texts))
        for text in texts:
            clean = cleaner.clean(text) if preprocess else text
            result = m.predict(clean)
            results_data.append({
                "text": text,
                "label": result.label,
                "score": result.score,
                **{f"score_{k}": v for k, v in result.scores.items()},
            })
            progress.advance(task)

    # Summary table
    import collections
    label_counts = collections.Counter(r["label"] for r in results_data)
    table = Table(title="\nSentiment Summary", show_header=True, header_style="bold cyan")
    table.add_column("Label", style="bold", width=12)
    table.add_column("Count", justify="right", width=8)
    table.add_column("Percentage", justify="right", width=12)
    for label in ("positive", "neutral", "negative"):
        count = label_counts.get(label, 0)
        pct = count / len(results_data) * 100
        color = LABEL_COLORS.get(label, "white")
        table.add_row(
            f"[{color}]{label.capitalize()}[/{color}]",
            str(count),
            f"{pct:.1f}%",
        )
    console.print(table)

    if output:
        out_df = pd.DataFrame(results_data)
        out_df.to_csv(output, index=False)
        console.print(f"\n[green]>> Results saved to '{output}'[/green]")


# ─────────────────────────────────────────────────────────
# train
# ─────────────────────────────────────────────────────────

@cli.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--model", "-m", default="ml",
              type=click.Choice(["ml", "transformer"]), show_default=True)
@click.option("--algo", default="logistic_regression",
              type=click.Choice(["logistic_regression", "svm", "naive_bayes", "random_forest"]),
              show_default=True)
@click.option("--text-col", default="text", show_default=True)
@click.option("--label-col", default="sentiment", show_default=True)
@click.option("--save-path", default="models/model.joblib", show_default=True)
@click.option("--cv-folds", default=5, type=int, show_default=True)
def train(data_path, model, algo, text_col, label_col, save_path, cv_folds):
    """Train a sentiment model on a labelled CSV dataset."""
    from src.training.trainer import SentimentTrainer

    _print_banner()
    console.print(f"\n[cyan]Training {model} model ({algo}) on '{data_path}' …[/cyan]\n")

    trainer = SentimentTrainer(model_type=model, algorithm=algo)

    with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"),
                  transient=True, console=console) as progress:
        progress.add_task("Training …")
        metrics = trainer.train_from_csv(
            data_path,
            text_col=text_col,
            label_col=label_col,
            cv_folds=cv_folds,
        )

    table = Table(title="Training Results", header_style="bold green")
    table.add_column("Metric", style="bold", width=22)
    table.add_column("Value", justify="right", width=12)
    for k, v in metrics.items():
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(table)

    trainer.save_model(save_path)
    console.print(f"\n[green]>> Model saved to '{save_path}'[/green]\n")


# ─────────────────────────────────────────────────────────
# evaluate
# ─────────────────────────────────────────────────────────

@cli.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--model", "-m", default="vader",
              type=click.Choice(["vader", "ml", "transformer"]))
@click.option("--model-path", default=None)
@click.option("--text-col", default="text", show_default=True)
@click.option("--label-col", default="sentiment", show_default=True)
@click.option("--plot/--no-plot", default=False,
              help="Show confusion matrix plot.")
@click.option("--report-path", default=None,
              help="Save evaluation report to file.")
def evaluate(data_path, model, model_path, text_col, label_col, plot, report_path):
    """Evaluate a model against a labelled test dataset."""
    import pandas as pd
    from src.evaluation.evaluator import SentimentEvaluator

    _print_banner()

    df = pd.read_csv(data_path)[[text_col, label_col]].dropna()
    texts = df[text_col].tolist()
    labels = df[label_col].str.lower().tolist()

    m = create_model(model_type=model, model_path=model_path)
    evaluator = SentimentEvaluator()

    with Progress(SpinnerColumn(), TextColumn("[cyan]Evaluating …"), transient=True,
                  console=console) as progress:
        progress.add_task("")
        metrics = evaluator.evaluate(m, texts, labels)

    evaluator.print_report()

    if plot:
        evaluator.plot_confusion_matrix()

    if report_path:
        evaluator.save_report(report_path)
        console.print(f"\n[green]>> Report saved to '{report_path}'[/green]")


# ─────────────────────────────────────────────────────────
# serve
# ─────────────────────────────────────────────────────────

@cli.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, type=int, show_default=True)
@click.option("--reload", is_flag=True, help="Enable auto-reload (development).")
@click.option("--workers", default=1, type=int, show_default=True)
def serve(host, port, reload, workers):
    """Start the SentimentAI REST API server."""
    import uvicorn
    _print_banner()
    console.print(f"\n[cyan]Starting API server on http://{host}:{port}[/cyan]")
    console.print(f"[dim]Docs: http://localhost:{port}/docs[/dim]\n")
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level="info",
    )


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
