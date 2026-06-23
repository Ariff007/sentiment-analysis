"""
Utility helpers: logging setup, config loading, model factory.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# ─────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
) -> None:
    """
    Configure the root logger with a console handler and optional rotating file handler.

    Args:
        level:        Logging level string (DEBUG, INFO, WARNING, ERROR).
        log_file:     If provided, logs are also written to this rotating file.
        max_bytes:    Maximum bytes per log file before rotation.
        backup_count: Number of backup log files to keep.
    """
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        root.addHandler(ch)

    # Rotating file handler
    if log_file:
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)


# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────

_DEFAULT_CONFIG_PATH = Path(__file__).parents[2] / "config.yaml"


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load and return the YAML configuration file.

    Args:
        path: Path to the YAML config file. Defaults to ``config.yaml``
              at the project root.

    Returns:
        Configuration dictionary.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        logging.warning("Config file not found at '%s'. Using defaults.", config_path)
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg or {}


# ─────────────────────────────────────────────────────────
# Model Factory
# ─────────────────────────────────────────────────────────

def create_model(
    model_type: str = "vader",
    algorithm: str = "logistic_regression",
    model_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
):
    """
    Factory function to create and optionally load a sentiment model.

    Args:
        model_type: One of ``"vader"``, ``"ml"``, ``"transformer"``.
        algorithm:  Algorithm for ML models (e.g. ``"logistic_regression"``).
        model_path: If provided, loads a saved model from this path.
        config:     Optional configuration dict.

    Returns:
        A ready-to-use :class:`BaseSentimentModel` instance.
    """
    from ..models.lexicon_model import LexiconSentimentModel
    from ..models.ml_model import MLSentimentModel
    from ..models.transformer_model import TransformerSentimentModel

    model_type = model_type.lower()

    if model_type == "vader":
        return LexiconSentimentModel(config=config)

    if model_type == "ml":
        m = MLSentimentModel(algorithm=algorithm, config=config)
        if model_path:
            m.load(model_path)
        return m

    if model_type == "transformer":
        hf_name = (config or {}).get(
            "model_name", "distilbert-base-uncased-finetuned-sst-2-english"
        )
        m = TransformerSentimentModel(model_name=hf_name, config=config)
        if model_path:
            m.load(model_path)
        return m

    raise ValueError(
        f"Unknown model_type '{model_type}'. Choose from: vader, ml, transformer."
    )
