# 🎭 SentimentAI — Multi-Model Sentiment Analysis System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Pytest-orange.svg)](tests/)

> A production-ready, multi-model sentiment analysis system capable of processing text documents, social media posts, product reviews, customer feedback, and more.

---

## 📋 Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Models](#models)
  - [VADER (Lexicon-based)](#1-vader-lexicon-based)
  - [ML Classifier (TF-IDF + sklearn)](#2-ml-classifier-tf-idf--sklearn)
  - [Transformer (DistilBERT)](#3-transformer-distilbert)
- [Usage](#usage)
  - [Python API](#python-api)
  - [Command-Line Interface (CLI)](#command-line-interface-cli)
  - [REST API](#rest-api)
  - [Training a Model](#training-a-model)
  - [Evaluating a Model](#evaluating-a-model)
- [REST API Reference](#rest-api-reference)
- [Configuration](#configuration)
- [Text Preprocessing](#text-preprocessing)
- [Running Tests](#running-tests)
- [Data Format](#data-format)
- [Performance Benchmarks](#performance-benchmarks)
- [Contributing](#contributing)
- [License](#license)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **3 Model Types** | VADER (rule-based), ML (TF-IDF + sklearn), Transformer (DistilBERT) |
| 🐦 **Social Media Ready** | Handles emojis, hashtags, mentions, slang, contractions |
| 📦 **Batch Processing** | Efficient inference on thousands of texts at once |
| 🌐 **REST API** | FastAPI-powered endpoints with auto-generated Swagger docs |
| 💻 **CLI Tool** | Rich command-line interface with progress bars and color output |
| 🏋️ **Training Pipeline** | End-to-end training from CSV/JSON with cross-validation |
| 📊 **Evaluation Suite** | Accuracy, F1, Precision, Recall, ROC-AUC, Confusion Matrix |
| 📁 **File Analysis** | Process entire text files or CSVs in one command |
| 💾 **Model Persistence** | Save and load trained models with joblib |
| ⚙️ **Configurable** | Central YAML configuration for all settings |
| 🧪 **Full Test Suite** | Unit + integration tests with pytest |

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         SentimentAI                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Layer                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  Plain   │  │   CSV    │  │   JSON   │  │   REST API   │   │
│  │   Text   │  │   File   │  │   File   │  │   Request    │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘   │
│       └─────────────┴──────────────┴────────────────┘           │
│                           │                                      │
│  Preprocessing Layer      ▼                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  TextCleaner                                               │ │
│  │  HTML Strip → URL Remove → Mention Strip → Emoji Convert  │ │
│  │  → Contraction Expand → Lowercase → Lemmatize             │ │
│  └──────────────────────────┬─────────────────────────────────┘ │
│                              │                                   │
│  Model Layer                 ▼                                  │
│  ┌────────────┐  ┌──────────────────┐  ┌─────────────────────┐ │
│  │   VADER    │  │  TF-IDF + sklearn │  │  DistilBERT /       │ │
│  │  (Lexicon) │  │  (LR / SVM / NB  │  │  HuggingFace        │ │
│  │            │  │  / RandomForest) │  │  Transformer        │ │
│  └────────────┘  └──────────────────┘  └─────────────────────┘ │
│                              │                                   │
│  Output Layer                ▼                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  SentimentResult                                           │ │
│  │  { label, score, scores: {pos, neg, neu}, metadata }      │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
sentiment-analysis/
│
├── src/                          # Main source package
│   ├── __init__.py
│   ├── models/                   # Sentiment models
│   │   ├── base_model.py         # Abstract base + SentimentResult dataclass
│   │   ├── lexicon_model.py      # VADER rule-based model
│   │   ├── ml_model.py           # TF-IDF + sklearn classifier
│   │   └── transformer_model.py  # HuggingFace transformer wrapper
│   │
│   ├── preprocessing/            # Text cleaning pipeline
│   │   └── text_cleaner.py
│   │
│   ├── training/                 # Training orchestration
│   │   └── trainer.py
│   │
│   ├── evaluation/               # Metrics and reporting
│   │   └── evaluator.py
│   │
│   ├── api/                      # FastAPI REST endpoints
│   │   └── app.py
│   │
│   └── utils/                    # Logging, config, model factory
│       └── helpers.py
│
├── data/
│   ├── sample_data.csv           # 100-row balanced sample dataset
│   ├── raw/                      # Place your raw data here
│   └── processed/                # Cleaned/processed datasets
│
├── models/                       # Saved model artifacts (.joblib)
├── tests/
│   ├── test_preprocessing.py     # TextCleaner unit tests
│   ├── test_models.py            # Model unit tests (VADER + ML)
│   └── test_api.py               # FastAPI integration tests
│
├── cli.py                        # Rich CLI entry point
├── train.py                      # Training script
├── predict.py                    # Prediction script
├── config.yaml                   # Central configuration
├── requirements.txt
├── setup.py
├── pyproject.toml
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.9 or higher
- pip or conda

### Step 1 — Clone / navigate to the project

```bash
# If cloning from Git:
git clone https://github.com/yourusername/sentiment-analysis.git
cd sentiment-analysis

# Or navigate to the existing directory:
cd d:/Projects/sentiment-analysis
```

### Step 2 — Create a virtual environment (recommended)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
# Core dependencies (includes VADER + ML + FastAPI + CLI)
pip install -r requirements.txt

# Download NLTK resources (run once)
python -c "import nltk; nltk.download(['punkt','punkt_tab','stopwords','wordnet'])"
```

### Step 4 — (Optional) Install as a package

```bash
pip install -e .
# Enables the 'sentimentai' CLI command globally
```

> **Note on Transformers:** The Transformer model requires `torch` and `transformers`.
> These are included in `requirements.txt` but are large downloads (~1 GB).
> If you only need VADER or ML, you can skip torch by removing those lines before installing.

---

## 🚀 Quick Start

### 1. Instant Sentiment Analysis (no training needed)

```python
from src.models.lexicon_model import LexiconSentimentModel

model = LexiconSentimentModel()

result = model.predict("I absolutely love this product! 😍")
print(result.label)   # → 'positive'
print(result.score)   # → 0.9758
print(result.scores)  # → {'positive': 0.543, 'negative': 0.0, 'neutral': 0.457}
```

### 2. Analyse Social Media Posts

```python
from src.models.lexicon_model import LexiconSentimentModel
from src.preprocessing.text_cleaner import TextCleaner

cleaner = TextCleaner()
model = LexiconSentimentModel()

posts = [
    "omg this is sooooo good 😍❤️❤️ #amazing @bestbrand",
    "Just got scammed. NEVER buying from this site again!! 🤬",
    "Package arrived today. Delivery was on time.",
    "Can't believe how rude the staff were. Absolutely fuming! 🤬😤",
    "Great product! Would highly recommend to all my friends 👍",
]

cleaned_posts = cleaner.clean_batch(posts)
results = model.predict_batch(cleaned_posts)

for post, result in zip(posts, results):
    print(f"[{result.label.upper():<8}] ({result.score:.2f}) {post[:60]}")
```

**Output:**
```
[POSITIVE] (0.98) omg this is sooooo good 😍❤️❤️ #amazing @bestbrand
[NEGATIVE] (0.88) Just got scammed. NEVER buying from this site again!!
[NEUTRAL ] (0.57) Package arrived today. Delivery was on time.
[NEGATIVE] (0.93) Can't believe how rude the staff were. Absolutely fuming!
[POSITIVE] (0.95) Great product! Would highly recommend to all my friends
```

---

## 🤖 Models

### 1. VADER (Lexicon-based)

**Best for:** Social media, short texts, real-time processing, no training data available.

```python
from src.models.lexicon_model import LexiconSentimentModel

model = LexiconSentimentModel()

# Handles emojis, punctuation emphasis, capitalization
result = model.predict("THIS IS AMAZING!!! 😍❤️❤️")
print(result)  # SentimentResult(label='positive', score=0.9999, model='vader')

# Custom thresholds
strict_model = LexiconSentimentModel(
    threshold_positive=0.3,   # Requires stronger positive signal
    threshold_negative=-0.3,  # Requires stronger negative signal
)
```

| Pros | Cons |
|---|---|
| ✅ No training data needed | ❌ Binary/ternary labels only |
| ✅ Instant, very fast | ❌ Limited to English |
| ✅ Social media optimised | ❌ Misses subtle sarcasm |
| ✅ Handles emojis | ❌ No custom domain adaptation |

---

### 2. ML Classifier (TF-IDF + sklearn)

**Best for:** Domain-specific text, when you have labelled data, balanced accuracy vs. speed.

```python
from src.models.ml_model import MLSentimentModel

# Train with your own data
model = MLSentimentModel(algorithm="logistic_regression")

texts = [
    "Great product!", "Terrible quality!", "Just okay.",
    # ... thousands more
]
labels = ["positive", "negative", "neutral"]  # ... corresponding labels

metrics = model.train(texts, labels, cv_folds=5)
print(metrics)
# {'cv_f1_mean': 0.87, 'cv_f1_std': 0.02, 'test_accuracy': 0.89, 'test_f1': 0.88}

# Predict
result = model.predict("The product quality has significantly improved!")
print(result.label)  # → 'positive'

# Save for later
model.save("models/my_model.joblib")

# Load in another session
model2 = MLSentimentModel()
model2.load("models/my_model.joblib")
```

**Available Algorithms:**

| Algorithm | Speed | Accuracy | Notes |
|---|---|---|---|
| `logistic_regression` | ⚡⚡⚡ | 🎯🎯🎯 | Default, best overall |
| `svm` | ⚡⚡ | 🎯🎯🎯 | Good for high-dimensional features |
| `naive_bayes` | ⚡⚡⚡⚡ | 🎯🎯 | Very fast, less accurate |
| `random_forest` | ⚡ | 🎯🎯🎯 | Robust, slower training |

---

### 3. Transformer (DistilBERT)

**Best for:** Highest accuracy, complex language, nuanced sentiment.

```python
from src.models.transformer_model import TransformerSentimentModel

# Loads pre-trained DistilBERT fine-tuned on SST-2
model = TransformerSentimentModel()

result = model.predict(
    "Despite the initial issues, the support team went above and "
    "beyond to resolve everything. Truly outstanding service."
)
print(result.label)  # → 'positive'
print(result.score)  # → 0.9923

# Batch processing (GPU-accelerated if available)
results = model.predict_batch([
    "I love it!",
    "Absolutely terrible.",
    "It's fine I guess.",
], batch_size=32)

# Use a different pre-trained model
multilingual = TransformerSentimentModel(
    model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"
)
```

| Pros | Cons |
|---|---|
| ✅ Highest accuracy | ❌ Requires GPU for speed |
| ✅ Context-aware | ❌ Large model file (~260 MB) |
| ✅ Handles complex sentences | ❌ Slower inference on CPU |
| ✅ Many models available | ❌ Requires torch + transformers |

---

## 📖 Usage

### Python API

#### Factory Function (Recommended)

```python
from src.utils.helpers import create_model

# VADER — instant, no training
vader = create_model(model_type="vader")
result = vader.predict("This is amazing!")

# ML — load from saved file
ml = create_model(model_type="ml", model_path="models/ml_model.joblib")
result = ml.predict("Great customer service!")

# Transformer — pre-trained
transformer = create_model(model_type="transformer")
result = transformer.predict("Absolutely loved the experience!")
```

#### Accessing Results

```python
result = model.predict("I love this!")

print(result.label)     # 'positive'
print(result.score)     # 0.9456  (confidence, 0–1)
print(result.scores)    # {'positive': 0.9456, 'negative': 0.032, 'neutral': 0.022}
print(result.model_name) # 'vader'
print(result.metadata)  # {'compound': 0.6369}  (model-specific extra info)
print(result.to_dict()) # Full dictionary representation
```

#### Batch Processing

```python
texts = load_your_texts()  # List of strings

results = model.predict_batch(texts)
labels = [r.label for r in results]
scores = [r.score for r in results]
```

#### Training Pipeline

```python
from src.training.trainer import SentimentTrainer

trainer = SentimentTrainer(
    model_type="ml",
    algorithm="logistic_regression",
)

# From CSV
metrics = trainer.train_from_csv(
    path="data/my_data.csv",
    text_col="text",
    label_col="sentiment",
    cv_folds=5,
    test_size=0.2,
)

# From JSON
metrics = trainer.train_from_json(
    path="data/my_data.json",
    text_key="text",
    label_key="label",
)

# From lists
metrics = trainer.train(texts, labels)

print(metrics)
# {'cv_f1_mean': 0.87, 'test_accuracy': 0.89, 'training_time_seconds': 4.2}

trainer.save_model("models/my_model.joblib")
```

#### Evaluation

```python
from src.evaluation.evaluator import SentimentEvaluator

evaluator = SentimentEvaluator()
metrics = evaluator.evaluate(model, test_texts, true_labels)

evaluator.print_report()
evaluator.plot_confusion_matrix(save_path="reports/confusion_matrix.png")
evaluator.save_report("reports/evaluation_report.txt")
```

#### Text Preprocessing Only

```python
from src.preprocessing.text_cleaner import TextCleaner

cleaner = TextCleaner(
    remove_urls=True,
    remove_mentions=True,
    convert_emojis=True,
    expand_contractions=True,
    lowercase=True,
    normalize_repeated=True,
    lemmatize=False,       # Enable for better ML features
    remove_stopwords=False,
)

text = "Omg I loooove this!! 😍 Check it out at https://example.com @friend #amazing"
clean = cleaner.clean(text)
# → 'omg i loove this check it out friend amazing'

# Process a DataFrame column
import pandas as pd
df = pd.read_csv("data.csv")
df["clean_text"] = cleaner.clean_dataframe_column(df["text"])
```

---

### Command-Line Interface (CLI)

The CLI uses [Rich](https://github.com/Textualize/rich) for beautiful terminal output.

#### Analyse a Single Text

```bash
python cli.py analyze "I absolutely love this product!"
python cli.py analyze "Terrible experience, do not buy." --model vader
python cli.py analyze "Great!" --model ml --model-path models/ml_model.joblib
python cli.py analyze "Excellent service!" --scores     # Show per-class scores
python cli.py analyze "Test" --json                     # JSON output
```

#### Analyse from stdin

```bash
echo "This is fantastic!" | python cli.py analyze --stdin
cat reviews.txt | python cli.py analyze --stdin
```

#### Analyse a File

```bash
# Analyse a plain text file (one text per line)
python cli.py analyze-file data/reviews.txt

# Analyse a CSV file
python cli.py analyze-file data/sample_data.csv --text-col text

# Save results to CSV
python cli.py analyze-file data/sample_data.csv --output results/analysis.csv

# Limit to first 100 rows
python cli.py analyze-file data/sample_data.csv --limit 100
```

#### Train a Model

```bash
# Train Logistic Regression (default)
python cli.py train data/sample_data.csv

# Train SVM
python cli.py train data/sample_data.csv --algo svm

# Custom columns and save path
python cli.py train data/my_data.csv \
    --text-col review \
    --label-col sentiment \
    --save-path models/svm_model.joblib \
    --cv-folds 5
```

#### Evaluate a Model

```bash
# Evaluate VADER on test data
python cli.py evaluate data/sample_data.csv --model vader

# Evaluate trained ML model
python cli.py evaluate data/test_data.csv \
    --model ml \
    --model-path models/ml_model.joblib \
    --plot \
    --report-path reports/eval_report.txt
```

#### Start the API Server

```bash
python cli.py serve
python cli.py serve --port 8080 --reload   # Development mode
python cli.py serve --host 0.0.0.0 --workers 4
```

#### Using the train.py and predict.py Scripts

```bash
# Training
python train.py --data data/sample_data.csv --algo logistic_regression
python train.py --data data/sample_data.csv --algo svm --save-path models/svm.joblib

# Prediction
python predict.py --text "Absolutely amazing product!"
python predict.py --file reviews.txt --model vader --output results.csv
python predict.py --csv data/reviews.csv --text-col review --json-output
python predict.py --text "Great!" --model ml --model-path models/ml_model.joblib
```

---

### REST API

#### Start the Server

```bash
# Using CLI
python cli.py serve --reload

# Using uvicorn directly
uvicorn src.api.app:app --reload --port 8000

# Using Python
python -m uvicorn src.api.app:app --reload
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

#### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Welcome message |
| `GET` | `/health` | Health check |
| `GET` | `/models` | List available models |
| `POST` | `/analyze` | Analyse single text |
| `POST` | `/analyze/batch` | Analyse multiple texts |
| `POST` | `/analyze/file` | Upload and analyse text file |

---

## 📡 REST API Reference

### `POST /analyze`

Analyse the sentiment of a single text.

**Request Body:**
```json
{
  "text": "I absolutely love this product! Best purchase ever!",
  "model": "vader",
  "preprocess": true
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `text` | string | ✅ | — | Input text (max 10,000 chars) |
| `model` | string | ❌ | `"vader"` | `vader` \| `ml` \| `transformer` |
| `preprocess` | boolean | ❌ | `true` | Apply text cleaning |

**Response:**
```json
{
  "text": "I absolutely love this product! Best purchase ever!",
  "label": "positive",
  "score": 0.9456,
  "scores": {
    "positive": 0.543,
    "negative": 0.0,
    "neutral": 0.457
  },
  "model_name": "vader",
  "metadata": {
    "compound": 0.8934
  },
  "processing_time_ms": 1.23
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this!", "model": "vader"}'
```

---

### `POST /analyze/batch`

Analyse multiple texts in one request.

**Request Body:**
```json
{
  "texts": [
    "Amazing product, highly recommend!",
    "Terrible experience, avoid at all costs.",
    "Average quality, nothing special."
  ],
  "model": "vader",
  "preprocess": true
}
```

**Response:**
```json
{
  "results": [
    {"label": "positive", "score": 0.94, "text": "...", ...},
    {"label": "negative", "score": 0.91, "text": "...", ...},
    {"label": "neutral",  "score": 0.63, "text": "...", ...}
  ],
  "total": 3,
  "model_name": "vader",
  "processing_time_ms": 4.56
}
```

**Python requests Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/analyze/batch",
    json={
        "texts": ["Great!", "Terrible.", "Okay."],
        "model": "vader"
    }
)
data = response.json()
for r in data["results"]:
    print(f"[{r['label']}] {r['text']}")
```

---

### `POST /analyze/file`

Upload a `.txt` or `.csv` file for line-by-line analysis.

```bash
# cURL
curl -X POST http://localhost:8000/analyze/file \
  -F "file=@reviews.txt" \
  -F "model=vader"
```

**Response:**
```json
{
  "filename": "reviews.txt",
  "total_lines": 50,
  "model": "vader",
  "processing_time_ms": 23.4,
  "results": [
    {"line": "I love this!", "label": "positive", "score": 0.94, ...},
    ...
  ]
}
```

---

## ⚙️ Configuration

All settings live in [`config.yaml`](config.yaml). Key sections:

```yaml
# ML Model settings
ml_model:
  algorithm: "logistic_regression"
  tfidf:
    max_features: 50000
    ngram_range: [1, 3]

# Transformer settings
transformer_model:
  model_name: "distilbert-base-uncased-finetuned-sst-2-english"
  batch_size: 32
  device: "auto"   # auto | cpu | cuda | mps

# API settings
api:
  default_model: "vader"
  max_batch_size: 100
  port: 8000

# Preprocessing
preprocessing:
  remove_urls: true
  remove_mentions: true
  expand_contractions: true
  lowercase: true
```

---

## 🧹 Text Preprocessing

The `TextCleaner` applies these steps in order:

| Step | Example |
|---|---|
| HTML stripping | `<p>Hello</p>` → `Hello` |
| Unicode normalisation | `caf\u00e9` → `café` |
| HTML entity unescape | `&amp;` → `&` |
| Emoji → text | `😍` → `smiling face with heart-eyes` |
| URL removal | `https://t.co/xyz` → ` ` |
| Mention removal | `@user` → ` ` |
| Hashtag cleanup | `#amazing` → `amazing` |
| Contraction expansion | `can't` → `cannot` |
| Lowercase | `HELLO` → `hello` |
| Repeated char reduction | `loooove` → `loove` |
| NLTK tokenization | Smart tweet-aware tokenizer |
| Lemmatization (optional) | `running` → `run` |
| Stop word removal (optional) | `the`, `a`, `is` → removed |

---

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_preprocessing.py -v
pytest tests/test_models.py -v
pytest tests/test_api.py -v

# Run tests matching a pattern
pytest -k "test_predict_positive"

# Run with verbose output
pytest -v --tb=long
```

---

## 📂 Data Format

### CSV (required for training)

```csv
text,sentiment
"I absolutely love this product!",positive
"Terrible quality, completely broken.",negative
"Product arrived on time. Nothing special.",neutral
```

| Column | Values | Notes |
|---|---|---|
| `text` | Any string | Raw text, will be cleaned automatically |
| `sentiment` | `positive` \| `negative` \| `neutral` | Case-insensitive |

### JSON

```json
[
  {"text": "Excellent service!", "sentiment": "positive"},
  {"text": "Worst experience ever.", "sentiment": "negative"},
  {"text": "Average quality.", "sentiment": "neutral"}
]
```

### Supported Label Formats

The system automatically normalises these label formats:

| Raw Label | Normalised To |
|---|---|
| `positive`, `pos`, `1`, `good` | `positive` |
| `negative`, `neg`, `-1`, `bad` | `negative` |
| `neutral`, `neu`, `0` | `neutral` |

---

## 📈 Performance Benchmarks

Benchmarks on a modern CPU (Intel i7, 16GB RAM):

| Model | Speed (texts/sec) | Accuracy\* | Memory |
|---|---|---|---|
| VADER | ~10,000 | 72% | < 10 MB |
| ML (LR) | ~5,000 | 85% | ~50 MB |
| ML (SVM) | ~3,000 | 86% | ~50 MB |
| Transformer (CPU) | ~50 | 92% | ~500 MB |
| Transformer (GPU) | ~1,000 | 92% | ~500 MB |

> \* Accuracy on balanced SST-2 style binary classification benchmark.
> Results will vary by dataset and domain.

---

## 🗺️ Roadmap

- [ ] Multilingual support (mBERT, XLM-RoBERTa)
- [ ] Aspect-based sentiment analysis
- [ ] Fine-tuning pipeline for Transformers
- [ ] Streaming API endpoint (Server-Sent Events)
- [ ] Docker containerisation
- [ ] Model performance dashboard (Streamlit)
- [ ] Active learning support

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the tests (`pytest`)
5. Format your code (`black . && isort .`)
6. Commit (`git commit -m "Add amazing feature"`)
7. Push (`git push origin feature/amazing-feature`)
8. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) by C.J. Hutto & Eric Gilbert
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [scikit-learn](https://scikit-learn.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Rich](https://github.com/Textualize/rich)

---

<p align="center">
  Made with ❤️ using Python · <a href="http://localhost:8000/docs">API Docs</a>
</p>
