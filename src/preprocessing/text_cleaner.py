"""
Text Pre-processing for Sentiment Analysis.

Provides a configurable pipeline that cleans and normalises raw text,
with special handling for social media content (hashtags, mentions,
emoji, URLs, slang contractions, etc.).

Usage::

    cleaner = TextCleaner()

    # Single text
    clean = cleaner.clean("Loooove this! 😍 #amazing @user https://t.co/xyz")
    # → "love this amazing"

    # Batch
    cleaned = cleaner.clean_batch(["Great! 👍", "Terrible 😤"])
"""

from __future__ import annotations

import html
import logging
import re
import unicodedata
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Optional dependencies ──────────────────────────────────

try:
    import contractions as contractions_lib
    _contractions_available = True
except ImportError:
    _contractions_available = False
    logger.debug("contractions library not found – contraction expansion disabled.")

try:
    import emoji as emoji_lib
    _emoji_available = True
except ImportError:
    _emoji_available = False
    logger.debug("emoji library not found – emoji conversion disabled.")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import TweetTokenizer, word_tokenize
    _nltk_available = True
except ImportError:
    _nltk_available = False
    logger.debug("NLTK not found – advanced tokenization/lemmatization disabled.")

try:
    from bs4 import BeautifulSoup
    _bs4_available = True
except ImportError:
    _bs4_available = False
    logger.debug("BeautifulSoup not found – HTML stripping will use regex fallback.")


# ─────────────────────────────────────────────────────────
# Regex Patterns
# ─────────────────────────────────────────────────────────

_URL_RE = re.compile(
    r"https?://\S+|www\.\S+", re.IGNORECASE
)
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")          # capture word after #
_REPEATED_CHARS_RE = re.compile(r"(.)\1{2,}")  # e.g. "loooove" → "loove"
_WHITESPACE_RE = re.compile(r"\s+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_NON_ASCII_PUNCT_RE = re.compile(r"[^\w\s.,!?;:'\"()\-]")


class TextCleaner:
    """
    Configurable text cleaning pipeline.

    Parameters
    ----------
    remove_urls : bool
        Strip HTTP/HTTPS URLs. Default True.
    remove_mentions : bool
        Strip @user mentions. Default True.
    remove_hashtags : bool
        Strip # symbol but keep the word. Default False.
    convert_emojis : bool
        Replace emoji characters with their text descriptions. Default True.
    expand_contractions : bool
        Expand English contractions ("can't" → "cannot"). Default True.
    lowercase : bool
        Convert all text to lowercase. Default True.
    strip_html : bool
        Remove HTML tags from text. Default True.
    lemmatize : bool
        Apply WordNet lemmatization (requires NLTK). Default False.
    remove_stopwords : bool
        Remove English stop words (requires NLTK). Default False.
    normalize_repeated : bool
        Compress repeated characters ("loooove" → "love"). Default True.
    min_length : int
        Minimum token length. Shorter tokens are dropped. Default 1.
    """

    def __init__(
        self,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_hashtags: bool = False,
        convert_emojis: bool = True,
        expand_contractions: bool = True,
        lowercase: bool = True,
        strip_html: bool = True,
        lemmatize: bool = False,
        remove_stopwords: bool = False,
        normalize_repeated: bool = True,
        min_length: int = 1,
    ):
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.convert_emojis = convert_emojis
        self.expand_contractions = expand_contractions
        self.lowercase = lowercase
        self.strip_html = strip_html
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords
        self.normalize_repeated = normalize_repeated
        self.min_length = min_length

        # NLTK setup
        self._lemmatizer = None
        self._stop_words = set()
        self._tokenizer = None
        if _nltk_available:
            self._ensure_nltk_resources()
            self._lemmatizer = WordNetLemmatizer()
            self._stop_words = set(stopwords.words("english"))
            self._tokenizer = TweetTokenizer(strip_handles=False, reduce_len=True)

    # ── Public API ────────────────────────────────────────

    def clean(self, text: str) -> str:
        """
        Apply the full cleaning pipeline to a single string.

        Args:
            text: Raw input text.

        Returns:
            Cleaned text string.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # 1. HTML
        text = self._strip_html(text)

        # 2. Unicode normalisation
        text = unicodedata.normalize("NFKC", text)

        # 3. Unescape HTML entities (&amp; → &)
        text = html.unescape(text)

        # 4. Emojis
        if self.convert_emojis and _emoji_available:
            text = emoji_lib.demojize(text, delimiters=(" ", " "))

        # 5. URLs
        if self.remove_urls:
            text = _URL_RE.sub(" ", text)

        # 6. Mentions
        if self.remove_mentions:
            text = _MENTION_RE.sub(" ", text)

        # 7. Hashtags
        if self.remove_hashtags:
            text = _HASHTAG_RE.sub(r"\1", text)   # keep word, drop #
        else:
            text = _HASHTAG_RE.sub(r"\1", text)   # always strip the # symbol

        # 8. Contractions
        if self.expand_contractions and _contractions_available:
            text = contractions_lib.fix(text)

        # 9. Lowercase
        if self.lowercase:
            text = text.lower()

        # 10. Repeated characters
        if self.normalize_repeated:
            text = _REPEATED_CHARS_RE.sub(r"\1\1", text)

        # 11. Tokenize
        tokens = self._tokenize(text)

        # 12. Lemmatize
        if self.lemmatize and _nltk_available and self._lemmatizer:
            tokens = [self._lemmatizer.lemmatize(t) for t in tokens]

        # 13. Stop words
        if self.remove_stopwords and _nltk_available:
            tokens = [t for t in tokens if t not in self._stop_words]

        # 14. Min length filter
        tokens = [t for t in tokens if len(t) >= self.min_length]

        return " ".join(tokens)

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Apply the cleaning pipeline to a list of texts."""
        return [self.clean(t) for t in texts]

    def clean_dataframe_column(self, series) -> "pd.Series":  # noqa: F821
        """Clean a pandas Series of texts in-place."""
        return series.apply(self.clean)

    # ── Internal Helpers ──────────────────────────────────

    def _strip_html(self, text: str) -> str:
        if not self.strip_html:
            return text
        if _bs4_available:
            return BeautifulSoup(text, "lxml").get_text(separator=" ")
        return _HTML_TAG_RE.sub(" ", text)

    def _tokenize(self, text: str) -> List[str]:
        if _nltk_available and self._tokenizer:
            return self._tokenizer.tokenize(text)
        # Fallback: simple whitespace split
        return _WHITESPACE_RE.split(text.strip())

    @staticmethod
    def _ensure_nltk_resources() -> None:
        """Download required NLTK data packages if missing."""
        resources = [
            ("tokenizers/punkt", "punkt"),
            ("tokenizers/punkt_tab", "punkt_tab"),
            ("corpora/stopwords", "stopwords"),
            ("corpora/wordnet", "wordnet"),
        ]
        for path, name in resources:
            try:
                nltk.data.find(path)
            except LookupError:
                logger.info("Downloading NLTK resource: %s", name)
                nltk.download(name, quiet=True)
