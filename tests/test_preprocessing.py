"""
Unit tests for TextCleaner preprocessing.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.preprocessing.text_cleaner import TextCleaner


class TestTextCleaner:
    """Test suite for TextCleaner."""

    @pytest.fixture
    def cleaner(self):
        return TextCleaner()

    @pytest.fixture
    def strict_cleaner(self):
        """Cleaner with all options enabled."""
        return TextCleaner(
            remove_urls=True,
            remove_mentions=True,
            remove_hashtags=True,
            lowercase=True,
            normalize_repeated=True,
        )

    # ── Basic Functionality ───────────────────────────────

    def test_clean_simple_text(self, cleaner):
        result = cleaner.clean("Hello World!")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_clean_empty_string(self, cleaner):
        assert cleaner.clean("") == ""

    def test_clean_whitespace_only(self, cleaner):
        assert cleaner.clean("   ") == ""

    def test_clean_none_like(self, cleaner):
        # Should not raise
        result = cleaner.clean(123)  # type: ignore
        assert result == ""

    def test_lowercase(self, cleaner):
        result = cleaner.clean("Hello WORLD")
        assert result == result.lower()

    # ── URL Handling ──────────────────────────────────────

    def test_remove_url_http(self, cleaner):
        text = "Check this out: https://www.example.com/some/path?q=1"
        result = cleaner.clean(text)
        assert "http" not in result
        assert "example.com" not in result

    def test_remove_url_www(self, cleaner):
        text = "Visit www.google.com for more info"
        result = cleaner.clean(text)
        assert "www" not in result

    def test_url_preserved_when_disabled(self):
        c = TextCleaner(remove_urls=False)
        text = "See https://example.com for details"
        result = c.clean(text)
        assert "example.com" in result

    # ── Mention Handling ──────────────────────────────────

    def test_remove_mention(self, cleaner):
        text = "Thanks @johndoe for your help!"
        result = cleaner.clean(text)
        assert "@johndoe" not in result
        assert "@" not in result

    def test_multiple_mentions(self, cleaner):
        text = "@alice and @bob both liked this post"
        result = cleaner.clean(text)
        assert "@" not in result

    def test_mention_preserved_when_disabled(self):
        c = TextCleaner(remove_mentions=False)
        result = c.clean("Hello @user!")
        assert "user" in result

    # ── Hashtag Handling ──────────────────────────────────

    def test_hashtag_symbol_removed(self, cleaner):
        text = "This is #amazing!"
        result = cleaner.clean(text)
        assert "#" not in result
        assert "amazing" in result

    def test_multiple_hashtags(self, cleaner):
        text = "#love #peace #happiness today"
        result = cleaner.clean(text)
        assert "#" not in result

    # ── Repeated Characters ───────────────────────────────

    def test_normalize_repeated(self, cleaner):
        result = cleaner.clean("I loooooove this!!")
        # "loooooove" → "loove" (max 2 repeats)
        assert "ooooo" not in result

    # ── Batch Processing ──────────────────────────────────

    def test_clean_batch_returns_list(self, cleaner):
        texts = ["Hello!", "World!", "Test."]
        results = cleaner.clean_batch(texts)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_clean_batch_preserves_order(self, cleaner):
        texts = ["first", "second", "third"]
        results = cleaner.clean_batch(texts)
        assert results[0].startswith("first")
        assert results[1].startswith("second")
        assert results[2].startswith("third")

    def test_clean_batch_empty_list(self, cleaner):
        assert cleaner.clean_batch([]) == []

    # ── HTML ──────────────────────────────────────────────

    def test_strip_html_tags(self, cleaner):
        text = "<p>This is <b>bold</b> text.</p>"
        result = cleaner.clean(text)
        assert "<" not in result
        assert ">" not in result
        assert "bold" in result

    def test_unescape_html_entities(self, cleaner):
        text = "This &amp; that are &lt;great&gt;"
        result = cleaner.clean(text)
        assert "&amp;" not in result


class TestTextCleanerConfig:
    """Test different cleaner configurations."""

    def test_no_lowercase(self):
        c = TextCleaner(lowercase=False)
        result = c.clean("Hello World CAPS")
        assert "Hello" in result or "CAPS" in result

    def test_minimal_cleaner(self):
        c = TextCleaner(
            remove_urls=False,
            remove_mentions=False,
            remove_hashtags=False,
            convert_emojis=False,
            expand_contractions=False,
            lowercase=False,
        )
        text = "Hello @world! #test"
        result = c.clean(text)
        assert "@world" in result or "world" in result
