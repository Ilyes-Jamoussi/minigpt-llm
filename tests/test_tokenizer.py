"""Tokenizer contract: round-trip, vocab size, and save/load."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tokenizer import BPETokenizer, CharTokenizer, build_tokenizer, load_tokenizer

SAMPLE = "Hello, world!\nThe quick brown fox jumps over the lazy dog.\n"
BPE_CORPUS = (
    "the cat sat on the mat. the cat ran. a dog sat on the log. "
    "the quick brown fox jumps over the lazy dog. " * 10
)


def test_char_tokenizer_roundtrip() -> None:
    tok = CharTokenizer.train(SAMPLE)
    assert tok.decode(tok.encode(SAMPLE)) == SAMPLE


def test_char_tokenizer_vocab_size_is_unique_chars() -> None:
    tok = CharTokenizer.train(SAMPLE)
    assert tok.vocab_size == len(set(SAMPLE))


def test_build_tokenizer_char_roundtrip() -> None:
    tok = build_tokenizer("char", SAMPLE, bpe_vocab_size=8192)
    assert tok.decode(tok.encode(SAMPLE)) == SAMPLE


def test_save_load_roundtrip(tmp_path: Path) -> None:
    tok = CharTokenizer.train(SAMPLE)
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    loaded = load_tokenizer(path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.decode(loaded.encode(SAMPLE)) == SAMPLE


def test_encode_unknown_character_raises() -> None:
    tok = CharTokenizer.train("abc")
    with pytest.raises(ValueError, match="not in the tokenizer vocabulary"):
        tok.encode("z")


def test_build_tokenizer_unknown_type_raises() -> None:
    with pytest.raises(ValueError, match="unknown tokenizer_type"):
        build_tokenizer("nope", SAMPLE, bpe_vocab_size=8192)


def test_bpe_roundtrip() -> None:
    tok = BPETokenizer.train(BPE_CORPUS, vocab_size=320)
    text = "the cat sat on the mat."
    assert tok.decode(tok.encode(text)) == text


def test_bpe_roundtrip_handles_unicode() -> None:
    tok = BPETokenizer.train(BPE_CORPUS, vocab_size=300)
    text = "caf\u00e9 na\u00efve \u2014 \U0001f600"  # accents, em dash, emoji
    assert tok.decode(tok.encode(text)) == text


def test_bpe_learns_merges_and_grows_vocab() -> None:
    tok = BPETokenizer.train(BPE_CORPUS, vocab_size=300)
    assert tok.vocab_size == 300
    # "the " is very frequent, so it should encode to fewer tokens than its bytes.
    assert len(tok.encode("the the the ")) < len(b"the the the ")


def test_bpe_save_load_roundtrip(tmp_path: Path) -> None:
    tok = BPETokenizer.train(BPE_CORPUS, vocab_size=320)
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    loaded = load_tokenizer(path)
    assert loaded.vocab_size == tok.vocab_size
    text = "the dog ran over the log."
    assert loaded.encode(text) == tok.encode(text)
    assert loaded.decode(loaded.encode(text)) == text


def test_bpe_train_rejects_tiny_vocab() -> None:
    with pytest.raises(ValueError, match="must be >="):
        BPETokenizer.train(BPE_CORPUS, vocab_size=100)
