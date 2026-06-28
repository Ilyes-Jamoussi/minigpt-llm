"""From-scratch tokenizers and their JSON persistence.

A ``Tokenizer`` is anything that maps text to integer ids and back. The
character-level tokenizer lives here now; a byte-level BPE tokenizer is added
later and selected via ``TrainConfig.tokenizer_type`` without changing callers.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Protocol

# GPT-2-style pre-tokenization: split into contractions, words, numbers,
# punctuation, and whitespace runs. Merges never cross these boundaries.
_SPLIT_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""")
_BYTE_VOCAB_SIZE = 256


class Tokenizer(Protocol):
    """Maps text to a list of token ids and back; persists to a JSON file."""

    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...

    def save(self, path: Path) -> None: ...


class CharTokenizer:
    """Character-level tokenizer: one id per unique character in the corpus.

    Trivial and trains instantly, so it is the fast path used to validate the
    whole pipeline end to end before moving to BPE.
    """

    type_name = "char"

    def __init__(self, chars: list[str]) -> None:
        self._itos = list(chars)
        self._stoi = {c: i for i, c in enumerate(self._itos)}

    @property
    def vocab_size(self) -> int:
        return len(self._itos)

    @classmethod
    def train(cls, text: str) -> CharTokenizer:
        """Build the vocabulary from the sorted set of characters in ``text``."""
        return cls(sorted(set(text)))

    def encode(self, text: str) -> list[int]:
        try:
            return [self._stoi[c] for c in text]
        except KeyError as exc:
            raise ValueError(
                f"character {exc.args[0]!r} is not in the tokenizer vocabulary"
            ) from exc

    def decode(self, ids: list[int]) -> str:
        return "".join(self._itos[i] for i in ids)

    def save(self, path: Path) -> None:
        payload = {"type": self.type_name, "chars": self._itos}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CharTokenizer:
        chars = data["chars"]
        if not isinstance(chars, list):
            raise ValueError("invalid char tokenizer file: 'chars' must be a list")
        return cls([str(c) for c in chars])


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every adjacent occurrence of ``pair`` in ``ids`` with ``new_id``."""
    merged: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged


class BPETokenizer:
    """Byte-level Byte-Pair Encoding, trained from scratch (GPT-2 style).

    The base vocabulary is the 256 byte values; training repeatedly merges the
    most frequent adjacent token pair (counted over pre-tokenized words) into a
    new token. Because it is byte-level it can encode and decode any text.
    """

    type_name = "bpe"

    def __init__(self, merges: list[tuple[int, int]]) -> None:
        # Merge rank == new token id; lower id means higher priority at encode time.
        self._merges: dict[tuple[int, int], int] = {
            pair: _BYTE_VOCAB_SIZE + i for i, pair in enumerate(merges)
        }
        self._vocab = self._build_vocab(self._merges)
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return _BYTE_VOCAB_SIZE + len(self._merges)

    @staticmethod
    def _build_vocab(merges: dict[tuple[int, int], int]) -> dict[int, bytes]:
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(_BYTE_VOCAB_SIZE)}
        for (first, second), new_id in merges.items():
            vocab[new_id] = vocab[first] + vocab[second]
        return vocab

    @classmethod
    def train(cls, text: str, vocab_size: int) -> BPETokenizer:
        """Learn merges until the vocabulary reaches ``vocab_size`` (>= 256)."""
        if vocab_size < _BYTE_VOCAB_SIZE:
            raise ValueError(f"BPE vocab_size must be >= {_BYTE_VOCAB_SIZE}, got {vocab_size}")
        word_freqs: Counter[str] = Counter(_SPLIT_PATTERN.findall(text))
        words: dict[str, list[int]] = {w: list(w.encode("utf-8")) for w in word_freqs}
        merges: list[tuple[int, int]] = []
        for _ in range(vocab_size - _BYTE_VOCAB_SIZE):
            pair_counts: Counter[tuple[int, int]] = Counter()
            for word, seq in words.items():
                freq = word_freqs[word]
                for pair in pairwise(seq):
                    pair_counts[pair] += freq
            if not pair_counts:
                break
            best = max(pair_counts, key=lambda p: pair_counts[p])
            new_id = _BYTE_VOCAB_SIZE + len(merges)
            words = {w: _merge(seq, best, new_id) for w, seq in words.items()}
            merges.append(best)
        return cls(merges)

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        seq = list(chunk.encode("utf-8"))
        while len(seq) >= 2:
            best_pair: tuple[int, int] | None = None
            best_id = -1
            for pair in pairwise(seq):
                merge_id = self._merges.get(pair)
                if merge_id is not None and (best_pair is None or merge_id < best_id):
                    best_pair, best_id = pair, merge_id
            if best_pair is None:
                break
            seq = _merge(seq, best_pair, best_id)
        self._cache[chunk] = seq
        return seq

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in _SPLIT_PATTERN.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self._vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    def save(self, path: Path) -> None:
        payload = {"type": self.type_name, "merges": [[a, b] for a, b in self._merges]}
        path.write_text(json.dumps(payload) + "\n")

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> BPETokenizer:
        raw_merges = data["merges"]
        if not isinstance(raw_merges, list):
            raise ValueError("invalid bpe tokenizer file: 'merges' must be a list")
        return cls([(int(pair[0]), int(pair[1])) for pair in raw_merges])


def build_tokenizer(tokenizer_type: str, text: str, bpe_vocab_size: int) -> Tokenizer:
    """Train a fresh tokenizer of the requested type on ``text``."""
    if tokenizer_type == "char":
        return CharTokenizer.train(text)
    if tokenizer_type == "bpe":
        return BPETokenizer.train(text, bpe_vocab_size)
    raise ValueError(f"unknown tokenizer_type: {tokenizer_type!r}")


def load_tokenizer(path: Path) -> Tokenizer:
    """Load a saved tokenizer, dispatching on the ``type`` field in the file."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"invalid tokenizer file: {path}")
    type_name = raw.get("type")
    if type_name == "char":
        return CharTokenizer.from_dict(raw)
    if type_name == "bpe":
        return BPETokenizer.from_dict(raw)
    raise ValueError(f"unknown tokenizer type in {path}: {type_name!r}")
