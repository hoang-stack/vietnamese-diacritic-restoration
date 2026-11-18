"""Tokenised corpus cache: every split as integer ids, with sentence and article boundaries.

Tokenising 100k news articles in Python takes minutes, so it is done once, in parallel, and
stored as a few NumPy arrays. Token ids index ``types`` (every distinct LM token, most frequent
first), so a capped vocabulary is just "ids >= cap become <unk>".
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vidiac import data
from vidiac.text import lm_sentences


def _tokenize_chunk(texts: list[str]) -> list[list[list[str]]]:
    return [list(lm_sentences(t)) for t in texts]


@dataclass
class TokenCorpus:
    types: list[str]  # id -> token, most frequent first
    counts: np.ndarray  # frequency of each type
    ids: np.ndarray  # all tokens of all sentences, concatenated (int32)
    sent_offsets: np.ndarray  # sentence i = ids[sent_offsets[i]:sent_offsets[i+1]]
    article_offsets: np.ndarray  # article j = sentences article_offsets[j]:article_offsets[j+1]

    @property
    def n_articles(self) -> int:
        return len(self.article_offsets) - 1

    @property
    def n_tokens(self) -> int:
        return len(self.ids)

    def sentences(self, n_articles: int | None = None) -> list[np.ndarray]:
        last = self.article_offsets[n_articles if n_articles is not None else self.n_articles]
        o = self.sent_offsets
        return [self.ids[o[i] : o[i + 1]] for i in range(last)]

    def word_counts(self, n_articles: int | None = None) -> Counter:
        """Counts of alphabetic tokens (the restorable syllables) in the first n articles."""
        last_sent = self.article_offsets[n_articles if n_articles is not None else self.n_articles]
        upto = self.sent_offsets[last_sent]
        freq = np.bincount(self.ids[:upto], minlength=len(self.types))
        return Counter(
            {t: int(c) for t, c in zip(self.types, freq, strict=True) if c and t.isalpha()}
        )

    # --- persistence ------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            counts=self.counts,
            ids=self.ids,
            sent_offsets=self.sent_offsets,
            article_offsets=self.article_offsets,
        )
        path.with_suffix(".types.json").write_text(json.dumps(self.types, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> TokenCorpus:
        z = np.load(path)
        types = json.loads(path.with_suffix(".types.json").read_text())
        return cls(types, z["counts"], z["ids"], z["sent_offsets"], z["article_offsets"])

    @classmethod
    def build(cls, texts: list[str], workers: int = 8, chunk: int = 500) -> TokenCorpus:
        chunks = [texts[i : i + chunk] for i in range(0, len(texts), chunk)]
        with ProcessPoolExecutor(workers) as pool:
            tokenised = [doc for part in pool.map(_tokenize_chunk, chunks) for doc in part]
        counter = Counter(tok for doc in tokenised for sent in doc for tok in sent)
        types = [t for t, _ in counter.most_common()]
        index = {t: i for i, t in enumerate(types)}
        ids, sent_off, art_off = [], [0], [0]
        for doc in tokenised:
            for sent in doc:
                ids.extend(index[t] for t in sent)
                sent_off.append(len(ids))
            art_off.append(len(sent_off) - 1)
        counts = np.array([counter[t] for t in types], dtype=np.int64)
        return cls(types, counts, np.array(ids, np.int32), np.array(sent_off), np.array(art_off))


def load_corpus(split: str, data_dir: str | Path = "data", workers: int = 8) -> TokenCorpus:
    path = Path(data_dir) / "cache" / f"{split}.npz"
    if path.exists():
        return TokenCorpus.load(path)
    corpus = TokenCorpus.build(list(data.articles(split, data_dir)), workers=workers)
    corpus.save(path)
    return corpus
