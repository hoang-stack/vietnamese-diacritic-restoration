"""Build, save and load a complete restoration system from a tokenised training corpus."""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vidiac.classifier import (
    ContextClassifier,
    ContextScorer,
    StrippedVocab,
    build_examples,
    candidate_matrix,
)
from vidiac.corpus import TokenCorpus
from vidiac.lm import UNK, KneserNeyLM, Vocabulary
from vidiac.restore import (
    CandidateTable,
    ClassifierRestorer,
    Identity,
    LMRestorer,
    MostFrequent,
    Restorer,
)


@dataclass
class SystemConfig:
    n_articles: int | None = None  # train on the first n articles (None = all)
    vocab_size: int = 1 << 15  # 15 bits per token id -> 4-grams fit in one int64
    min_count: int = 2  # candidate forms must occur at least this often
    max_candidates: int = 16
    order: int = 4
    prune: int = 0  # drop order >= 3 n-grams with adjusted count below this (0 = keep all)
    beam: int = 16
    classifier: dict[str, Any] = field(default_factory=dict)  # ContextClassifier kwargs
    weight: float = 1.0  # classifier weight in the hybrid decoder


@dataclass
class System:
    config: SystemConfig
    vocab: Vocabulary
    candidates: CandidateTable
    lm: KneserNeyLM | None = None
    scorer: ContextScorer | None = None
    timings: dict[str, float] = field(default_factory=dict)

    def restorer(self, kind: str, **overrides) -> Restorer:
        cfg = self.config
        beam = overrides.get("beam", cfg.beam)
        weight = overrides.get("weight", cfg.weight)
        if kind == "identity":
            return Identity(self.candidates)
        if kind == "mostfreq":
            return MostFrequent(self.candidates)
        if kind == "lm":
            return LMRestorer(self.candidates, self.lm, self.vocab, beam=beam)
        if kind == "classifier":
            return ClassifierRestorer(self.candidates, self.scorer)
        if kind == "hybrid":
            return LMRestorer(
                self.candidates, self.lm, self.vocab, beam=beam, scorer=self.scorer, weight=weight
            )
        raise ValueError(f"unknown restorer {kind!r}")

    # --- persistence ------------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        meta = {"config": self.config.__dict__, "timings": self.timings}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @staticmethod
    def load(path: str | Path) -> System:
        # Pickle runs code on load: only open artifacts produced by `vidiac train`.
        with open(path, "rb") as fh:
            system = pickle.load(fh)
        if not isinstance(system, System):
            raise TypeError(f"{path} is not a vidiac System")
        return system


def build_system(
    corpus: TokenCorpus,
    config: SystemConfig,
    with_lm: bool = True,
    with_classifier: bool = True,
    log=print,
) -> System:
    cfg = config
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    # Vocabulary: the most frequent types among the tokens actually used for training (a prefix
    # or a subset of the corpus). Stable sorting keeps the full corpus's order on ties.
    n_art = cfg.n_articles if cfg.n_articles is not None else corpus.n_articles
    upto = corpus.sent_offsets[corpus.article_offsets[n_art]]
    freq = np.bincount(corpus.ids[:upto], minlength=len(corpus.types))
    order = np.argsort(-freq, kind="stable")
    types = [corpus.types[i] for i in order if freq[i]]
    vocab = Vocabulary(types[: cfg.vocab_size - 3])
    candidates = CandidateTable.build(
        corpus.word_counts(cfg.n_articles), cfg.min_count, cfg.max_candidates, allowed=vocab.stoi
    )
    timings["vocab_and_candidates"] = time.perf_counter() - t0
    system = System(cfg, vocab, candidates, timings=timings)

    if with_lm:
        t0 = time.perf_counter()
        type_to_lm = np.array([vocab.stoi.get(t, vocab.stoi[UNK]) for t in corpus.types], np.int32)
        sents = [type_to_lm[s] for s in corpus.sentences(cfg.n_articles)]
        system.lm = KneserNeyLM(cfg.order).fit(sents, len(vocab), prune=cfg.prune)
        del sents
        timings["lm"] = time.perf_counter() - t0
        log(f"LM order {cfg.order}: {system.lm.n_ngrams} n-grams, {system.lm.nbytes / 1e6:.0f} MB")

    if with_classifier:
        t0 = time.perf_counter()
        strip_vocab = StrippedVocab(vocab.itos)
        cmat = candidate_matrix(candidates, vocab, strip_vocab, cfg.max_candidates)
        examples = build_examples(corpus, cfg.n_articles, vocab, strip_vocab, cmat)
        log(f"classifier: {len(examples):,} ambiguous training tokens")
        clf = ContextClassifier(**cfg.classifier).fit(examples, log=log)
        del examples
        system.scorer = ContextScorer(clf, vocab, strip_vocab, cfg.max_candidates)
        timings["classifier"] = time.perf_counter() - t0
    return system
