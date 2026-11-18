"""Scoring restored text against the original.

Stripping never changes where words start and end, so the gold text and a system's output
tokenise into the same word sequence and can be compared position by position. Scores are kept
per article, so confidence intervals can be bootstrapped over articles (the unit that was
sampled), not over words, which are strongly correlated within an article.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from vidiac.restore import CandidateTable
from vidiac.text import sentences, strip_diacritics, tokenize


@dataclass
class ArticleScore:
    words: int = 0
    correct: int = 0
    ambiguous: int = 0  # words whose stripped form has >= 2 candidates
    ambiguous_correct: int = 0
    sentences: int = 0
    sentences_correct: int = 0


@dataclass
class EvalResult:
    articles: list[ArticleScore]
    confusions: Counter = field(default_factory=Counter)  # (gold, predicted) lower-case pairs

    def _arr(self, attr: str) -> np.ndarray:
        return np.array([getattr(a, attr) for a in self.articles], dtype=np.float64)

    @property
    def summary(self) -> dict[str, float]:
        w, c = self._arr("words").sum(), self._arr("correct").sum()
        aw, ac = self._arr("ambiguous").sum(), self._arr("ambiguous_correct").sum()
        sw, sc = self._arr("sentences").sum(), self._arr("sentences_correct").sum()
        return {
            "words": int(w),
            "accuracy": c / w,
            "ambiguous_share": aw / w,
            "ambiguous_accuracy": ac / aw if aw else float("nan"),
            "sentence_accuracy": sc / sw,
        }

    def bootstrap(self, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
        """Percentile CI of word accuracy, resampling articles (ratio of sums estimator)."""
        rng = np.random.default_rng(seed)
        w, c = self._arr("words"), self._arr("correct")
        idx = rng.integers(0, len(w), size=(n_boot, len(w)))
        acc = c[idx].sum(axis=1) / w[idx].sum(axis=1)
        lo, hi = np.quantile(acc, [alpha / 2, 1 - alpha / 2])
        return {"accuracy": float(c.sum() / w.sum()), "ci_low": float(lo), "ci_high": float(hi)}


def score_article(
    gold: str, predicted: str, candidates: CandidateTable
) -> tuple[ArticleScore, Counter]:
    score, conf = ArticleScore(), Counter()
    for sent in sentences(tokenize(gold)):
        if not any(t.kind == "word" for t in sent):
            continue
        start, end = sent[0].start, sent[-1].end
        pred_tokens = [t for t in tokenize(predicted[start:end]) if t.kind == "word"]
        gold_tokens = [t for t in sent if t.kind == "word"]
        if len(pred_tokens) != len(gold_tokens):
            raise ValueError("prediction does not align with the gold text")
        all_ok = True
        for g, p in zip(gold_tokens, pred_tokens, strict=True):
            ok = g.text == p.text
            amb = len(candidates.forms(strip_diacritics(g.text.lower()))) >= 2
            score.words += 1
            score.correct += ok
            score.ambiguous += amb
            score.ambiguous_correct += ok and amb
            if not ok:
                all_ok = False
                conf[(g.text.lower(), p.text.lower())] += 1
        score.sentences += 1
        score.sentences_correct += all_ok
    return score, conf


def evaluate(restorer, golds: list[str], candidates: CandidateTable) -> EvalResult:
    inputs = [strip_diacritics(g) for g in golds]
    if hasattr(restorer, "restore_many"):
        outputs = restorer.restore_many(inputs)
    else:
        outputs = [restorer.restore(x) for x in inputs]
    return score_outputs(golds, outputs, candidates)


def score_outputs(golds: list[str], outputs: list[str], candidates: CandidateTable) -> EvalResult:
    result = EvalResult([])
    for gold, out in zip(golds, outputs, strict=True):
        score, conf = score_article(gold, out, candidates)
        result.articles.append(score)
        result.confusions.update(conf)
    return result


def paired_bootstrap(a: EvalResult, b: EvalResult, n_boot: int = 2000, seed: int = 0) -> dict:
    """CI of accuracy(a) - accuracy(b) on the same articles, resampling articles."""
    rng = np.random.default_rng(seed)
    w = a._arr("words")
    ca, cb = a._arr("correct"), b._arr("correct")
    idx = rng.integers(0, len(w), size=(n_boot, len(w)))
    ww = w[idx].sum(axis=1)
    diff = (ca[idx].sum(axis=1) - cb[idx].sum(axis=1)) / ww
    lo, hi = np.quantile(diff, [0.025, 0.975])
    p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
    return {
        "diff": float((ca.sum() - cb.sum()) / w.sum()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(1.0, p)),
    }
