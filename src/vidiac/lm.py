"""Syllable n-gram language model with interpolated modified Kneser-Ney smoothing.

Everything is NumPy arrays, no Python dicts of n-grams:

* every token gets an integer id; an n-gram is packed into one int64 key
  (``bits`` bits per id, so ``order * bits <= 63``);
* counting is ``np.unique`` over all sliding windows of the corpus;
* lookups are ``np.searchsorted`` over the sorted key arrays.

Estimation follows Chen & Goodman (1998) and KenLM (Heafield, 2011): the highest order uses raw
counts, lower orders use *continuation* counts (how many distinct words precede the n-gram),
except n-grams that start with ``<s>``, which keep raw counts because nothing can precede them.
Three discounts D1, D2, D3+ per order are estimated from count-of-counts. The model is stored
in back-off form: log10-free natural-log probabilities for every seen n-gram, plus a back-off
weight for every seen context, so that

    log P(w | h) = log P*(h w)                     if h w was seen
                 = log gamma(h) + log P(w | h[1:])  otherwise.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import numpy as np

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


class Vocabulary:
    """Token <-> id mapping. Ids 0-2 are reserved for <s>, </s> and <unk>."""

    def __init__(self, tokens: list[str]):
        self.itos = [BOS, EOS, UNK, *tokens]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(cls, counts: Counter, max_size: int, min_count: int = 1) -> Vocabulary:
        keep = [t for t, c in counts.most_common(max_size - 3) if c >= min_count]
        return cls(keep)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: Iterable[str]) -> np.ndarray:
        unk = self.stoi[UNK]
        return np.fromiter((self.stoi.get(t, unk) for t in tokens), dtype=np.int64)


def _padded(ids_per_sentence: list[np.ndarray], order: int) -> tuple[np.ndarray, np.ndarray]:
    """One flat sequence with (order-1) <s> before and </s> after every sentence, plus the
    positions where a valid n-gram can end (a real token or </s>, never padding)."""
    # Memory matters here (60M tokens): int32 everywhere, temporaries freed as soon as possible.
    lengths = np.fromiter((len(s) for s in ids_per_sentence), np.int64, len(ids_per_sentence))
    if not len(lengths):
        return np.zeros(0, np.int32), np.zeros(0, np.int32)
    block = lengths + order  # (order-1) pads + tokens + </s>
    block_start = np.concatenate([[0], np.cumsum(block)[:-1]])
    seq = np.zeros(int(block.sum()), np.int32)  # zeros are <s> padding
    first_tok = block_start + order - 1
    tok_pos = np.arange(int(lengths.sum())) + np.repeat(
        first_tok - np.cumsum(lengths) + lengths, lengths
    )
    seq[tok_pos] = np.concatenate(ids_per_sentence)
    del tok_pos
    seq[first_tok + lengths] = 1  # </s>
    # Real tokens and </s> are exactly the non-zero entries (<s> is id 0 and only pads).
    ends = np.flatnonzero(seq).astype(np.int32)
    return seq, ends


class KneserNeyLM:
    def __init__(self, order: int = 4):
        if not 2 <= order <= 6:
            raise ValueError("order must be between 2 and 6")
        self.order = order

    # --- key packing ------------------------------------------------------------------------
    def _pack(self, cols: np.ndarray) -> np.ndarray:
        key = np.zeros(cols.shape[0], dtype=np.int64)
        for j in range(cols.shape[1]):
            key = (key << self.bits) | cols[:, j]
        return key

    # --- estimation -------------------------------------------------------------------------
    def fit(
        self, ids_per_sentence: list[np.ndarray], vocab_size: int, prune: int = 0
    ) -> KneserNeyLM:
        """Estimate the model. ``prune`` > 1 drops n-grams of order >= 3 seen fewer times."""
        self.prune = prune
        self.vocab_size = vocab_size
        self.bits = max(1, int(np.ceil(np.log2(vocab_size))))
        if self.bits * self.order > 63:
            raise ValueError(f"vocab of {vocab_size} is too large for order {self.order}")
        N = self.order
        keys, adj = self._adjusted_counts(ids_per_sentence)
        self.discounts_ = {n: self._discounts(adj[n]) for n in range(1, N + 1)}
        self._estimate(keys, adj)
        if prune > 1:
            self._prune(adj, prune)
        return self

    def _adjusted_counts(self, ids_per_sentence):
        """Unique n-gram keys and their KN adjusted counts for every order (index 1..N).

        The padded corpus lives only inside this method, so its memory is released as soon
        as counting is done.
        """
        N = self.order
        seq, ends = _padded(ids_per_sentence, N)

        def window_keys(n: int) -> np.ndarray:
            # Packed key of the n tokens ending at every valid position, without ever
            # materialising a (positions x n) matrix.
            key = np.zeros(len(ends), np.int64)
            for j in range(n - 1, -1, -1):
                key = (key << self.bits) | seq[ends - j].astype(np.int64)
            return key

        keys: list[np.ndarray] = [None] * (N + 1)
        adj: list[np.ndarray] = [None] * (N + 1)
        keys[N], adj[N] = np.unique(window_keys(N), return_counts=True)
        for n in range(N - 1, 0, -1):
            # Every n-gram is the suffix of a window. Raw count = number of windows it ends;
            # continuation count = number of distinct (n+1)-grams it is the suffix of.
            uniq, raw = np.unique(window_keys(n), return_counts=True)
            ext_suffix = keys[n + 1] & ((1 << (self.bits * n)) - 1)
            cont = np.bincount(np.searchsorted(uniq, ext_suffix), minlength=len(uniq))
            first = uniq >> (self.bits * (n - 1))  # first token of the n-gram
            keys[n] = uniq
            adj[n] = np.where(first == 0, raw, cont)  # n-grams starting with <s>: raw count
        return keys, adj

    def _unpack(self, keys: np.ndarray, n: int) -> np.ndarray:
        mask = (1 << self.bits) - 1
        shifts = self.bits * np.arange(n - 1, -1, -1)
        return (keys[:, None] >> shifts[None, :]) & mask

    def _prune(self, adj, min_count: int, chunk: int = 1 << 21) -> None:
        """Drop n-grams (order >= 3) whose adjusted count is below ``min_count``, then
        recompute each context's back-off weight so that P(. | h) still sums to one:

            gamma'(h) = (1 - sum_{kept w} P(w | h)) / (1 - sum_{kept w} P_lower(w | h[1:]))

        Orders are pruned bottom-up, so P_lower is always the final, already-pruned model.
        """
        b = self.bits
        for n in range(3, self.order + 1):
            keep = adj[n] >= min_count
            keys, logp = self.ngram_keys[n][keep], self.ngram_logp[n][keep]
            ctx_keys = self.ctx_keys[n]
            inv = np.searchsorted(ctx_keys, keys >> b)
            lower = np.empty(len(keys))
            for s in range(0, len(keys), chunk):
                cols = self._unpack(keys[s : s + chunk], n)[:, 1:]
                lower[s : s + chunk] = np.exp(self._logprob_cols(cols, n - 1))
            kept_mass = np.bincount(
                inv, weights=np.exp(logp.astype(np.float64)), minlength=len(ctx_keys)
            )
            lower_mass = np.bincount(inv, weights=lower, minlength=len(ctx_keys))
            gamma = np.maximum(1.0 - kept_mass, 1e-12) / np.maximum(1.0 - lower_mass, 1e-12)
            # A context with nothing left has gamma == 1, i.e. it behaves as if unseen: drop it.
            has_kept = np.bincount(inv, minlength=len(ctx_keys)) > 0
            self.ngram_keys[n], self.ngram_logp[n] = keys, logp
            self.ctx_keys[n] = ctx_keys[has_kept]
            self.ctx_logbow[n] = np.log(gamma[has_kept]).astype(np.float32)

    @staticmethod
    def _discounts(a: np.ndarray) -> np.ndarray:
        """Modified KN discounts [0, D1, D2, D3+] from counts-of-counts (Chen & Goodman)."""
        n1, n2, n3, n4 = (max(int((a == k).sum()), 1) for k in (1, 2, 3, 4))
        y = n1 / (n1 + 2 * n2)
        d = np.array([0.0, 1 - 2 * y * n2 / n1, 2 - 3 * y * n3 / n2, 3 - 4 * y * n4 / n3])
        # With very little data the formulas can give D <= 0, which would leave no probability
        # mass for unseen words (log gamma = -inf). Keep every discount in [0.1, k].
        d[1:] = np.clip(d[1:], 0.1, [1.0, 2.0, 3.0])
        return d

    def _estimate(self, keys, adj) -> None:
        N, b = self.order, self.bits
        self.ngram_keys, self.ngram_logp = {}, {}
        self.ctx_keys, self.ctx_logbow = {}, {}
        lower_p = None  # interpolated probabilities of order n-1, aligned with keys[n-1]
        for n in range(1, N + 1):
            a = adj[n].astype(np.float64)
            D = self.discounts_[n][np.minimum(adj[n], 3)]
            ctx = keys[n] >> b  # (n-1)-gram context key; 0 for unigrams
            ctx_uniq, inv = np.unique(ctx, return_inverse=True)
            total = np.bincount(inv, weights=a)
            mass = np.bincount(inv, weights=D)  # discount mass freed in each context
            gamma = mass / total
            discounted = (a - D) / total[inv]
            if n == 1:
                lower = np.full(len(a), 1.0 / self.vocab_size)
            else:
                suffix = keys[n] & ((1 << (b * (n - 1))) - 1)
                lower = lower_p[np.searchsorted(keys[n - 1], suffix)]
            p = discounted + gamma[inv] * lower
            if n == 1:
                self.unigram_floor_ = float(gamma[0] / self.vocab_size)  # P(unseen word)
            self.ngram_keys[n], self.ngram_logp[n] = keys[n], np.log(p).astype(np.float32)
            self.ctx_keys[n], self.ctx_logbow[n] = ctx_uniq, np.log(gamma).astype(np.float32)
            lower_p = p

    # --- querying ---------------------------------------------------------------------------
    def logprob(self, history: np.ndarray, word: np.ndarray) -> np.ndarray:
        """Natural-log P(word | history) for a batch.

        ``history`` has shape (B, order-1) with the most recent token last (pad with <s>);
        ``word`` has shape (B,).
        """
        N = self.order
        cols = np.concatenate([history[:, -(N - 1) :], word[:, None]], 1)
        return self._logprob_cols(cols, N)

    def _logprob_cols(self, cols: np.ndarray, top: int) -> np.ndarray:
        """Like :meth:`logprob` for an explicit (B, top) n-gram matrix, using orders <= top."""
        N, b = top, self.bits
        word = cols[:, -1]
        result = np.full(len(word), np.nan)
        backoff = np.zeros(len(word))
        for n in range(N, 0, -1):
            todo = np.isnan(result)
            if not todo.any():
                break
            key = self._pack(cols[todo, N - n :])
            k_arr = self.ngram_keys[n]
            pos = np.minimum(np.searchsorted(k_arr, key), len(k_arr) - 1)
            hit = k_arr[pos] == key
            idx = np.flatnonzero(todo)
            result[idx[hit]] = backoff[idx[hit]] + self.ngram_logp[n][pos[hit]]
            if n > 1:  # missed: add gamma(context) if the context itself was seen
                miss = idx[~hit]
                ctx = key[~hit] >> b
                c_arr = self.ctx_keys[n]
                cpos = np.minimum(np.searchsorted(c_arr, ctx), len(c_arr) - 1)
                seen = c_arr[cpos] == ctx
                backoff[miss[seen]] += self.ctx_logbow[n][cpos[seen]]
        unseen = np.isnan(result)
        result[unseen] = backoff[unseen] + np.log(self.unigram_floor_)
        return result

    def sentence_logprob(self, ids: np.ndarray) -> float:
        seq = np.concatenate([np.zeros(self.order - 1, dtype=np.int64), ids, [1]])
        hist = np.lib.stride_tricks.sliding_window_view(seq[:-1], self.order - 1)
        return float(self.logprob(hist[-(len(ids) + 1) :], seq[self.order - 1 :]).sum())

    @property
    def n_ngrams(self) -> dict[int, int]:
        return {n: len(k) for n, k in self.ngram_keys.items()}

    @property
    def nbytes(self) -> int:
        arrays = [*self.ngram_keys.values(), *self.ngram_logp.values()]
        arrays += [*self.ctx_keys.values(), *self.ctx_logbow.values()]
        return sum(a.nbytes for a in arrays)
