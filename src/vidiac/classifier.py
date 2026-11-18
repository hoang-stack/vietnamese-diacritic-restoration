"""Discriminative context classifier: P(accented form | stripped context window).

An n-gram LM reads left to right, so when it scores a syllable it has not seen the words that
follow. This model looks both ways. For an ambiguous stripped syllable x_i it chooses among
x_i's candidate forms y with a log-linear (multinomial logistic regression) model

    score(y) = sum over templates t of  w[ hash(t, context_t, y) ]

where each template reads stripped syllables around the target, e.g. x_{i-1}, x_{i+1},
(x_{i-1}, x_{i+1}) or (x_{i+1}, x_{i+2}). Feature *conjunctions with the candidate* are hashed
into a fixed table of 2^bits weights (the "hashing trick", Weinberger et al. 2009), so the model
never builds a feature dictionary. Training is mini-batch softmax cross-entropy with Adagrad.
"""

from __future__ import annotations

import numpy as np

from vidiac.text import strip_diacritics

# Offsets relative to the target syllable. () is a per-candidate bias, i.e. a prior.
TEMPLATES: tuple[tuple[int, ...], ...] = (
    (),
    (-1,),
    (1,),
    (-2,),
    (2,),
    (-1, 1),
    (-2, -1),
    (1, 2),
)
WINDOW = 2  # context reaches 2 syllables either side

_P = np.array(
    [0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0xD6E8FEB86659FD93],
    dtype=np.uint64,
)


def _mix(x: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser: a cheap, well-distributed 64-bit hash."""
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


class StrippedVocab:
    """Ids for stripped tokens. 0 = sentence start pad, 1 = sentence end pad, 2 = unknown."""

    def __init__(self, accented_types: list[str]):
        stripped = {strip_diacritics(t) for t in accented_types}
        self.itos = ["<s>", "</s>", "<unk>", *sorted(stripped - {"<s>", "</s>", "<unk>"})]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def lookup(self, tokens: list[str]) -> np.ndarray:
        return np.array([self.stoi.get(t, 2) for t in tokens], dtype=np.int64)


class ContextClassifier:
    def __init__(
        self,
        bits: int = 24,
        lr: float = 0.1,
        epochs: int = 1,
        batch_size: int = 8192,
        max_examples: int | None = None,
        seed: int = 0,
    ):
        self.bits, self.lr, self.epochs = bits, lr, epochs
        self.batch_size, self.max_examples, self.seed = batch_size, max_examples, seed

    # --- features ---------------------------------------------------------------------------
    def _feature_index(self, ctx: np.ndarray, cands: np.ndarray) -> np.ndarray:
        """Hashed weight indices, shape (B, n_templates, C).

        ``ctx`` is (B, 2*WINDOW+1) stripped ids centred on the target; ``cands`` is (B, C)
        accented candidate ids (-1 = padding; its index is never used thanks to masking).
        """
        B, C = cands.shape
        y = cands.astype(np.uint64)  # (B, C); h[:, None] below is (B, 1) -> result (B, C)
        out = np.empty((B, len(TEMPLATES), C), dtype=np.int64)
        mask = np.uint64((1 << self.bits) - 1)
        for t, offsets in enumerate(TEMPLATES):
            h = np.full(B, np.uint64(t + 1)) * _P[0]
            for j, off in enumerate(offsets):
                h = _mix(h ^ (ctx[:, WINDOW + off].astype(np.uint64) * _P[1 + j]))
            out[:, t, :] = (_mix(h[:, None] ^ (y * _P[3])) & mask).astype(np.int64)
        return out

    def _scores(self, idx: np.ndarray, valid: np.ndarray) -> np.ndarray:
        s = self.w[idx].sum(axis=1)
        return np.where(valid, s, -np.inf)

    @staticmethod
    def _log_softmax(s: np.ndarray) -> np.ndarray:
        m = s.max(axis=1, keepdims=True)
        z = np.log(np.exp(s - m).sum(axis=1, keepdims=True)) + m
        return s - z

    # --- training ---------------------------------------------------------------------------
    def fit(self, examples: ExampleSet, log=None) -> ContextClassifier:
        rng = np.random.default_rng(self.seed)
        self.w = np.zeros(1 << self.bits, dtype=np.float32)
        G = np.full(1 << self.bits, 1e-6, dtype=np.float32)  # Adagrad accumulator
        N = len(examples)
        n = N if self.max_examples is None else min(N, self.max_examples)
        self.loss_history_ = []
        for _ in range(self.epochs):
            order = rng.permutation(N)[:n]
            total = 0.0
            for start in range(0, n, self.batch_size):
                ctx, cands, gold = examples.batch(order[start : start + self.batch_size])
                idx = self._feature_index(ctx, cands)
                valid = cands >= 0
                logp = self._log_softmax(self._scores(idx, valid))
                rows = np.arange(len(gold))
                total -= float(logp[rows, gold].sum())
                g = np.exp(logp)  # d(-log p_gold)/d score = p - onehot
                g[rows, gold] -= 1.0
                g = np.where(valid, g, 0.0).astype(np.float32) / len(gold)
                flat = idx.reshape(-1)
                grads = np.broadcast_to(g[:, None, :], idx.shape).reshape(-1)
                uniq, inv = np.unique(flat, return_inverse=True)
                gsum = np.bincount(inv, weights=grads, minlength=len(uniq)).astype(np.float32)
                G[uniq] += gsum * gsum
                self.w[uniq] -= self.lr * gsum / np.sqrt(G[uniq])
            self.loss_history_.append(total / n)
            if log:
                log(
                    f"classifier epoch {len(self.loss_history_)}: "
                    f"loss {total / n:.4f} on {n:,} examples"
                )
        return self

    # --- inference --------------------------------------------------------------------------
    def log_proba(self, ctx: np.ndarray, cands: np.ndarray) -> np.ndarray:
        """(B, C) log-probabilities over candidates (-inf on padding)."""
        idx = self._feature_index(ctx, cands)
        return self._log_softmax(self._scores(idx, cands >= 0))

    @property
    def nbytes(self) -> int:
        return self.w.nbytes


class ExampleSet:
    """Training examples for the classifier, stored compactly as positions in one padded,
    stripped token sequence. Context windows and candidate lists are gathered per batch, so
    tens of millions of examples fit in memory."""

    def __init__(self, stripped_seq, positions, gold, cand_matrix):
        self.seq, self.pos, self.gold, self.cand_matrix = stripped_seq, positions, gold, cand_matrix
        self._offsets = np.arange(-WINDOW, WINDOW + 1)

    def __len__(self) -> int:
        return len(self.pos)

    def batch(self, ix: np.ndarray):
        p = self.pos[ix]
        ctx = self.seq[p[:, None] + self._offsets]
        return ctx, self.cand_matrix[self.seq[p]], self.gold[ix].astype(np.int64)


def candidate_matrix(candidates, vocab, strip_vocab: StrippedVocab, width: int) -> np.ndarray:
    """(n_stripped, width) matrix of accented LM ids per stripped syllable, -1 padded."""
    mat = np.full((len(strip_vocab), width), -1, dtype=np.int64)
    for key, forms in candidates.table.items():
        sid = strip_vocab.stoi.get(key)
        if sid is not None:
            ids = [vocab.stoi[f] for f, _ in forms][:width]
            mat[sid, : len(ids)] = ids
    return mat


def build_examples(
    corpus, n_articles, vocab, strip_vocab, cand_matrix, chunk=1 << 22
) -> ExampleSet:
    """One example per ambiguous word token (>= 2 candidates) whose gold form is a candidate."""
    unk = vocab.stoi["<unk>"]
    type_lm = np.array([vocab.stoi.get(t, unk) for t in corpus.types], dtype=np.int64)
    type_strip = np.array(
        [strip_vocab.stoi.get(strip_diacritics(t), 2) for t in corpus.types], dtype=np.int64
    )
    sents = corpus.sentences(n_articles)
    lengths = np.fromiter((len(x) for x in sents), np.int64, len(sents))
    block = lengths + 2 * WINDOW
    start = np.concatenate([[0], np.cumsum(block)[:-1]])
    seq = np.zeros(int(block.sum()), dtype=np.int32)  # left pads: <s> (0)
    tok_pos = np.arange(int(lengths.sum())) + np.repeat(
        start + WINDOW - np.cumsum(lengths) + lengths, lengths
    )
    toks = np.concatenate(sents)
    seq[tok_pos] = type_strip[toks]
    for k in range(WINDOW):  # right pads: </s> (1)
        seq[start + WINDOW + lengths + k] = 1
    accented = type_lm[toks]
    n_cands = (cand_matrix >= 0).sum(axis=1)
    amb = n_cands[seq[tok_pos]] >= 2
    pos, gold_ids = tok_pos[amb], accented[amb]
    gold = np.full(len(pos), -1, dtype=np.int8)
    for s in range(0, len(pos), chunk):
        cm = cand_matrix[seq[pos[s : s + chunk]]]
        hit = cm == gold_ids[s : s + chunk, None]
        gold[s : s + chunk] = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
    keep = gold >= 0
    return ExampleSet(seq, pos[keep].astype(np.int64), gold[keep], cand_matrix)


class ContextScorer:
    """Adapter used by the decoders: fills ``slot.prior`` with classifier log-probabilities."""

    def __init__(
        self, classifier: ContextClassifier, vocab, strip_vocab: StrippedVocab, width: int
    ):
        self.classifier, self.vocab, self.strip_vocab, self.width = (
            classifier,
            vocab,
            strip_vocab,
            width,
        )

    def annotate(self, batch, chunk: int = 65536) -> None:
        ctx_rows, cand_rows, slots = [], [], []
        unk = self.vocab.stoi["<unk>"]
        for sent in batch:
            if not any(len(s.forms) > 1 for s in sent):
                continue
            sids = (
                [0] * WINDOW
                + [self.strip_vocab.stoi.get(strip_diacritics(s.token.lm_form), 2) for s in sent]
                + [1] * WINDOW
            )
            for i, slot in enumerate(sent):
                if len(slot.forms) > 1:
                    ctx_rows.append(sids[i : i + 2 * WINDOW + 1])
                    ids = [self.vocab.stoi.get(f, unk) for f in slot.forms][: self.width]
                    cand_rows.append(ids + [-1] * (self.width - len(ids)))
                    slots.append(slot)
        if not slots:
            return
        ctx, cands = np.array(ctx_rows, np.int64), np.array(cand_rows, np.int64)
        for s in range(0, len(slots), chunk):
            lp = self.classifier.log_proba(ctx[s : s + chunk], cands[s : s + chunk])
            for slot, row in zip(slots[s : s + chunk], lp, strict=True):
                slot.prior = row[: len(slot.forms)]
