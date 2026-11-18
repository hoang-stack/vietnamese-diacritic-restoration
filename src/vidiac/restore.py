"""Turning stripped text back into Vietnamese: candidate generation + decoding."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from vidiac.lm import UNK, KneserNeyLM, Vocabulary
from vidiac.text import Token, has_diacritics, sentences, strip_diacritics, tokenize, transfer_case


class CandidateTable:
    """Stripped lower-case syllable -> accented forms seen in training, most frequent first.

    ``min_count`` drops typos ("mă" seen once) and ``max_candidates`` caps the branching factor.
    The table is what turns restoration into a choice among a handful of options per syllable.
    """

    def __init__(self, table: dict[str, list[tuple[str, int]]]):
        self.table = table

    @classmethod
    def build(
        cls,
        word_counts: Counter,
        min_count: int = 2,
        max_candidates: int = 16,
        allowed: set[str] | dict | None = None,
    ):
        """``allowed`` restricts forms to the language model's vocabulary. Out-of-vocabulary
        forms would all be scored as <unk>, whose probability pools *every* rare token and is
        therefore far too high for any single one of them."""
        grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for word, count in word_counts.items():
            if count >= min_count and (allowed is None or word in allowed):
                grouped[strip_diacritics(word)].append((word, count))
        table = {
            key: sorted(forms, key=lambda f: (-f[1], f[0]))[:max_candidates]
            for key, forms in grouped.items()
        }
        return cls(table)

    def forms(self, stripped_lower: str) -> list[str]:
        return [f for f, _ in self.table.get(stripped_lower, [])]

    def most_frequent(self, stripped_lower: str) -> str | None:
        forms = self.table.get(stripped_lower)
        return forms[0][0] if forms else None

    def __len__(self) -> int:
        return len(self.table)


@dataclass
class Slot:
    """One token position during decoding: the original token and the forms it may take."""

    token: Token
    forms: list[str]  # lower-case options (one option for numbers, punctuation, fixed words)
    prior: np.ndarray | None = None  # optional log-scores per form from a context classifier


def make_slots(tokens: list[Token], candidates: CandidateTable) -> list[Slot]:
    slots = []
    for tok in tokens:
        if tok.kind != "word":
            forms = [tok.lm_form]
        elif has_diacritics(tok.text):  # the user already typed the marks: keep them
            forms = [tok.text.lower()]
        else:
            forms = candidates.forms(tok.text.lower()) or [tok.text.lower()]
        slots.append(Slot(tok, forms))
    return slots


def render(text: str, slots: list[Slot], choice: list[int]) -> str:
    """Write chosen forms back into the original string, keeping spacing and capitalisation."""
    out, cursor = [], 0
    for slot, c in zip(slots, choice, strict=True):
        tok = slot.token
        out.append(text[cursor : tok.start])
        out.append(transfer_case(tok.text, slot.forms[c]) if tok.kind == "word" else tok.text)
        cursor = tok.end
    out.append(text[cursor:])
    return "".join(out)


class Restorer:
    """Base class: subclasses choose one form per slot for a sentence."""

    name = "base"

    def __init__(self, candidates: CandidateTable):
        self.candidates = candidates

    def choose(self, slots: list[Slot]) -> list[int]:
        raise NotImplementedError

    def restore(self, text: str) -> str:
        tokens = tokenize(text)
        slots: list[Slot] = []
        choice: list[int] = []
        for sent in sentences(tokens):
            s = make_slots(sent, self.candidates)
            slots.extend(s)
            choice.extend(self.choose(s))
        return render(text, slots, choice)


class Identity(Restorer):
    """Lower bound: leave the stripped text as it is."""

    name = "No restoration"

    def choose(self, slots):
        return [s.forms.index(s.token.lm_form) if s.token.lm_form in s.forms else 0 for s in slots]

    def restore(self, text: str) -> str:
        return text


class MostFrequent(Restorer):
    """Context-free baseline: every syllable gets its most frequent accented form."""

    name = "Most frequent form"

    def choose(self, slots):
        return [0 for _ in slots]


class BatchRestorer(Restorer):
    """A restorer that decodes many sentences at once (override ``choose_batch``)."""

    batch_size = 2048  # sentences per decoding batch

    def choose(self, slots: list[Slot]) -> list[int]:
        return self.choose_batch([slots])[0]

    def choose_batch(self, batch: list[list[Slot]]) -> list[list[int]]:
        raise NotImplementedError

    def restore_many(self, texts: list[str]) -> list[str]:
        per_text: list[list[list[Slot]]] = []
        flat: list[list[Slot]] = []
        for text in texts:
            sents = [make_slots(s, self.candidates) for s in sentences(tokenize(text))]
            per_text.append(sents)
            flat.extend(sents)
        choices: list[list[int]] = []
        for start in range(0, len(flat), self.batch_size):
            choices.extend(self.choose_batch(flat[start : start + self.batch_size]))
        out, i = [], 0
        for text, sents in zip(texts, per_text, strict=True):
            slots = [s for sent in sents for s in sent]
            choice = [c for ch in choices[i : i + len(sents)] for c in ch]
            i += len(sents)
            out.append(render(text, slots, choice))
        return out

    def restore(self, text: str) -> str:
        return self.restore_many([text])[0]

    def restore_detailed(self, text: str) -> tuple[str, list[tuple[Slot, int]]]:
        """Restored text plus every word slot with the index of the chosen form."""
        sents = [make_slots(s, self.candidates) for s in sentences(tokenize(text))]
        choices = self.choose_batch(sents) if sents else []
        slots = [s for sent in sents for s in sent]
        choice = [c for ch in choices for c in ch]
        details = [(s, c) for s, c in zip(slots, choice, strict=True) if s.token.kind == "word"]
        return render(text, slots, choice), details


class LMRestorer(BatchRestorer):
    """Beam search over candidate forms, scored by an n-gram language model.

    Hypotheses that end in the same (order-1) tokens are recombined (only the best survives),
    because the language model cannot tell them apart from then on. With order 2 that makes
    beam search exact Viterbi decoding once the beam is at least the number of candidates.

    Decoding is *batched*: position t of every sentence in the batch is expanded in one
    vectorised step, so the per-call overhead of NumPy is paid once per position rather than
    once per position per sentence.
    """

    name = "N-gram LM + beam search"

    def __init__(
        self,
        candidates: CandidateTable,
        lm: KneserNeyLM,
        vocab: Vocabulary,
        beam: int = 16,
        scorer=None,
        weight: float = 0.0,
    ):
        """``scorer`` (e.g. a :class:`~vidiac.classifier.ContextScorer`) adds ``weight`` times
        its log-probability for each option to the LM score: a log-linear combination."""
        super().__init__(candidates)
        self.lm, self.vocab, self.beam = lm, vocab, beam
        self.scorer, self.weight = scorer, weight
        self._unk = vocab.stoi[UNK]
        if scorer is not None and weight > 0:
            self.name = "LM + context classifier"

    def _ids(self, forms: list[str]) -> list[int]:
        return [self.vocab.stoi.get(f, self._unk) for f in forms]

    def choose_batch(self, batch: list[list[Slot]]) -> list[list[int]]:
        lm, k, B = self.lm, self.lm.order - 1, self.beam
        lengths = np.array([len(s) for s in batch])
        S, T = len(batch), int(lengths.max(initial=0))
        if S << (k * lm.bits) >= 1 << 62:
            raise ValueError("batch too large for packed state keys")

        use_prior = self.scorer is not None and self.weight > 0
        if use_prior:
            self.scorer.annotate(batch)

        # Candidate ids (and optional prior scores) for every (sentence, position), flattened
        # per position.
        flat_ids, flat_prior, offsets, counts = [], [], [], []
        for t in range(T):
            ids_t, pri_t = [], []
            off_t, cnt_t = np.zeros(S, np.int64), np.zeros(S, np.int64)
            for s in np.flatnonzero(lengths > t):
                slot = batch[s][t]
                ids = self._ids(slot.forms)
                off_t[s], cnt_t[s] = len(ids_t), len(ids)
                ids_t.extend(ids)
                if use_prior:
                    pri_t.extend(slot.prior if slot.prior is not None else [0.0] * len(ids))
            flat_ids.append(np.array(ids_t, np.int64))
            flat_prior.append(np.array(pri_t, np.float64))
            offsets.append(off_t)
            counts.append(cnt_t)

        hyp_sent = np.flatnonzero(lengths > 0)  # one empty hypothesis per non-empty sentence
        states = np.zeros((len(hyp_sent), k), np.int64)
        scores = np.zeros(len(hyp_sent))
        parents: list[np.ndarray] = []
        options: list[np.ndarray] = []
        final = np.zeros(S, np.int64)  # best hypothesis index at the sentence's last step

        for t in range(T + 1):
            done = lengths[hyp_sent] == t
            if t > 0 and done.any():
                idx = np.flatnonzero(done)
                eos = np.ones(len(idx), np.int64)
                fin = scores[idx] + lm.logprob(states[idx], eos)
                order = np.lexsort((-fin, hyp_sent[idx]))
                s_sorted = hyp_sent[idx][order]
                first = np.r_[True, s_sorted[1:] != s_sorted[:-1]]
                final[s_sorted[first]] = idx[order[first]]
            if t == T:
                break
            alive = np.flatnonzero(~done)
            hs, st, sc = hyp_sent[alive], states[alive], scores[alive]

            cnt = counts[t][hs]
            rep = np.repeat(np.arange(len(hs)), cnt)
            within = np.arange(len(rep)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            flat_pos = offsets[t][hs][rep] + within
            word = flat_ids[t][flat_pos]
            hist = st[rep]
            sent = hs[rep]
            total = sc[rep] + lm.logprob(hist, word)
            if use_prior:
                total = total + self.weight * flat_prior[t][flat_pos]
            new_states = np.concatenate([hist[:, 1:], word[:, None]], axis=1)

            # Best first within each sentence; recombine identical LM states; keep top B.
            order = np.lexsort((-total, sent))
            key = (sent << (k * lm.bits)) | lm._pack(new_states)
            _, first = np.unique(key[order], return_index=True)
            keep = order[np.sort(first)]
            s_keep = sent[keep]
            rank = np.arange(len(keep)) - np.searchsorted(s_keep, s_keep, side="left")
            keep = keep[rank < B]

            parents.append(alive[rep[keep]])
            options.append(within[keep])
            hyp_sent, states, scores = sent[keep], new_states[keep], total[keep]

        choices = []
        for s in range(S):
            h, picks = final[s], []
            for t in range(lengths[s] - 1, -1, -1):
                picks.append(int(options[t][h]))
                h = parents[t][h]
            choices.append(picks[::-1])
        return choices


class ClassifierRestorer(BatchRestorer):
    """Pointwise decoding: each syllable independently takes the classifier's best form."""

    name = "Context classifier"

    def __init__(self, candidates: CandidateTable, scorer):
        super().__init__(candidates)
        self.scorer = scorer

    def choose_batch(self, batch: list[list[Slot]]) -> list[list[int]]:
        self.scorer.annotate(batch)
        return [
            [int(np.argmax(s.prior)) if s.prior is not None else 0 for s in sent] for sent in batch
        ]


class PrecomputedScorer:
    """Feeds per-slot log-probabilities computed elsewhere (e.g. the Kaggle Transformer).

    The scores must come in the exact order the decoder visits ambiguous slots: text by text,
    sentence by sentence, left to right, one array per slot with two or more forms.
    """

    def __init__(self, values: np.ndarray, lengths: np.ndarray):
        self.values, self.lengths = values, lengths
        self.offsets = np.concatenate([[0], np.cumsum(lengths)])
        self.cursor = 0

    def reset(self) -> None:
        self.cursor = 0

    def annotate(self, batch: list[list[Slot]]) -> None:
        for sent in batch:
            for slot in sent:
                if len(slot.forms) > 1:
                    k = self.cursor
                    if self.lengths[k] != len(slot.forms):
                        raise ValueError("precomputed scores do not line up with the slots")
                    slot.prior = self.values[self.offsets[k] : self.offsets[k + 1]].astype(
                        np.float64
                    )
                    self.cursor += 1
