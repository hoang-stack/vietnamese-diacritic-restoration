import itertools
from collections import Counter

import numpy as np
import pytest

from tests.conftest import build_toy
from vidiac.restore import CandidateTable, LMRestorer, MostFrequent, make_slots
from vidiac.text import sentences, strip_diacritics, tokenize


def test_candidate_table_orders_by_frequency_and_filters():
    counts = Counter({"bạn": 10, "bán": 6, "bàn": 5, "bân": 1, "ban": 2, "bãn": 7})
    table = CandidateTable.build(counts, min_count=2, allowed={"bạn", "bán", "bàn", "ban", "bân"})
    assert table.forms("ban") == ["bạn", "bán", "bàn", "ban"]  # "bân" too rare, "bãn" not allowed
    assert table.most_frequent("ban") == "bạn"
    assert table.forms("xyz") == []


def brute_force(restorer, slots):
    """Exhaustively score every combination of forms with the LM (exact argmax)."""
    lm, vocab = restorer.lm, restorer.vocab
    best, best_choice = -np.inf, None
    for choice in itertools.product(*[range(len(s.forms)) for s in slots]):
        ids = np.array([vocab.stoi.get(s.forms[c], 2) for s, c in zip(slots, choice, strict=True)])
        score = lm.sentence_logprob(ids)
        if score > best:
            best, best_choice = score, list(choice)
    return best_choice


@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize(
    "text", ["toi ban ban cho ban", "ban toi toi nha ma", "nam moi toi yeu nam"]
)
def test_beam_search_with_recombination_is_exact(order, text):
    vocab, lm, cands, _ = build_toy(order=order)
    restorer = LMRestorer(cands, lm, vocab, beam=64)
    slots = make_slots(tokenize(text), cands)
    assert restorer.choose(slots) == brute_force(restorer, slots)


def test_batched_decoding_matches_one_sentence_at_a_time(toy):
    vocab, lm, cands, _ = toy
    texts = ["toi yeu ban.", "ban toi ban ban. ma toi ban ca!", "nam nay toi toi ha noi", ""]
    restorer = LMRestorer(cands, lm, vocab, beam=4)
    together = restorer.restore_many(texts)
    restorer.batch_size = 1
    assert restorer.restore_many(texts) == together
    assert [restorer.restore(t) for t in texts] == together


def test_restoration_uses_context(toy):
    vocab, lm, cands, _ = toy
    out = LMRestorer(cands, lm, vocab).restore("Toi ban ban cho ban. Toi yeu Viet Nam.")
    assert out == "Tôi bán bàn cho bạn. Tôi yêu Việt Nam."


def test_layout_case_and_typed_diacritics_are_preserved(toy):
    vocab, lm, cands, _ = toy
    r = LMRestorer(cands, lm, vocab)
    text = "  TOI   yeu   Viet Nam!!  (2024)  xyzq  "
    out = r.restore(text)
    assert out == "  TÔI   yêu   Việt Nam!!  (2024)  xyzq  "
    assert len(out) == len(text)
    # A form the user typed with diacritics is kept even if the model would choose otherwise.
    assert r.restore("Toi bàn ban").startswith("Tôi bàn")


def test_most_frequent_baseline(toy):
    _, _, cands, _ = toy
    assert MostFrequent(cands).restore("toi ban") == "tôi bạn"


def test_slots_follow_sentences(toy):
    _, _, cands, _ = toy
    text = strip_diacritics("Tôi yêu bạn. Bạn tới nhà tôi.")
    slots = [make_slots(s, cands) for s in sentences(tokenize(text))]
    assert len(slots) == 2
    assert [s.forms for s in slots[0]][0] == cands.forms("toi")


def test_corpus_subset_keeps_whole_articles():
    from vidiac.corpus import TokenCorpus

    corpus = TokenCorpus.build(["a b. c d e.", "f g.", "h. i j. k."], workers=1)
    sub = corpus.subset(np.array([0, 2]))
    assert sub.n_articles == 2
    assert [[corpus.types[i] for i in s] for s in sub.sentences()] == [
        ["a", "b"],
        ["c", "d", "e"],
        ["h"],
        ["i", "j"],
        ["k"],
    ]
