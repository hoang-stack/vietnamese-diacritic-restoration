import numpy as np
import pytest

from tests.conftest import build_toy
from vidiac.lm import KneserNeyLM


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("prune", [0, 2])
def test_conditional_distributions_sum_to_one(order, prune):
    vocab, lm, _, _ = build_toy(order=order, prune=prune)
    V = len(vocab)
    words = np.arange(V)
    histories = [["<s>"] * 3, ["tôi", "bán", "bàn"], ["của", "tôi", "yêu"], ["xyz", "abc", "nhà"]]
    for h in histories:
        hist = np.tile(vocab.encode(h[-(order - 1) :]), (V, 1))
        assert np.exp(lm.logprob(hist, words)).sum() == pytest.approx(1.0, abs=1e-5)


def test_discounts_stay_positive_on_degenerate_counts():
    d = KneserNeyLM._discounts(np.array([3, 3, 3, 3]))  # no singletons at all
    assert (d[1:] >= 0.1).all()


def test_discounts_follow_chen_and_goodman():
    counts = np.array([1] * 50 + [2] * 20 + [3] * 10 + [4] * 5)
    d = KneserNeyLM._discounts(counts)
    y = 50 / (50 + 2 * 20)
    assert d[1] == pytest.approx(1 - 2 * y * 20 / 50)
    assert d[2] == pytest.approx(2 - 3 * y * 10 / 20)
    assert d[3] == pytest.approx(3 - 4 * y * 5 / 10)


def test_sentence_logprob_is_the_sum_of_token_logprobs(toy):
    vocab, lm, _, _ = toy
    ids = vocab.encode(["tôi", "bán", "bàn"])
    hist = np.array([[0, 0], [0, ids[0]], [ids[0], ids[1]], [ids[1], ids[2]]])
    words = np.array([*ids, 1])
    assert lm.sentence_logprob(ids) == pytest.approx(lm.logprob(hist, words).sum())


def test_context_matters(toy):
    vocab, lm, _, _ = toy
    lp = lambda h, w: lm.logprob(vocab.encode(h)[None, :], vocab.encode([w]))[0]  # noqa: E731
    assert lp(["tôi", "yêu"], "bạn") > lp(["tôi", "yêu"], "bán")
    assert lp(["<s>", "tôi"], "bán") > lp(["<s>", "tôi"], "bàn")


def test_pruning_shrinks_the_model():
    _, full, _, _ = build_toy(order=3, repeat=1)
    _, pruned, _, _ = build_toy(order=3, prune=2, repeat=1)
    assert pruned.n_ngrams[3] < full.n_ngrams[3]
    assert pruned.nbytes < full.nbytes


def test_too_large_vocabulary_is_rejected():
    with pytest.raises(ValueError):
        KneserNeyLM(5).fit([np.array([3, 4])], vocab_size=1 << 15)
