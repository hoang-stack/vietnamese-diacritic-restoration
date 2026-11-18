import numpy as np

from tests.conftest import build_toy
from vidiac.classifier import (
    TEMPLATES,
    ContextClassifier,
    ContextScorer,
    StrippedVocab,
    build_examples,
    candidate_matrix,
)
from vidiac.corpus import TokenCorpus
from vidiac.restore import ClassifierRestorer, LMRestorer, make_slots
from vidiac.text import tokenize


def toy_corpus():
    vocab, lm, cands, sents = build_toy()
    corpus = TokenCorpus.build([" ".join(s) + " ." for s in sents], workers=1)
    return vocab, lm, cands, corpus


def test_feature_index_shape_regression():
    """(B, T, C) exactly: a broadcasting slip here once produced a (B, B, C) array (8 GB)."""
    clf = ContextClassifier(bits=10)
    ctx = np.random.randint(0, 50, size=(300, 5))
    cands = np.random.randint(0, 50, size=(300, 7))
    idx = clf._feature_index(ctx, cands)
    assert idx.shape == (300, len(TEMPLATES), 7)
    assert idx.min() >= 0 and idx.max() < 1 << 10


def test_examples_and_training_learn_the_toy_patterns():
    vocab, lm, cands, corpus = toy_corpus()
    sv = StrippedVocab(vocab.itos)
    cmat = candidate_matrix(cands, vocab, sv, width=8)
    ex = build_examples(corpus, None, vocab, sv, cmat)
    assert len(ex) > 50
    ctx, cand, gold = ex.batch(np.arange(len(ex)))
    assert ctx.shape == (len(ex), 5)
    assert (cand[np.arange(len(ex)), gold] >= 0).all()
    clf = ContextClassifier(bits=16, lr=0.5, epochs=30, batch_size=64).fit(ex)
    assert clf.loss_history_[-1] < 0.5 * clf.loss_history_[0]
    scorer = ContextScorer(clf, vocab, sv, width=8)
    out = ClassifierRestorer(cands, scorer).restore("toi ban ban cho ban")
    assert out == "tôi bán bàn cho bạn"
    # Hybrid decoding runs and the prior is attached to ambiguous slots only.
    slots = make_slots(tokenize("toi ban ban"), cands)
    scorer.annotate([slots])
    assert all((s.prior is not None) == (len(s.forms) > 1) for s in slots)
    hybrid = LMRestorer(cands, lm, vocab, scorer=scorer, weight=0.5)
    assert hybrid.restore("toi ban ban cho ban") == "tôi bán bàn cho bạn"
