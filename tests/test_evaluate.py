from vidiac.evaluate import evaluate, paired_bootstrap, score_article
from vidiac.restore import Identity, MostFrequent
from vidiac.text import nfc


def test_score_article_counts_words_ambiguity_and_sentences(toy):
    _, _, cands, _ = toy
    gold = nfc("Tôi bán bàn. Bạn tới!")
    pred = nfc("Tôi bạn bàn. Bạn tới!")
    score, conf = score_article(gold, pred, cands)
    assert (score.words, score.correct) == (5, 4)
    assert (score.sentences, score.sentences_correct) == (2, 1)
    assert conf == {("bán", "bạn"): 1}


def test_evaluate_and_bootstrap(toy):
    _, _, cands, _ = toy
    golds = [nfc("Tôi yêu bạn. Bạn tới nhà tôi."), nfc("Má tôi bán cá."), nfc("Năm mới tới.")]
    good = evaluate(MostFrequent(cands), golds, cands)
    bad = evaluate(Identity(cands), golds, cands)
    assert good.summary["accuracy"] > bad.summary["accuracy"]
    ci = good.bootstrap(n_boot=200)
    assert ci["ci_low"] <= ci["accuracy"] <= ci["ci_high"]
    diff = paired_bootstrap(good, bad, n_boot=200)
    assert diff["diff"] > 0
