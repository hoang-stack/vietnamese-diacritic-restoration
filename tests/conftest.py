from collections import Counter

import pytest

from vidiac.lm import KneserNeyLM, Vocabulary
from vidiac.restore import CandidateTable
from vidiac.text import lm_sentences, nfc

# A tiny corpus with real Vietnamese ambiguity: "ban" -> bạn / bán / bàn, "toi" -> tôi / tới,
# "ma" -> mà / má, "nam" -> năm / nam. Context decides which form is right.
CORPUS = nfc(
    """Tôi yêu bạn. Bạn của tôi bán hàng. Tôi tới nhà bạn. Bàn này của tôi.
    Tôi mua bàn mới. Họ bán bàn gỗ. Bạn tôi tới chơi. Năm nay tôi tới Hà Nội.
    Má tôi bán cá mà không lời. Năm mới bạn tới nhà má. Anh ấy là nam sinh.
    Tôi yêu Việt Nam. Bạn yêu Việt Nam. Tôi bán bàn cho bạn. Bạn tới nhà tôi.
    Má tôi yêu nhà. Nhà của bạn ở Hà Nội. Tôi về nhà vào năm mới."""
)


def build_toy(order=3, prune=0, repeat=3):
    sents = [s for line in CORPUS.split("\n") for s in lm_sentences(line)] * repeat
    counts = Counter(t for s in sents for t in s)
    vocab = Vocabulary.build(counts, max_size=1 << 10)
    lm = KneserNeyLM(order).fit([vocab.encode(s) for s in sents], len(vocab), prune=prune)
    words = Counter({w: c for w, c in counts.items() if w.isalpha()})
    cands = CandidateTable.build(words, min_count=1, allowed=vocab.stoi)
    return vocab, lm, cands, sents


@pytest.fixture(scope="session")
def toy():
    return build_toy()
