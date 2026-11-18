import pytest

from vidiac.text import (
    lm_sentences,
    nfc,
    set_tone_style,
    split_tone,
    strip_diacritics,
    tokenize,
    transfer_case,
)


def test_strip_diacritics_removes_tones_marks_and_d_bar_keeping_length():
    s = nfc("Đường đến Hà Nội, ĐẠI HỌC Bách Khoa: ưu tiên, tiếng Việt!")
    stripped = strip_diacritics(s)
    assert stripped == "Duong den Ha Noi, DAI HOC Bach Khoa: uu tien, tieng Viet!"
    assert len(stripped) == len(s)


def test_nfc_matters_decomposed_input_is_normalised():
    decomposed = "Việt"  # e + dot below + circumflex, as found in the raw corpus
    assert strip_diacritics(nfc(decomposed)) == "Viet"


@pytest.mark.parametrize(
    ("new", "old"),
    [("hoà", "hòa"), ("khoẻ", "khỏe"), ("thuỷ", "thủy"), ("Hoà Bình", "Hòa Bình"), ("TOÀ", "TÒA")],
)
def test_tone_style_round_trip(new, old):
    assert set_tone_style(nfc(new), "old") == nfc(old)
    assert set_tone_style(nfc(old), "new") == nfc(new)


@pytest.mark.parametrize("word", ["quý", "hoàng", "ngoài", "thuyền", "người", "tôi"])
def test_tone_style_leaves_unaffected_syllables_alone(word):
    assert set_tone_style(nfc(word), "old") == nfc(word)
    assert set_tone_style(nfc(word), "new") == nfc(word)


def test_split_tone():
    assert split_tone(nfc("hoà")) == ("hoa", "̀")
    assert split_tone(nfc("ươn")) == (nfc("ươn"), None)


def test_tokenize_spans_rebuild_the_text_and_classify_tokens():
    text = "VKSND T X.Phú Mỹ, ngày 22/6 có 3,5% người."
    toks = tokenize(text)
    assert [t.kind for t in toks][:6] == ["word", "word", "word", "punct", "word", "word"]
    assert all(text[t.start : t.end] == t.text for t in toks)
    assert ("num", "3,5") in [(t.kind, t.text) for t in toks]


def test_sentence_split_ignores_abbreviation_dots():
    sents = list(lm_sentences("T X.Phú Mỹ ra quyết định. Cùng ngày 22/6 , VKSND phê chuẩn!"))
    assert sents == [
        ["t", "x", ".", "phú", "mỹ", "ra", "quyết", "định"],
        ["cùng", "ngày", "<num>", "/", "<num>", ",", "vksnd", "phê", "chuẩn"],
    ]


def test_transfer_case():
    assert transfer_case("VIET", "việt") == "VIỆT"
    assert transfer_case("Viet", "việt") == "Việt"
    assert transfer_case("viet", "việt") == "việt"
    assert transfer_case("iPhone", "iphone") == "iPhone"
