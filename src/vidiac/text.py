"""Vietnamese text utilities.

Vietnamese marks two different things on vowels:

* **vowel quality**: a/ă/â, e/ê, o/ô/ơ, u/ư (plus the consonant d/đ), and
* **tone**: none, grave (huyền), acute (sắc), hook (hỏi), tilde (ngã), dot below (nặng).

"Diacritic restoration" puts both back. In NFC every Vietnamese letter is a single code point,
so stripping maps one character to one character. That is what makes copying the input's
capitalisation onto the restored text trivial (:func:`transfer_case`).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass

TONE_MARKS = {
    "̀": "grave",  # huyền  à
    "́": "acute",  # sắc    á
    "̉": "hook",  # hỏi    ả
    "̃": "tilde",  # ngã    ã
    "̣": "dot",  # nặng   ạ
}

_LOWER = "aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵđ"
_VOWELS = set("aăâeêioôơuưy")


def _base(ch: str) -> str:
    if ch in "đĐ":
        return "d" if ch == "đ" else "D"
    decomposed = unicodedata.normalize("NFD", ch)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


_STRIP_TABLE = str.maketrans({c: _base(c) for c in _LOWER + _LOWER.upper()})


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def strip_diacritics(text: str) -> str:
    """Remove tones and vowel marks, and map đ -> d. Expects NFC text; keeps its length."""
    return text.translate(_STRIP_TABLE)


def has_diacritics(token: str) -> bool:
    return strip_diacritics(token) != token


# --- tokenisation ---------------------------------------------------------------------------
# A "word" is a run of letters. Digits form numbers; anything else that is not whitespace is a
# single punctuation token. Sentence-final punctuation also closes the current sentence.
_TOKEN_RE = re.compile(r"(?P<word>[^\W\d_]+)|(?P<num>\d+(?:[.,]\d+)*)|(?P<punct>[^\w\s]|_)")
_SENTENCE_END = set(".!?…")
NUM = "<num>"


@dataclass(frozen=True)
class Token:
    kind: str  # "word" | "num" | "punct"
    text: str
    start: int
    end: int

    @property
    def lm_form(self) -> str:
        """What the language model sees for this token."""
        if self.kind == "word":
            return self.text.lower()
        if self.kind == "num":
            return NUM
        return self.text


def tokenize(text: str) -> list[Token]:
    return [Token(m.lastgroup, m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def _ends_sentence(tok: Token, nxt: Token | None) -> bool:
    # "X.Phú" or "3.5" are not sentence ends: the full stop must be followed by a gap.
    return (
        tok.kind == "punct" and tok.text in _SENTENCE_END and (nxt is None or nxt.start > tok.end)
    )


def sentences(tokens: list[Token]) -> Iterator[list[Token]]:
    """Split a token list after sentence-final punctuation that is followed by whitespace."""
    current: list[Token] = []
    for i, tok in enumerate(tokens):
        current.append(tok)
        if _ends_sentence(tok, tokens[i + 1] if i + 1 < len(tokens) else None):
            yield current
            current = []
    if current:
        yield current


def lm_sentences(text: str) -> Iterator[list[str]]:
    """Lower-cased LM token sequences of a text, one list per sentence (no end punctuation)."""
    for sent in sentences(tokenize(text)):
        if sent[-1].kind == "punct" and sent[-1].text in _SENTENCE_END:
            sent = sent[:-1]
        if sent:
            yield [t.lm_form for t in sent]


# --- capitalisation -------------------------------------------------------------------------
def transfer_case(template: str, word: str) -> str:
    """Copy the capitalisation of ``template`` onto ``word`` (same length in NFC)."""
    if template.islower():
        return word
    if len(template) == len(word):
        return "".join(w.upper() if t.isupper() else w for t, w in zip(template, word, strict=True))
    if template.isupper():
        return word.upper()
    return word[:1].upper() + word[1:] if template[:1].isupper() else word


# --- tone placement style -------------------------------------------------------------------
def split_tone(syllable: str) -> tuple[str, str | None]:
    """'hoà' -> ('hoa', '̀'). Vowel-quality marks are kept, only the tone is removed."""
    decomposed = unicodedata.normalize("NFD", syllable)
    tone = next((c for c in decomposed if c in TONE_MARKS), None)
    toneless = nfc("".join(c for c in decomposed if c not in TONE_MARKS))
    return toneless, tone


def _put_tone(toneless: str, tone: str, index: int) -> str:
    return nfc(toneless[: index + 1] + tone + toneless[index + 1 :])


_STYLE_CLUSTERS = ("oa", "oe", "uy")


def set_tone_style(text: str, style: str = "old") -> str:
    """Convert between the two accepted tone-placement conventions.

    Only syllables that *end* in oa / oe / uy differ (and not "qu"+y, where u belongs to the
    consonant): the old style marks the first vowel (hòa, khỏe, thủy), the new style the second
    one (hoà, khoẻ, thuỷ). Everything else is identical in both styles.
    """
    if style not in {"old", "new"}:
        raise ValueError("style must be 'old' or 'new'")

    def convert(match: re.Match) -> str:
        word = match.group()
        toneless, tone = split_tone(word)
        lower = toneless.lower()
        if tone is None or not lower.endswith(_STYLE_CLUSTERS) or lower.endswith("quy"):
            return word
        if len(lower) >= 3 and lower[-3] in _VOWELS:  # e.g. "ngoai"-like triphthongs: untouched
            return word
        index = len(toneless) - (2 if style == "old" else 1)
        return _put_tone(toneless, tone, index)

    return re.sub(r"[^\W\d_]+", convert, text)
