import pytest
from fastapi.testclient import TestClient

from tests.conftest import build_toy
from vidiac.api import create_app
from vidiac.system import System, SystemConfig


@pytest.fixture(scope="module")
def client():
    vocab, lm, cands, _ = build_toy()
    system = System(SystemConfig(order=3), vocab, cands, lm=lm)
    with TestClient(create_app(system)) as c:
        yield c


def test_restore_returns_text_and_per_word_details(client):
    body = client.post("/restore", json={"text": "Toi ban ban cho ban.", "style": "old"}).json()
    assert body["text"] == "Tôi bán bàn cho bạn."
    words = body["words"]
    assert [w["input"] for w in words] == ["Toi", "ban", "ban", "cho", "ban"]
    assert words[0]["changed"] and words[0]["output"] == "Tôi"
    assert set(words[1]["alternatives"]) >= {"bạn", "bán", "bàn"}
    assert body["text"][words[4]["start"] : words[4]["end"]] == "bạn"


def test_tone_style_option(client):
    # The toy corpus has no oa/oe/uy syllables, so check that the option is accepted and that
    # unaffected text is identical in both styles.
    old = client.post("/restore", json={"text": "toi yeu ban", "style": "old"}).json()["text"]
    new = client.post("/restore", json={"text": "toi yeu ban", "style": "new"}).json()["text"]
    assert old == new == "tôi yêu bạn"


def test_validation_and_demo(client):
    assert client.post("/restore", json={"text": "x", "style": "weird"}).status_code == 422
    assert client.post("/restore", json={"text": "a" * 20_001}).status_code == 413
    assert "<html" in client.get("/").text.lower()
    assert client.get("/health").json()["status"] == "ok"
