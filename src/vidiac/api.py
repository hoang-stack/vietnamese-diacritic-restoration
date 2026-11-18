"""REST API and demo page. Run with ``vidiac serve`` (artifact from ``vidiac train``)."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from vidiac import __version__
from vidiac.system import System
from vidiac.text import nfc, set_tone_style, transfer_case

DEFAULT_ARTIFACT = "artifacts/vidiac.pkl"
MAX_CHARS = 20_000


class RestoreRequest(BaseModel):
    text: str = Field(..., examples=["Toi yeu tieng Viet, hom nay troi dep qua!"])
    style: Literal["old", "new"] = Field(
        "old", description="Tone placement: old = hòa/thủy (most keyboards), new = hoà/thuỷ"
    )


class Word(BaseModel):
    start: int
    end: int
    input: str
    output: str
    changed: bool
    alternatives: list[str]


class RestoreResponse(BaseModel):
    text: str
    words: list[Word]
    elapsed_ms: float


def create_app(system: System | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.system = system or System.load(
            os.environ.get("VIDIAC_ARTIFACT", DEFAULT_ARTIFACT)
        )
        app.state.restorer = app.state.system.restorer("lm")
        yield

    app = FastAPI(
        title="vidiac",
        version=__version__,
        lifespan=lifespan,
        description="Vietnamese diacritic restoration (n-gram LM + beam search).",
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def demo() -> str:
        return (files("vidiac") / "static" / "index.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health(request: Request) -> dict:
        s: System = request.app.state.system
        return {
            "status": "ok",
            "model": f"{s.lm.order}-gram Kneser-Ney LM + beam search ({s.config.beam})",
            "ngrams": s.lm.n_ngrams,
            "candidate_keys": len(s.candidates),
        }

    @app.post("/restore", response_model=RestoreResponse)
    def restore(request: Request, body: RestoreRequest):
        if len(body.text) > MAX_CHARS:
            raise HTTPException(413, f"text longer than {MAX_CHARS} characters")
        t0 = time.perf_counter()
        text = nfc(body.text)
        out, details = request.app.state.restorer.restore_detailed(text)
        words = []
        for slot, _choice in details:
            tok = slot.token
            alts = [set_tone_style(transfer_case(tok.text, f), body.style) for f in slot.forms]
            output = set_tone_style(out[tok.start : tok.end], body.style)
            words.append(
                Word(
                    start=tok.start,
                    end=tok.end,
                    input=tok.text,
                    output=output,
                    changed=output != tok.text,
                    alternatives=alts,
                )
            )
        return RestoreResponse(
            text=set_tone_style(out, body.style),
            words=words,
            elapsed_ms=round(1000 * (time.perf_counter() - t0), 2),
        )

    return app


app = create_app()
