"""The VietNews corpus: download, clean, and serve articles as normalised text.

VietNews (Nguyen et al., 2019) is a news summarisation corpus. Its Hugging Face copy
(``harouzie/vietnews``, Apache-2.0) ships three article-level splits. The text has been
word-segmented (multi-syllable words joined with "_", e.g. "nhân_viên") and is not consistently
NFC-normalised, so :func:`clean` undoes both before anything else touches it.
"""

from __future__ import annotations

import gzip
import hashlib
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from vidiac.text import nfc

HF_BASE = "https://huggingface.co/datasets/harouzie/vietnews/resolve/main/data"
FILES = {
    "train": (
        "train-00000-of-00001-faa04287fdc0e299.parquet",
        "3cdfffd15803fa10ed8e78204dd96783dd9c1fbca34a978010985aeb3f5f39bb",
    ),
    "validation": (
        "validation-00000-of-00001-10a6e2d4aced7ee4.parquet",
        "002ef3e6a08598c798cc5fb32c8e98d1106cd2047c2783c7653c805f55ebe2e1",
    ),
    "test": (
        "test-00000-of-00001-d0d073af901f3b49.parquet",
        "0384f789702b5be10907a26739055ff5ae67a9cede00844182c326991df5290a",
    ),
}
SPLITS = tuple(FILES)


def clean(text: str) -> str:
    """Undo word segmentation, normalise to NFC and collapse whitespace."""
    return " ".join(nfc(text.replace("_", " ")).split())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(data_dir: str | Path = "data") -> Path:
    """Fetch the three parquet files (≈250 MB) and verify their checksums."""
    raw = Path(data_dir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.values():
        path = raw / name
        if not path.exists() or _sha256(path) != digest:
            urllib.request.urlretrieve(f"{HF_BASE}/{name}", path)
            if _sha256(path) != digest:
                path.unlink()
                raise RuntimeError(f"Checksum mismatch for {name}")
    return raw


def prepare(data_dir: str | Path = "data", keep_raw: bool = False) -> Path:
    """Write ``processed/<split>.txt.gz``: one cleaned article (title + summary + body) per line."""
    import pandas as pd

    raw = download(data_dir)
    out = Path(data_dir) / "processed"
    out.mkdir(parents=True, exist_ok=True)
    for split, (name, _) in FILES.items():
        df = pd.read_parquet(raw / name, columns=["title", "abstract", "article"])
        with gzip.open(out / f"{split}.txt.gz", "wt", encoding="utf-8") as fh:
            for title, abstract, article in df.itertuples(index=False):
                fh.write(clean(f"{title} . {abstract} {article}") + "\n")
        if not keep_raw:
            (raw / name).unlink()
    return out


def articles(split: str, data_dir: str | Path = "data", limit: int | None = None) -> Iterator[str]:
    path = Path(data_dir) / "processed" / f"{split}.txt.gz"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run `vidiac prepare` first")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for n, line in enumerate(fh):
            if limit is not None and n >= limit:
                return
            yield line.rstrip("\n")
