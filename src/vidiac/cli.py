"""Command line: ``vidiac {prepare,experiments,train,restore,serve,report}``."""

from __future__ import annotations

import argparse
import logging
import os
import sys


def _prepare(args) -> None:
    from vidiac.corpus import load_corpus
    from vidiac.data import prepare

    print(f"cleaned corpus in {prepare(args.data_dir)}")
    for split in ("train", "validation", "test"):
        c = load_corpus(split, args.data_dir)
        print(f"{split}: {c.n_articles:,} articles, {c.n_tokens:,} tokens")


def _experiments(args) -> None:
    from vidiac.experiments import run

    run(
        args.stages,
        args.data_dir,
        args.results_dir,
        args.kaggle_dir,
        args.neural_weight,
        args.clf_weight,
    )


def _train(args) -> None:
    from vidiac.corpus import load_corpus
    from vidiac.system import SystemConfig, build_system

    cfg = SystemConfig(order=args.order, prune=args.prune, beam=args.beam)
    system = build_system(load_corpus("train", args.data_dir), cfg, with_classifier=False)
    system.save(args.out)
    print(f"saved {args.out} ({system.lm.nbytes / 1e6:.0f} MB of n-grams)")


def _restore(args) -> None:
    from vidiac.system import System
    from vidiac.text import nfc, set_tone_style

    restorer = System.load(args.artifact).restorer("lm")
    text = " ".join(args.text) if args.text else sys.stdin.read()
    print(
        set_tone_style(restorer.restore(nfc(text)), args.style), end="" if not args.text else "\n"
    )


def _serve(args) -> None:
    import uvicorn

    os.environ["VIDIAC_ARTIFACT"] = args.artifact
    uvicorn.run("vidiac.api:app", host=args.host, port=args.port)


def _report(args) -> None:
    from vidiac.report import build_report

    print(f"report written to {build_report(args.results_dir, args.report_dir)}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="vidiac", description=__doc__)
    p.add_argument("--data-dir", default="data")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("prepare", help="download + clean VietNews, cache tokens").set_defaults(
        func=_prepare
    )
    ex = sub.add_parser("experiments", help="run CPU experiments on validation")
    ex.add_argument(
        "stages",
        nargs="+",
        choices=["orders", "beam", "prune", "curve", "hybrid", "neural", "test"],
    )
    ex.add_argument("--kaggle-dir", default=None)
    ex.add_argument("--neural-weight", type=float, default=None)
    ex.add_argument("--clf-weight", type=float, default=None)
    ex.add_argument("--results-dir", default="results")
    ex.set_defaults(func=_experiments)

    tr = sub.add_parser("train", help="fit the serving model on all training data")
    tr.add_argument("--order", type=int, default=4)
    tr.add_argument("--prune", type=int, default=2)
    tr.add_argument("--beam", type=int, default=16)
    tr.add_argument("--out", default="artifacts/vidiac.pkl")
    tr.set_defaults(func=_train)

    rs = sub.add_parser("restore", help="restore diacritics of TEXT (or stdin)")
    rs.add_argument("text", nargs="*")
    rs.add_argument("--artifact", default="artifacts/vidiac.pkl")
    rs.add_argument("--style", choices=["old", "new"], default="old")
    rs.set_defaults(func=_restore)

    sv = sub.add_parser("serve", help="REST API + demo page")
    sv.add_argument("--artifact", default="artifacts/vidiac.pkl")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=_serve)

    rp = sub.add_parser("report", help="figures + reports/results.md from results/")
    rp.add_argument("--results-dir", default="results")
    rp.add_argument("--report-dir", default="reports")
    rp.set_defaults(func=_report)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args.func(args)


if __name__ == "__main__":
    main()
