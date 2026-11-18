"""All classical (CPU) experiments: tuning on validation, then one evaluation on test.

Every stage appends rows to ``results/runs.jsonl`` as soon as they finish, so a long run that is
interrupted keeps what it already measured, and a stage can be re-run on its own.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from vidiac import data
from vidiac.corpus import load_corpus
from vidiac.evaluate import EvalResult, evaluate, paired_bootstrap
from vidiac.system import System, SystemConfig, build_system

log = logging.getLogger("vidiac")

VAL_ARTICLES, TEST_ARTICLES = 1000, 3000
VAL_SEED, TEST_SEED = 1, 2


def sample_articles(split: str, n: int, seed: int, data_dir="data") -> tuple[np.ndarray, list[str]]:
    """A fixed random sample of n articles (sorted ids, texts). Same sample for every model."""
    arts = list(data.articles(split, data_dir))
    idx = np.sort(np.random.default_rng(seed).choice(len(arts), n, replace=False))
    return idx, [arts[i] for i in idx]


class Recorder:
    def __init__(self, out_dir: Path):
        self.out = out_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / "runs.jsonl"

    def add(self, stage: str, name: str, split: str, res: EvalResult, **extra) -> dict:
        row = {"stage": stage, "name": name, "split": split, **res.summary, **extra}
        if split == "test":
            row.update({k: v for k, v in res.bootstrap().items() if k != "accuracy"})
        with open(self.path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        log.info(
            "%s | %s | %s: acc %.4f (amb %.4f)",
            stage,
            split,
            name,
            row["accuracy"],
            row["ambiguous_accuracy"],
        )
        return row


def timed_eval(restorer, golds, system: System) -> tuple[EvalResult, float]:
    t0 = time.perf_counter()
    res = evaluate(restorer, golds, system.candidates)
    return res, res.summary["words"] / (time.perf_counter() - t0)


def run(
    stages: list[str],
    data_dir="data",
    out_dir="results",
    kaggle_dir=None,
    neural_weight=None,
    clf_weight=None,
) -> None:
    rec = Recorder(Path(out_dir))
    train = load_corpus("train", data_dir)
    _, val = sample_articles("validation", VAL_ARTICLES, VAL_SEED, data_dir)
    log.info(
        "train: %d articles, %d tokens; val sample: %d articles",
        train.n_articles,
        train.n_tokens,
        len(val),
    )

    if "orders" in stages:  # baselines + n-gram order, full training data
        for order in (2, 3, 4):
            t0 = time.perf_counter()
            system = build_system(
                train, SystemConfig(order=order), with_classifier=False, log=log.info
            )
            fit = time.perf_counter() - t0
            if order == 2:
                for kind in ("identity", "mostfreq"):
                    res, wps = timed_eval(system.restorer(kind), val, system)
                    rec.add("orders", kind, "validation", res, words_per_sec=wps)
            res, wps = timed_eval(system.restorer("lm"), val, system)
            rec.add(
                "orders",
                f"lm{order}",
                "validation",
                res,
                order=order,
                words_per_sec=wps,
                fit_seconds=fit,
                model_mb=system.lm.nbytes / 1e6,
                ngrams=system.lm.n_ngrams,
            )

    if "beam" in stages:  # beam width vs accuracy and speed (order 4)
        system = build_system(train, SystemConfig(order=4), with_classifier=False, log=log.info)
        for beam in (1, 2, 4, 8, 16, 32):
            res, wps = timed_eval(system.restorer("lm", beam=beam), val, system)
            rec.add("beam", f"beam{beam}", "validation", res, beam=beam, words_per_sec=wps)

    if "prune" in stages:  # model size vs accuracy
        for prune in (0, 2, 3, 5):
            system = build_system(
                train, SystemConfig(order=4, prune=prune), with_classifier=False, log=log.info
            )
            res, wps = timed_eval(system.restorer("lm"), val, system)
            rec.add(
                "prune",
                f"prune{prune}",
                "validation",
                res,
                prune=prune,
                words_per_sec=wps,
                model_mb=system.lm.nbytes / 1e6,
                ngrams=system.lm.n_ngrams,
            )

    if "curve" in stages:  # learning curve: how much text does each model need?
        # Nested *random* subsets: the corpus is ordered (by source/date), so its first n
        # articles are not a representative sample and would distort the curve.
        perm = np.random.default_rng(0).permutation(train.n_articles)
        for n in (1000, 3000, 10000, 30000, train.n_articles):
            sub = train.subset(np.sort(perm[:n]))
            cfg = SystemConfig(order=4, classifier={"epochs": 1})
            system = build_system(sub, cfg, log=log.info)
            words = int(sub.n_tokens)
            for kind in ("mostfreq", "lm", "classifier"):
                res, _ = timed_eval(system.restorer(kind), val, system)
                rec.add("curve", kind, "validation", res, n_articles=n, train_tokens=words)

    if "hybrid" in stages:  # does the bidirectional classifier add anything to the LM?
        system = build_system(train, SystemConfig(order=4, classifier={"epochs": 2}), log=log.info)
        res, wps = timed_eval(system.restorer("classifier"), val, system)
        rec.add(
            "hybrid",
            "classifier",
            "validation",
            res,
            words_per_sec=wps,
            classifier_mb=system.scorer.classifier.nbytes / 1e6,
        )
        for w in (0.0, 0.1, 0.25, 0.5, 1.0):
            res, wps = timed_eval(system.restorer("hybrid", weight=w), val, system)
            rec.add("hybrid", f"hybrid_w{w}", "validation", res, weight=w, words_per_sec=wps)

    kdir = Path(kaggle_dir) if kaggle_dir else None
    if "neural" in stages:
        system = build_system(train, SystemConfig(order=4), with_classifier=False, log=log.info)
        neural_weight = neural_stage(rec, system, val, kdir)
        log.info("best Transformer + LM weight on validation: %s", neural_weight)
    if "test" in stages:
        test_stage(rec, train, data_dir, Path(out_dir), kdir, neural_weight, clf_weight)


def load_neural(split: str, kaggle_dir: Path):
    """Restored texts and per-slot log-probs written by kaggle/train_transformer.py."""
    from vidiac.restore import PrecomputedScorer

    name = {"validation": "val", "test": "test"}[split]
    lines = (kaggle_dir / f"outputs_{name}.jsonl").read_text(encoding="utf-8").splitlines()
    outputs = [json.loads(line) for line in lines]
    z = np.load(kaggle_dir / f"priors_{name}.npz")
    return outputs, PrecomputedScorer(z["values"], z["lengths"])


def neural_stage(rec: Recorder, system: System, val: list[str], kaggle_dir: Path) -> float:
    """Transformer alone and Transformer + LM on validation; returns the best weight."""
    from vidiac.evaluate import score_outputs
    from vidiac.restore import LMRestorer

    outputs, _ = load_neural("validation", kaggle_dir)
    rec.add("neural", "transformer", "validation", score_outputs(val, outputs, system.candidates))
    best_w, best_acc = 0.0, -1.0
    for w in (0.25, 0.5, 1.0, 2.0, 4.0):
        _, scorer = load_neural("validation", kaggle_dir)
        r = LMRestorer(
            system.candidates,
            system.lm,
            system.vocab,
            beam=system.config.beam,
            scorer=scorer,
            weight=w,
        )
        res, wps = timed_eval(r, val, system)
        rec.add("neural", f"transformer+lm_w{w}", "validation", res, weight=w)
        if res.summary["accuracy"] > best_acc:
            best_w, best_acc = w, res.summary["accuracy"]
    return best_w


def test_stage(
    rec: Recorder,
    train,
    data_dir,
    out: Path,
    kaggle_dir: Path | None,
    neural_weight: float | None,
    clf_weight: float | None,
) -> None:
    """Every final model, once, on the test sample; CIs, paired tests, confusions."""
    from vidiac.evaluate import score_outputs
    from vidiac.restore import LMRestorer

    _, test = sample_articles("test", TEST_ARTICLES, TEST_SEED, data_dir)
    system = build_system(train, SystemConfig(order=4, classifier={"epochs": 2}), log=log.info)
    results: dict[str, EvalResult] = {}

    def add(name, res, **extra):
        results[name] = res
        rec.add("test", name, "test", res, **extra)

    for kind, label in (
        ("mostfreq", "Most frequent form"),
        ("classifier", "Context classifier"),
        ("lm", "4-gram LM + beam search"),
    ):
        res, wps = timed_eval(system.restorer(kind), test, system)
        add(label, res, words_per_sec=wps)
    if clf_weight:
        res, wps = timed_eval(system.restorer("hybrid", weight=clf_weight), test, system)
        add("4-gram LM + classifier", res, words_per_sec=wps, weight=clf_weight)
    pruned = build_system(
        train, SystemConfig(order=4, prune=2), with_classifier=False, log=log.info
    )
    res, wps = timed_eval(pruned.restorer("lm"), test, pruned)
    add("4-gram LM, pruned (served)", res, words_per_sec=wps, model_mb=pruned.lm.nbytes / 1e6)
    if kaggle_dir is not None:
        outputs, _ = load_neural("test", kaggle_dir)
        add("Transformer", score_outputs(test, outputs, system.candidates))
        if neural_weight:
            _, scorer = load_neural("test", kaggle_dir)
            r = LMRestorer(
                system.candidates,
                system.lm,
                system.vocab,
                beam=system.config.beam,
                scorer=scorer,
                weight=neural_weight,
            )
            res, wps = timed_eval(r, test, system)
            add("Transformer + 4-gram LM", res, words_per_sec=wps, weight=neural_weight)

    best = max(results, key=lambda k: results[k].summary["accuracy"])
    sig = {k: paired_bootstrap(results[best], v) for k, v in results.items() if k != best}
    (out / "significance.json").write_text(json.dumps({"best": best, "vs": sig}, indent=2))
    np.savez_compressed(
        out / "test_per_article.npz",
        **{
            k.replace(" ", "_"): np.array([[a.words, a.correct] for a in v.articles])
            for k, v in results.items()
        },
    )
    conf = {
        k: [[g, p, n] for (g, p), n in v.confusions.most_common(40)] for k, v in results.items()
    }
    (out / "confusions.json").write_text(json.dumps(conf, ensure_ascii=False, indent=1))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stages",
        nargs="+",
        choices=["orders", "beam", "prune", "curve", "hybrid", "neural", "test"],
    )
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--kaggle-dir", default=None, help="downloaded Transformer outputs")
    p.add_argument("--neural-weight", type=float, default=None)
    p.add_argument("--clf-weight", type=float, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    run(args.stages, args.data_dir, args.out, args.kaggle_dir, args.neural_weight, args.clf_weight)


if __name__ == "__main__":
    main()


__all__ = ["paired_bootstrap", "run", "sample_articles"]
