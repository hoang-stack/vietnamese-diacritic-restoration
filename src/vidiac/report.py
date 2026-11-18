"""Figures and reports/results.md from results/runs.jsonl (+ significance and confusions)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
SERIES = {  # fixed colour + marker per model family
    "lm": ("#2a78d6", "o"),
    "classifier": ("#eb6834", "s"),
    "mostfreq": ("#1baf7a", "D"),
}
ACCENT = "#2a78d6"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
        }
    )


def load_runs(results: Path) -> list[dict]:
    """Latest row per (stage, name, split, n_articles)."""
    rows: dict[tuple, dict] = {}
    path = results / "runs.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            rows[(r["stage"], r["name"], r["split"], r.get("n_articles"))] = r
    return list(rows.values())


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_curve(runs, out: Path, transformer_acc: float | None) -> Path | None:
    rows = [r for r in runs if r["stage"] == "curve"]
    if not rows:
        return None
    labels = {
        "mostfreq": "Most frequent form",
        "classifier": "Context classifier",
        "lm": "4-gram LM + beam search",
    }
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    for kind in ("lm", "classifier", "mostfreq"):
        pts = sorted((r["train_tokens"], r["accuracy"]) for r in rows if r["name"] == kind)
        if not pts:
            continue
        color, marker = SERIES[kind]
        ax.plot(
            [p[0] for p in pts],
            [100 * p[1] for p in pts],
            marker=marker,
            ms=6,
            color=color,
            mec=SURFACE,
            mew=1.2,
            label=labels[kind],
        )
        ax.annotate(
            f"{100 * pts[-1][1]:.2f}%",
            pts[-1] and (pts[-1][0], 100 * pts[-1][1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK_2,
        )
    if transformer_acc is not None:
        full = max(r["train_tokens"] for r in rows)
        ax.plot(
            [full],
            [100 * transformer_acc],
            "*",
            ms=12,
            color="#6250d6",
            mec=SURFACE,
            mew=1.0,
            label="Transformer (GPU)",
        )
    ax.set_xscale("log")
    ax.set_xlabel("training tokens (log scale)")
    ax.set_ylabel("validation word accuracy (%)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("More text helps every model, the LM most")
    return _save(fig, out / "learning_curve.png")


def plot_tradeoffs(runs, out: Path) -> Path | None:
    beam = sorted(
        (r["beam"], r["accuracy"], r["words_per_sec"]) for r in runs if r["stage"] == "beam"
    )
    prune = sorted(
        (r["model_mb"], r["accuracy"], r["prune"]) for r in runs if r["stage"] == "prune"
    )
    if not beam and not prune:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    if beam:
        ax = axes[0]
        ax.plot(
            [b[0] for b in beam],
            [100 * b[1] for b in beam],
            "-o",
            ms=5,
            color=ACCENT,
            mec=SURFACE,
            mew=1.2,
        )
        for b, acc, wps in beam:
            ax.annotate(
                f"{wps / 1000:.0f}k w/s",
                (b, 100 * acc),
                xytext=(0, -14),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=MUTED,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks([b[0] for b in beam], [str(b[0]) for b in beam])
        ax.set_xlabel("beam width (labels: decoding speed)")
        ax.set_ylabel("validation accuracy (%)")
        ax.set_title("Beam width: accuracy vs speed")
    if prune:
        ax = axes[1]
        ax.plot(
            [p[0] for p in prune],
            [100 * p[1] for p in prune],
            "-o",
            ms=5,
            color=ACCENT,
            mec=SURFACE,
            mew=1.2,
        )
        for mb, acc, k in prune:
            ax.annotate(
                "no pruning" if k == 0 else f"count < {k} pruned",
                (mb, 100 * acc),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=8,
                color=INK_2,
            )
        ax.set_xlabel("n-gram model size (MB, in memory)")
        ax.set_title("Pruning: model size vs accuracy")
        ax.margins(x=0.25, y=0.3)
    fig.tight_layout()
    return _save(fig, out / "tradeoffs.png")


def plot_test(runs, out: Path) -> Path | None:
    rows = [r for r in runs if r["split"] == "test"]
    if not rows:
        return None
    rows.sort(key=lambda r: r["accuracy"])
    fig, ax = plt.subplots(figsize=(8.6, 0.55 * len(rows) + 1.2))
    for y, r in enumerate(rows):
        ax.plot([100 * r["ci_low"], 100 * r["ci_high"]], [y, y], color=ACCENT, alpha=0.45)
        ax.plot(100 * r["accuracy"], y, "o", ms=7, color=ACCENT, mec=SURFACE, mew=1.5, zorder=3)
        ax.annotate(
            f"{100 * r['accuracy']:.2f}%",
            (100 * r["ci_high"], y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK_2,
        )
    ax.set_yticks(range(len(rows)), [r["name"] for r in rows])
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    lo = min(100 * r["ci_low"] for r in rows if r["accuracy"] > 0.9)
    ax.set_xlim(lo - 0.5, 100.2)
    ax.set_xlabel("test word accuracy (%), 95% bootstrap CI over articles")
    ax.set_title("Test accuracy (3,000 held-out articles)")
    return _save(fig, out / "test_accuracy.png")


def results_markdown(runs, results: Path, kaggle: dict | None) -> str:
    def table(rows, cols):
        head = "| " + " | ".join(c[0] for c in cols) + " |"
        sep = "| " + " | ".join("---" if i == 0 else "---:" for i in range(len(cols))) + " |"
        body = ["| " + " | ".join(c[1](r) for c in cols) + " |" for r in rows]
        return "\n".join([head, sep, *body])

    pct = lambda k: lambda r: f"{100 * r[k]:.2f}%" if r.get(k) is not None else "–"  # noqa: E731
    lines = ["# Results", ""]
    test = sorted([r for r in runs if r["split"] == "test"], key=lambda r: -r["accuracy"])
    if test:
        lines += [
            "## Test (3,000 held-out articles)",
            "",
            table(
                test,
                [
                    ("Model", lambda r: r["name"]),
                    ("Word accuracy", pct("accuracy")),
                    ("95% CI", lambda r: f"{100 * r['ci_low']:.2f}–{100 * r['ci_high']:.2f}%"),
                    ("Ambiguous words", pct("ambiguous_accuracy")),
                    ("Sentence accuracy", pct("sentence_accuracy")),
                    (
                        "Words/s",
                        lambda r: f"{r['words_per_sec']:,.0f}" if r.get("words_per_sec") else "–",
                    ),
                ],
            ),
            "",
        ]
        sig_path = results / "significance.json"
        if sig_path.exists():
            sig = json.loads(sig_path.read_text())
            lines += [
                f"Paired bootstrap against **{sig['best']}** (resampling articles):",
                "",
                "| vs. | Δ accuracy | 95% CI | p |",
                "| --- | ---: | ---: | ---: |",
            ]
            for name, s in sorted(sig["vs"].items(), key=lambda kv: kv[1]["diff"]):
                p = "< 0.001" if s["p_value"] < 0.001 else f"{s['p_value']:.3f}"
                lines.append(
                    f"| {name} | {100 * s['diff']:+.2f} pp | {100 * s['ci_low']:+.2f} to "
                    f"{100 * s['ci_high']:+.2f} | {p} |"
                )
            lines.append("")
    for stage, title in (
        ("orders", "N-gram order"),
        ("beam", "Beam width"),
        ("prune", "Pruning"),
        ("hybrid", "LM + context classifier"),
        ("neural", "Transformer (+ LM)"),
    ):
        rows = [r for r in runs if r["stage"] == stage and r["split"] == "validation"]
        if not rows:
            continue
        cols = [
            ("Run", lambda r: r["name"]),
            ("Accuracy", pct("accuracy")),
            ("Ambiguous", pct("ambiguous_accuracy")),
        ]
        if any("words_per_sec" in r for r in rows):
            cols.append(("Words/s", lambda r: f"{r.get('words_per_sec', 0):,.0f}"))
        if any("model_mb" in r for r in rows):
            cols.append(("Model MB", lambda r: f"{r.get('model_mb', 0):,.0f}"))
        lines += [f"## Validation: {title}", "", table(rows, cols), ""]
    curve = sorted(
        [r for r in runs if r["stage"] == "curve"], key=lambda r: (r["name"], r["n_articles"])
    )
    if curve:
        lines += [
            "## Validation: learning curve",
            "",
            table(
                curve,
                [
                    ("Model", lambda r: r["name"]),
                    ("Training articles", lambda r: f"{r['n_articles']:,}"),
                    ("Training tokens", lambda r: f"{r['train_tokens']:,}"),
                    ("Accuracy", pct("accuracy")),
                ],
            ),
            "",
        ]
    conf_path = results / "confusions.json"
    if conf_path.exists():
        conf = json.loads(conf_path.read_text())
        best = max(test, key=lambda r: r["accuracy"])["name"] if test else next(iter(conf))
        lines += [
            f"## Most frequent errors of {best} (test)",
            "",
            "| Correct | Predicted | Count |",
            "| --- | --- | ---: |",
        ]
        lines += [f"| {g} | {p} | {n} |" for g, p, n in conf.get(best, [])[:20]]
        lines.append("")
    if kaggle:
        lines += [
            "## Transformer training (Kaggle)",
            "",
            f"{kaggle['parameters'] / 1e6:.1f}M parameters on {kaggle['device']}, "
            f"config: `{json.dumps(kaggle['config'])}`",
            "",
            "| Epoch | Train loss | Val accuracy | Seconds |",
            "| ---: | ---: | ---: | ---: |",
        ]
        lines += [
            f"| {h['epoch']} | {h['train_loss']:.4f} | {100 * h['val']['accuracy']:.2f}% | "
            f"{h['epoch_seconds']:.0f} |"
            for h in kaggle["history"]
        ]
        lines.append("")
    return "\n".join(lines)


def build_report(
    results_dir="results", report_dir="reports", kaggle_dir="kaggle/output/main"
) -> Path:
    results, out = Path(results_dir), Path(report_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    _style()
    runs = load_runs(results)
    kpath = Path(kaggle_dir) / "metrics.json"
    kaggle = json.loads(kpath.read_text()) if kpath.exists() else None
    t_acc = None
    if kaggle:
        t_acc = kaggle["history"][-1]["val"]["accuracy"]
    plot_curve(runs, out / "figures", t_acc)
    plot_tradeoffs(runs, out / "figures")
    plot_test(runs, out / "figures")
    md = out / "results.md"
    md.write_text(results_markdown(runs, results, kaggle), encoding="utf-8")
    return md
