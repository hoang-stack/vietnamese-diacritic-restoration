"""Neural diacritic restoration: a bidirectional Transformer tagger (runs on a Kaggle GPU).

The n-gram LM reads left to right. This model reads the whole stripped sentence at once and,
for every ambiguous syllable, scores only that syllable's candidate forms (the same candidate
table the n-gram system uses) with a dot product between the contextual vector and an output
embedding of each accented form ("candidate-restricted softmax").

It reuses the project's own ``vidiac`` package (shipped in the Kaggle dataset) for tokenisation,
candidate generation, rendering and scoring, so its numbers are directly comparable with the
CPU models. Outputs in /kaggle/working:

* metrics.json            validation/test accuracy, timing, config
* outputs_{val,test}.jsonl restored text for the fixed evaluation samples
* priors_{val,test}.npz   per-slot log-probabilities, used locally for LM + Transformer decoding
* model.pt                fp16 weights
"""

from __future__ import annotations

import glob
import gzip
import json
import math
import os
import sys
import time
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CFG = {
    "d_model": 384,
    "n_layers": 6,
    "n_heads": 6,
    "d_ff": 1536,
    "dropout": 0.1,
    "max_len": 192,
    "tokens_per_batch": 16000,
    "epochs": int(os.environ.get("EPOCHS", 3)),
    "lr": 6e-4,
    "warmup": 2000,
    "weight_decay": 0.01,
    "n_input": 50000,
    "max_candidates": 16,
    "seed": 0,
    "train_articles": int(os.environ.get("TRAIN_ARTICLES", 0)) or None,  # None = all
}

OUT = Path("/kaggle/working")
t_start = time.time()


def log(*args):
    print(f"[{time.time() - t_start:7.0f}s]", *args, flush=True)


# --- locate the dataset (directories may arrive zipped) -----------------------------------------
res_path = Path(glob.glob("/kaggle/input/**/resources.json", recursive=True)[0])
ROOT = res_path.parent
for z in ROOT.glob("*.zip"):
    with zipfile.ZipFile(z) as zf:
        zf.extractall(OUT / "unzipped")
SRC = next(
    p for p in [ROOT / "src", OUT / "unzipped" / "src", OUT / "unzipped"] if (p / "vidiac").exists()
)
# Kaggle decompresses uploaded .gz files, so accept both train.txt.gz and train.txt.
PROC = next(
    p
    for p in [ROOT / "processed", OUT / "unzipped" / "processed", OUT / "unzipped"]
    if (p / "train.txt.gz").exists() or (p / "train.txt").exists()
)
sys.path.insert(0, str(SRC))

from vidiac.evaluate import score_outputs  # noqa: E402
from vidiac.restore import CandidateTable, make_slots, render  # noqa: E402
from vidiac.text import lm_sentences, sentences, strip_diacritics, tokenize  # noqa: E402

torch.manual_seed(CFG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log("device", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

# --- shared resources (identical to the CPU systems) --------------------------------------------
R = json.loads(res_path.read_text())
out_itos = R["vocab"]
out_stoi = {t: i for i, t in enumerate(out_itos)}
cands = CandidateTable({k: [tuple(f) for f in v] for k, v in R["candidates"].items()})
in_counts: Counter = Counter()
for t, c in zip(R["input_types"], R["input_counts"], strict=True):
    in_counts[strip_diacritics(t)] += c
in_itos = ["<pad>", "<unk>"] + [t for t, _ in in_counts.most_common(CFG["n_input"] - 2)]
in_stoi = {t: i for i, t in enumerate(in_itos)}
C = CFG["max_candidates"]
cand_tab = np.full((len(in_itos), C), -1, dtype=np.int64)
for key, forms in cands.table.items():
    i = in_stoi.get(key)
    if i is not None:
        ids = [out_stoi[f] for f, _ in forms][:C]
        cand_tab[i, : len(ids)] = ids
log(f"input vocab {len(in_itos)}, output vocab {len(out_itos)}, candidate keys {len(cands)}")


def read_articles(split):
    gz = PROC / f"{split}.txt.gz"
    if gz.exists():
        with gzip.open(gz, "rt", encoding="utf-8") as fh:
            return [line.rstrip("\n") for line in fh]
    with open(PROC / f"{split}.txt", encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh]


def encode_chunk(texts):
    xs, ys, lens = [], [], []
    for text in texts:
        for sent in lm_sentences(text):
            xs.extend(in_stoi.get(strip_diacritics(t), 1) for t in sent)
            ys.extend(out_stoi.get(t, -1) for t in sent)
            lens.append(len(sent))
    return np.array(xs, np.int32), np.array(ys, np.int32), np.array(lens, np.int64)


# --- training data ------------------------------------------------------------------------------
train_texts = read_articles("train")[: CFG["train_articles"]]
with Pool(os.cpu_count()) as pool:
    parts = pool.map(
        encode_chunk, [train_texts[i : i + 1000] for i in range(0, len(train_texts), 1000)]
    )
X = np.concatenate([p[0] for p in parts])
Y = np.concatenate([p[1] for p in parts])
L = np.concatenate([p[2] for p in parts])
del parts, train_texts
log(f"train: {len(L):,} sentences, {len(X):,} tokens")

# Split long sentences into max_len pieces, then bucket by length into token-budget batches.
starts = np.concatenate([[0], np.cumsum(L)[:-1]])
pieces = [
    (s + o, min(CFG["max_len"], n - o))
    for s, n in zip(starts, L, strict=True)
    for o in range(0, n, CFG["max_len"])
]
pieces = np.array(pieces, dtype=np.int64)
order = np.argsort(pieces[:, 1], kind="stable")
batches, cur, cur_max = [], [], 0
for idx in order:
    n = pieces[idx, 1]
    if cur and max(cur_max, n) * (len(cur) + 1) > CFG["tokens_per_batch"]:
        batches.append(np.array(cur))
        cur, cur_max = [], 0
    cur.append(idx)
    cur_max = max(cur_max, n)
if cur:
    batches.append(np.array(cur))
log(f"{len(pieces):,} pieces in {len(batches):,} batches per epoch")

cand_tab_t = torch.from_numpy(cand_tab).to(device)


def make_batch(piece_ids):
    lens = pieces[piece_ids, 1]
    T = int(lens.max())
    x = np.zeros((len(piece_ids), T), np.int64)
    y = np.full((len(piece_ids), T), -1, np.int64)
    for r, (s, n) in enumerate(pieces[piece_ids]):
        x[r, :n] = X[s : s + n]
        y[r, :n] = Y[s : s + n]
    return torch.from_numpy(x).to(device, non_blocking=True), torch.from_numpy(y).to(
        device, non_blocking=True
    )


# --- model --------------------------------------------------------------------------------------
class Tagger(nn.Module):
    def __init__(self, n_in, n_out, d, layers, heads, ff, drop, max_len):
        super().__init__()
        self.emb = nn.Embedding(n_in, d, padding_idx=0)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(
            d, heads, ff, drop, batch_first=True, norm_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Embedding(n_out, d)
        self.bias = nn.Embedding(n_out, 1)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=0.02)
        nn.init.zeros_(self.bias.weight)

    def encode(self, x):
        pos = torch.arange(x.shape[1], device=x.device)
        h = self.emb(x) + self.pos(pos)[None]
        return self.norm(self.enc(h, src_key_padding_mask=x == 0))

    def candidate_logits(self, h, cand):
        """h: (N, d) vectors of ambiguous positions; cand: (N, C) accented ids (-1 = none)."""
        c = cand.clamp(min=0)
        logits = torch.einsum("nd,ncd->nc", h, self.out(c)) + self.bias(c).squeeze(-1)
        return logits.float().masked_fill(cand < 0, -1e9)


model = Tagger(
    len(in_itos),
    len(out_itos),
    CFG["d_model"],
    CFG["n_layers"],
    CFG["n_heads"],
    CFG["d_ff"],
    CFG["dropout"],
    CFG["max_len"],
).to(device)
n_params = sum(p.numel() for p in model.parameters())
log(f"parameters: {n_params / 1e6:.1f}M")
opt = torch.optim.AdamW(
    model.parameters(), lr=CFG["lr"], betas=(0.9, 0.98), weight_decay=CFG["weight_decay"]
)
total_steps = CFG["epochs"] * len(batches)


def lr_at(step):
    if step < CFG["warmup"]:
        return CFG["lr"] * (step + 1) / CFG["warmup"]
    progress = (step - CFG["warmup"]) / max(1, total_steps - CFG["warmup"])
    return CFG["lr"] * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress)))


scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")


def ambiguous_targets(x, y):
    """Mask of positions with >= 2 candidates whose gold form is one of them, and its index."""
    cand = cand_tab_t[x]  # (B, T, C)
    n_c = (cand >= 0).sum(-1)
    hit = cand == y.unsqueeze(-1)
    mask = (n_c >= 2) & hit.any(-1)
    return mask, cand, hit.float().argmax(-1)


# --- evaluation on the fixed samples (same pipeline as the CPU systems) -------------------------
def slot_input_ids(sent_slots):
    return [in_stoi.get(strip_diacritics(s.token.lm_form), 1) for s in sent_slots]


@torch.no_grad()
def predict(golds, batch_sents=512, want_priors=False):
    """Restore stripped versions of ``golds``; optionally return per-slot log-probs."""
    model.eval()
    inputs = [strip_diacritics(g) for g in golds]
    per_text = [[make_slots(s, cands) for s in sentences(tokenize(t))] for t in inputs]
    flat = [s for sents in per_text for s in sents]
    choices, priors = [], []
    t0 = time.time()
    for b in range(0, len(flat), batch_sents):
        group = flat[b : b + batch_sents]
        T = min(CFG["max_len"], max(len(s) for s in group))
        x = np.zeros((len(group), T), np.int64)
        for r, s in enumerate(group):
            ids = slot_input_ids(s)[:T]
            x[r, : len(ids)] = ids
        xt = torch.from_numpy(x).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            h = model.encode(xt)
        for r, s in enumerate(group):
            pick, pri = [], []
            amb = [i for i, slot in enumerate(s) if len(slot.forms) > 1 and i < T]
            amb_set = set(amb)
            if amb:
                cand_np = np.full((len(amb), C), -1, dtype=np.int64)
                for k, i in enumerate(amb):
                    row = [out_stoi[f] for f in s[i].forms][:C]
                    cand_np[k, : len(row)] = row
                cand = torch.from_numpy(cand_np).to(device)
                lp = (
                    F.log_softmax(model.candidate_logits(h[r, amb].float(), cand), -1).cpu().numpy()
                )
            j = 0
            for i, slot in enumerate(s):
                if i in amb_set:
                    row = lp[j, : len(slot.forms)]
                    pick.append(int(row.argmax()))
                    pri.append(row.astype(np.float32))
                    j += 1
                else:
                    pick.append(0)
                    if len(slot.forms) > 1:  # beyond max_len: fall back to the most frequent form
                        pri.append(np.zeros(len(slot.forms), np.float32))
            choices.append(pick)
            priors.append(pri)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    outputs, k = [], 0
    for text, sents in zip(inputs, per_text, strict=True):
        slots = [s for sent in sents for s in sent]
        choice = [c for ch in choices[k : k + len(sents)] for c in ch]
        k += len(sents)
        outputs.append(render(text, slots, choice))
    model.train()
    flat_priors = [p for sent in priors for p in sent]
    return outputs, elapsed, flat_priors


def sample(split, ids):
    arts = read_articles(split)
    return [arts[i] for i in ids]


val_golds = sample("validation", R["val_ids"])
test_golds = sample("test", R["test_ids"])

# --- train --------------------------------------------------------------------------------------
history = []
step = 0
rng = np.random.default_rng(CFG["seed"])
for epoch in range(CFG["epochs"]):
    model.train()
    tot_loss, tot_n, t_ep = 0.0, 0, time.time()
    for bi in rng.permutation(len(batches)):
        x, y = make_batch(batches[bi])
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            h = model.encode(x)
            mask, cand, target = ambiguous_targets(x, y)
            logits = model.candidate_logits(h[mask], cand[mask])
            loss = F.cross_entropy(logits, target[mask])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        n = int(mask.sum())
        tot_loss += loss.detach().item() * n
        tot_n += n
        step += 1
        if step % 500 == 0:
            log(
                f"epoch {epoch + 1} step {step}/{total_steps} "
                f"loss {tot_loss / tot_n:.4f} lr {lr_at(step):.2e}"
            )
    outs, secs, _ = predict(val_golds)
    val = score_outputs(val_golds, outs, cands).summary
    history.append(
        {
            "epoch": epoch + 1,
            "train_loss": tot_loss / tot_n,
            "val": val,
            "epoch_seconds": time.time() - t_ep,
        }
    )
    log(
        f"epoch {epoch + 1} done in {time.time() - t_ep:.0f}s: val accuracy {val['accuracy']:.4f} "
        f"(ambiguous {val['ambiguous_accuracy']:.4f})"
    )

# --- final outputs ------------------------------------------------------------------------------
metrics = {
    "config": CFG,
    "parameters": n_params,
    "history": history,
    "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
}
for name, golds in (("val", val_golds), ("test", test_golds)):
    outs, secs, pri = predict(golds)
    res = score_outputs(golds, outs, cands)
    metrics[name] = {
        **res.summary,
        "bootstrap": res.bootstrap(),
        "gpu_seconds": secs,
        "words_per_sec": res.summary["words"] / secs,
    }
    with open(OUT / f"outputs_{name}.jsonl", "w", encoding="utf-8") as fh:
        for o in outs:
            fh.write(json.dumps(o, ensure_ascii=False) + "\n")
    lengths = np.array([len(p) for p in pri], np.int64)
    np.savez_compressed(
        OUT / f"priors_{name}.npz",
        values=np.concatenate(pri) if pri else np.zeros(0),
        lengths=lengths,
    )
    log(
        f"{name}: accuracy {res.summary['accuracy']:.4f}, "
        f"{metrics[name]['words_per_sec']:.0f} words/s"
    )

# CPU inference speed on a small sample (what serving without a GPU would look like).
model_cpu = model.float().to("cpu")
device_backup, device = device, torch.device("cpu")
model, cand_tab_t = model_cpu, cand_tab_t.cpu()
torch.set_num_threads(os.cpu_count())
outs, secs, _ = predict(val_golds[:100])
words = score_outputs(val_golds[:100], outs, cands).summary["words"]
metrics["cpu_words_per_sec"] = words / secs
log(f"CPU inference: {metrics['cpu_words_per_sec']:.0f} words/s on {os.cpu_count()} threads")

torch.save({k: v.half() for k, v in model.state_dict().items()}, OUT / "model.pt")
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
log("done")
