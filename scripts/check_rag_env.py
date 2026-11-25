#!/usr/bin/env python3
"""Check the retrieval/classification env: packages, the repo's BANKING77 copy, and one small run of each local model.

Each retriever (Qwen3-Embedding with FAISS, MiniLM, the bge reranker, BM25) must rank a declined-payment FAQ entry
first for a declined-payment question; a quick TF-IDF + LR fit must beat chance by far; DistilBERT must take one
training step. CPU by default (slurm/build_rag_env.slurm runs it that way); --device cuda checks a GPU node.

    module load python/3.13.5          # on Explorer: the env's Python needs the module's libpython
    /scratch/$USER/cs6120/venvs/rag/bin/python scripts/check_rag_env.py
    /scratch/$USER/cs6120/venvs/rag/bin/python scripts/check_rag_env.py --device cuda
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import os
import re
import sys
import time
from pathlib import Path

DISTS = ["torch", "transformers", "sentence-transformers", "accelerate", "datasets", "scikit-learn", "faiss-cpu",
         "rank-bm25", "pandas", "pyarrow", "numpy", "matplotlib", "openai"]
QUERY = "My card payment was declined at the shop today."
DOCS = [  # the first entry should rank first
    "Why was my card payment declined? A payment can be declined if your balance is too low, the card is frozen, "
    "or the payment is over your daily limit.",
    "Why is my card payment still pending? Payments stay pending until the merchant confirms them.",
    "How do I change my PIN? You can change your PIN at any of our ATMs.",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=f"/scratch/{os.environ.get('USER', '')}/cs6120",
                    help="folder with models/ from slurm/fetch_models.slurm")
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    args = ap.parse_args()
    models = Path(args.root) / "models"
    data = Path(__file__).resolve().parents[1] / "data" / "banking77"  # committed with the repo
    dev = args.device
    results = []

    def step(name, fn):
        t0 = time.perf_counter()
        try:
            ok, detail = fn()
        except Exception as e:  # report and keep going
            ok, detail = False, f"{type(e).__name__}: {e}"
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'} {name} ({time.perf_counter() - t0:.1f} s): {detail}", flush=True)

    def packages():
        return True, ", ".join(f"{d} {md.version(d)}" for d in DISTS)

    def torch_build():
        import torch
        cuda = torch.version.cuda or "none"
        detail = f"torch {torch.__version__}, CUDA build {cuda}, GPU visible: {torch.cuda.is_available()}"
        ok = cuda.startswith("12.8")  # Explorer's driver 570 runs up to CUDA 12.8
        if dev.startswith("cuda"):
            if not torch.cuda.is_available():
                return False, detail + "; --device cuda but no GPU"
            major, minor = torch.cuda.get_device_capability()
            arch = f"sm_{major}{minor}"
            x = torch.randn(1024, 1024, device=dev)
            float((x @ x).sum())
            ok = ok and arch in torch.cuda.get_arch_list()
            detail += f"; {torch.cuda.get_device_name()} ({arch}, in this build: {arch in torch.cuda.get_arch_list()})"
        return ok, detail

    def banking77():
        import pandas as pd
        train = pd.read_parquet(data / "train.parquet")
        test = pd.read_parquet(data / "test.parquet")
        names = pd.read_csv(data / "labels.csv")
        ok = (len(train) == 10003 and len(test) == 3080 and train["label"].nunique() == 77
              and list(names["label"]) == list(range(77)) and set(train["label"]) <= set(names["label"]))
        return ok, (f"train {len(train)}, test {len(test)}, {train['label'].nunique()} intents, columns {list(train)}, "
                    f"labels.csv: {len(names)} names (0 = {names['intent'][0]})")

    def tfidf_lr():
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        train = pd.read_parquet(data / "train.parquet")
        test = pd.read_parquet(data / "test.parquet")
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        clf = LogisticRegression(max_iter=1000).fit(vec.fit_transform(train["text"]), train["label"])
        pred = clf.predict(vec.transform(test["text"]))
        acc, f1 = accuracy_score(test["label"], pred), f1_score(test["label"], pred, average="macro")
        return acc > 0.5, f"untuned sanity fit: test accuracy {acc:.3f}, macro-F1 {f1:.3f}"

    def dense(name: str, dim: int):
        def run():
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(str(models / name), device=dev)
            kw = {"prompt_name": "query"} if "query" in (m.prompts or {}) else {}
            q = m.encode([QUERY], normalize_embeddings=True, **kw).astype(np.float32)
            d = m.encode(DOCS, normalize_embeddings=True).astype(np.float32)
            index = faiss.IndexFlatIP(d.shape[1])
            index.add(d)
            _, top = index.search(q, len(DOCS))
            ok = d.shape[1] == dim and int(top[0][0]) == 0
            return ok, f"dim {d.shape[1]}, FAISS order {top[0].tolist()}, cosine {np.round(q @ d.T, 3).tolist()[0]}"
        return run

    def reranker():
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(str(models / "bge-reranker-v2-m3"), device=dev)
        scores = ce.predict([(QUERY, d) for d in DOCS])
        order = sorted(range(len(DOCS)), key=lambda i: -scores[i])
        return order[0] == 0, f"order {order}, scores {[round(float(s), 3) for s in scores]}"

    def bm25():
        from rank_bm25 import BM25Okapi
        tok = lambda s: re.findall(r"\w+", s.lower())  # noqa: E731
        scores = BM25Okapi([tok(d) for d in DOCS]).get_scores(tok(QUERY))
        order = sorted(range(len(DOCS)), key=lambda i: -scores[i])
        return order[0] == 0, f"order {order}, scores {[round(float(s), 3) for s in scores]}"

    def distilbert():
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        path = str(models / "distilbert-base-uncased")
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=77).to(dev)
        batch = tok([QUERY] + DOCS, padding=True, truncation=True, return_tensors="pt").to(dev)
        labels = torch.tensor([0, 0, 1, 2], device=dev)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
        out = model(**batch, labels=labels)
        out.loss.backward()
        opt.step()
        ok = tuple(out.logits.shape) == (4, 77) and bool(torch.isfinite(out.loss))
        return ok, f"logits {tuple(out.logits.shape)}, loss {out.loss.item():.3f}, one AdamW step on {dev}"

    print(f"checking {sys.executable} on {dev}, models in {models}", flush=True)
    step("packages", packages)
    step("torch build", torch_build)
    step("BANKING77 files", banking77)
    step("TF-IDF + LR", tfidf_lr)
    step("Qwen3-Embedding-0.6B + FAISS", dense("Qwen3-Embedding-0.6B", 1024))
    step("all-MiniLM-L6-v2", dense("all-MiniLM-L6-v2", 384))
    step("bge-reranker-v2-m3", reranker)
    step("BM25", bm25)
    step("DistilBERT training step", distilbert)
    print(f"ENV CHECK {'PASS' if all(results) else 'FAIL'} ({sum(results)}/{len(results)})")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
