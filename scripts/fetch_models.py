#!/usr/bin/env python3
"""Download the project's models and the BANKING77 dataset, then check every file against the Hub's sizes.

Each repo is pinned to the revision it resolves to at download time. Files already present at the right size are
not downloaded again, so a rerun (for example after a scratch purge) fetches only what is missing. Revisions and
sizes go to <root>/manifest.json. Qwen 3.8 27B (55.6 GB) is not downloaded here: slurm/fetch_models.slurm links the
copy that is already on Explorer.

    python scripts/fetch_models.py --root /scratch/$USER/cs6120            # Explorer, via slurm/fetch_models.slurm
    python scripts/fetch_models.py --root ~/cs6120_assets                 # a laptop; keep it outside the repo
    python scripts/fetch_models.py --root ~/cs6120_assets --only MiniLM   # one item
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

# (repo id, repo type, folder under --root, files to skip: other frameworks' copies of the weights, images)
ITEMS = [
    ("Qwen/Qwen3-Embedding-0.6B", "model", "models/Qwen3-Embedding-0.6B", []),
    ("BAAI/bge-reranker-v2-m3", "model", "models/bge-reranker-v2-m3", ["assets/*"]),
    ("distilbert/distilbert-base-uncased", "model", "models/distilbert-base-uncased",
     ["*.msgpack", "*.h5", "*.ot", "pytorch_model.bin"]),
    ("sentence-transformers/all-MiniLM-L6-v2", "model", "models/all-MiniLM-L6-v2",
     ["onnx/*", "openvino/*", "*.h5", "*.ot", "pytorch_model.bin"]),
    ("PolyAI/banking77", "dataset", "data/banking77", []),
]


def problems(dest: Path, expected: dict[str, int]) -> list[str]:
    """Files that are missing or whose size differs from the Hub's."""
    return [p for p, size in expected.items() if not (dest / p).is_file() or (dest / p).stat().st_size != size]


def fetch(api: HfApi, root: Path, repo: str, repo_type: str, subdir: str, skip: list[str]) -> dict:
    dest = root / subdir
    rev = api.repo_info(repo, repo_type=repo_type).sha
    expected = {
        f.path: f.size
        for f in api.list_repo_tree(repo, repo_type=repo_type, revision=rev, recursive=True)
        if getattr(f, "size", None) is not None and not any(fnmatch.fnmatch(f.path, s) for s in skip)
    }
    stale = problems(dest, expected)
    if stale:
        # At a pinned revision, snapshot_download keeps any local file its metadata vouches for, whatever its size,
        # so remove partial or damaged files first to make it fetch them again.
        for p in stale:
            (dest / p).unlink(missing_ok=True)
        snapshot_download(repo, repo_type=repo_type, revision=rev, local_dir=dest, ignore_patterns=skip or None)
    bad = problems(dest, expected)
    if bad:
        raise RuntimeError(f"{len(bad)} file(s) missing or the wrong size after download: {', '.join(bad[:5])}")
    total = sum(expected.values())
    print(f"OK   {repo}@{rev[:12]}: {len(expected)} files, {total / 1e9:.2f} GB in {dest}", flush=True)
    return {"repo_type": repo_type, "revision": rev, "dir": str(dest), "files": len(expected), "bytes": total}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, type=Path, help="folder that gets models/, data/ and manifest.json")
    ap.add_argument("--only", help="fetch only the items whose repo id contains this text (case-insensitive)")
    args = ap.parse_args()
    root = args.root.expanduser().resolve()
    items = [item for item in ITEMS if not args.only or args.only.lower() in item[0].lower()]
    if not items:
        ap.error(f"--only {args.only!r} matches none of: {', '.join(item[0] for item in ITEMS)}")

    api = HfApi()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    failed = []
    for repo, *rest in items:
        try:
            manifest[repo] = fetch(api, root, repo, *rest)
        except Exception as e:  # keep going so one bad repo doesn't block the rest; the exit code reports it
            print(f"FAIL {repo}: {type(e).__name__}: {e}", flush=True)
            failed.append(repo)

    root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if failed:
        print(f"{len(failed)} of {len(items)} item(s) failed: {', '.join(failed)}")
        return 1
    print(f"all {len(items)} item(s) complete; manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
