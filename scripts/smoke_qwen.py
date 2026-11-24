#!/usr/bin/env python3
"""Smoke test for the Qwen 3.8 27B endpoint started by slurm/serve_qwen.slurm.

Checks that the model is listed; warms the server up with untimed requests (the first requests of a fresh server pay
for one-time kernel compiles); then times a grounded FAQ reply with thinking off (time to first token, decode speed,
whether it uses the FAQ) and a burst of parallel requests (combined throughput); and checks that a request can still
turn thinking on. Standard library only, so it runs from any Python 3.8+: on the GPU node, or on a laptop through an
SSH tunnel. Exits 1 if any check fails.

    python scripts/smoke_qwen.py --endpoint /scratch/$USER/cs6120/run/qwen_endpoint.env
    python scripts/smoke_qwen.py --base-url http://localhost:8000/v1 --api-key "$QWEN_API_KEY" --quick
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
import urllib.error
import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # bypass the cluster's proxy for this server
# Qwen's sampling for non-thinking mode (model card), without its presence_penalty of 1.5, which pushes the model away
# from repeating the FAQ's wording
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}

SYSTEM = ("You are a customer support assistant for a bank. Answer the customer using only the FAQ entries below. "
          "If they do not answer the question, say that a human agent will follow up. "
          "Reply in at most 120 words, in a friendly and professional tone.")
FAQ = [  # mock entries in the shape of our knowledge base: (intent, question, answer)
    ("declined_card_payment", "Why was my card payment declined?",
     "A card payment can be declined if your balance is too low, the card is frozen in the app, the payment is over "
     "your daily spending limit, or the merchant is in a category you have blocked. Check the card's status and "
     "limits under Cards in the app. If none of these apply, contact us so we can review the payment."),
    ("pending_card_payment", "Why is my card payment still pending?",
     "Card payments stay pending until the merchant confirms them, which usually takes up to 3 business days. "
     "The amount is held from your balance until then."),
    ("card_not_working", "My card isn't working at all.",
     "If your card is declined everywhere, make sure it has been activated in the app and has not expired. "
     "A damaged chip can also stop it working; you can order a replacement under Cards."),
]
QUERY = ("Hi, my card got declined at the supermarket this morning even though I have money in my account. "
         "What's going on?")
GROUNDING_TERMS = ["frozen", "limit", "merchant", "categor", "activat", "expir", "chip"]  # FAQ facts beyond "balance"
BURST_QUERIES = [
    "Why is the coffee I bought yesterday still showing as pending?",
    "My new card doesn't work anywhere. Help?",
    "An online payment was declined a minute ago. Why?",
    "How long until a pending card payment goes through?",
    "Could a blocked merchant category be why my payment failed?",
    "The chip on my card is scratched and now it doesn't work.",
    "Is there a daily limit on card spending?",
    "Can I get a loan to buy a car?",  # not in the FAQ: should hand over to a human agent
]


def read_env(path: str) -> dict[str, str]:
    with open(path) as f:
        return dict(line.strip().split("=", 1) for line in f if "=" in line)


def call(base: str, key: str, path: str, payload: dict | None = None, timeout: float = 600):
    req = urllib.request.Request(
        base.rstrip("/") + path, data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})})
    try:
        return OPENER.open(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} on {path}: {e.read()[:300].decode('utf-8', 'replace')}") from None


def messages(query: str) -> list[dict]:
    faq = "\n\n".join(f"[{i}] ({intent}) Q: {q}\nA: {a}" for i, (intent, q, a) in enumerate(FAQ, 1))
    return [{"role": "system", "content": f"{SYSTEM}\n\nFAQ entries:\n{faq}"}, {"role": "user", "content": query}]


def chat(base: str, key: str, model: str, msgs: list[dict], max_tokens: int, stream: bool = False, **extra) -> dict:
    payload = {"model": model, "messages": msgs, "max_tokens": max_tokens, **SAMPLING, **extra}
    t0 = time.perf_counter()
    if not stream:
        with call(base, key, "/chat/completions", payload) as r:
            out = json.load(r)
        choice, usage = out["choices"][0], out.get("usage") or {}
        msg = choice["message"]
        return {"content": msg.get("content") or "", "finish": choice.get("finish_reason"),
                "reasoning": msg.get("reasoning") or msg.get("reasoning_content") or "",
                "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0),
                "seconds": time.perf_counter() - t0}
    payload.update(stream=True, stream_options={"include_usage": True})
    first, content, reasoning, finish, usage = None, [], [], None, {}
    with call(base, key, "/chat/completions", payload) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            usage = event.get("usage") or usage
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content") or ""
                think = delta.get("reasoning") or delta.get("reasoning_content") or ""
                if (piece or think) and first is None:
                    first = time.perf_counter()
                content.append(piece)
                reasoning.append(think)
                finish = choice.get("finish_reason") or finish
    t1 = time.perf_counter()
    n = usage.get("completion_tokens", 0)
    return {"content": "".join(content), "reasoning": "".join(reasoning), "finish": finish,
            "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": n, "seconds": t1 - t0,
            "ttft_s": first - t0 if first else None,
            "decode_tok_s": (n - 1) / (t1 - first) if first and n > 1 and t1 > first else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--endpoint", help="env file from slurm/serve_qwen.slurm (QWEN_BASE_URL, QWEN_API_KEY, QWEN_MODEL)")
    ap.add_argument("--base-url", help="for example http://localhost:8000/v1")
    ap.add_argument("--api-key")
    ap.add_argument("--model")
    ap.add_argument("--quick", action="store_true", help="only the model list and one grounded reply")
    ap.add_argument("--burst", type=int, default=8, help="parallel requests in the throughput check (default 8)")
    ap.add_argument("--no-warmup", action="store_true", help="time a cold server: skip the untimed warm-up requests")
    ap.add_argument("--out", help="write the results here as JSON (the API key is left out)")
    ap.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE", help="extra fields for --out")
    args = ap.parse_args()
    ep = read_env(args.endpoint) if args.endpoint else {}
    base = args.base_url or ep.get("QWEN_BASE_URL")
    key = args.api_key or ep.get("QWEN_API_KEY", "")
    model = args.model or ep.get("QWEN_MODEL", "qwen3.8-27b")
    if not base:
        ap.error("give --endpoint or --base-url")

    checks, metrics, reply = [], {}, None
    burst_queries = (BURST_QUERIES * (args.burst // len(BURST_QUERIES) + 1))[:args.burst]

    def check(name: str, fn) -> None:
        try:
            ok, detail = fn()
        except Exception as e:  # report and keep going: later checks still tell us something
            ok, detail = False, f"{type(e).__name__}: {e}"
        checks.append({"name": name, "ok": ok, "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)

    def models_listed():
        with call(base, key, "/models") as r:
            ids = [m["id"] for m in json.load(r)["data"]]
        return model in ids, f"served: {', '.join(ids)}"

    def warm_up():
        t0 = time.perf_counter()
        chat(base, key, model, messages(QUERY), max_tokens=64)
        with cf.ThreadPoolExecutor(max_workers=args.burst) as pool:
            list(pool.map(lambda q: chat(base, key, model, messages(q), max_tokens=64), burst_queries))
        metrics["warmup_s"] = time.perf_counter() - t0
        return True, f"1 + {args.burst} untimed requests in {metrics['warmup_s']:.1f} s"

    def grounded():
        nonlocal reply
        r = chat(base, key, model, messages(QUERY), max_tokens=300, stream=True)
        reply = r["content"]
        used = [t for t in GROUNDING_TERMS if t in reply.lower()]
        metrics.update(grounded_ttft_s=r["ttft_s"], grounded_decode_tok_s=r["decode_tok_s"],
                       grounded_prompt_tokens=r["prompt_tokens"], grounded_completion_tokens=r["completion_tokens"])
        no_think = not r["reasoning"].strip() and "<think>" not in reply and "</think>" not in reply
        ok = bool(reply.strip()) and no_think and r["finish"] == "stop" and len(used) >= 2
        ttft = f"{r['ttft_s']:.2f} s" if r["ttft_s"] is not None else "?"
        speed = f"{r['decode_tok_s']:.1f} tok/s" if r["decode_tok_s"] else "?"
        return ok, (f"{r['completion_tokens']} tokens, first token {ttft}, {speed}, finish={r['finish']}, "
                    f"thinking text: {'none' if no_think else repr(r['reasoning'][:80])}, "
                    f"FAQ terms used: {', '.join(used) or 'none'}")

    def burst():
        queries = burst_queries
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=args.burst) as pool:
            results = list(pool.map(lambda q: chat(base, key, model, messages(q), max_tokens=200), queries))
        wall = time.perf_counter() - t0
        tokens = sum(r["completion_tokens"] for r in results)
        good = sum(bool(r["content"].strip()) and not r["reasoning"].strip() for r in results)
        metrics.update(burst_requests=args.burst, burst_wall_s=wall, burst_completion_tokens=tokens,
                       burst_tok_s=tokens / wall if wall else None)
        handover = results[queries.index(BURST_QUERIES[-1])]["content"] if BURST_QUERIES[-1] in queries else ""
        return good == len(results), (f"{good}/{len(results)} answered without thinking text; {tokens} tokens in "
                                      f"{wall:.1f} s = {tokens / wall:.0f} tok/s combined; out-of-FAQ question "
                                      f"{'hands over' if 'agent' in handover.lower() else 'answer: ' + repr(handover[:80])}")

    def thinking_on():
        r = chat(base, key, model, [{"role": "user", "content": "What is 17 + 25? Reply with just the number."}],
                 max_tokens=1024, chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "low"})
        ok = bool(r["reasoning"].strip()) and "42" in r["content"]
        return ok, (f"{len(r['reasoning'])} chars of thinking, answer {r['content'].strip()[:20]!r}, "
                    f"finish={r['finish']}, {r['completion_tokens']} tokens in {r['seconds']:.1f} s")

    check("model listed", models_listed)
    if not args.no_warmup:
        check("warm-up", warm_up)
    check("grounded reply, thinking off", grounded)
    if not args.quick:
        check(f"burst of {args.burst} parallel requests", burst)
        check("thinking on when a request asks", thinking_on)
    if reply is not None:
        print("--- grounded reply ---\n" + reply.strip() + "\n---")

    passed = all(c["ok"] for c in checks)
    print(f"SMOKE {'PASS' if passed else 'FAIL'} ({sum(c['ok'] for c in checks)}/{len(checks)} checks)")
    if args.out:
        meta = dict(m.split("=", 1) for m in args.meta if "=" in m)
        result = {"base_url": base, "model": model, **meta, "passed": passed, "checks": checks, "metrics": metrics,
                  "query": QUERY, "grounded_reply": reply, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
