"""Fit the gateway's verifier rule per question type, using the MAC's own verifier scores.

The 4060 and the Mac run slightly different 4-bit versions of the model, so scores shift
a little (mean 0.06). Thresholds are therefore refitted here, the same way as before:
    1. score every saved answer on the Mac, with the LoRA on (fine-tuned) and off (untrained)
    2. on the TRAINING half, set each threshold to catch as many wrong answers as the old rule
    3. pick fine-tuned vs untrained by precision on the TRAINING half (no peeking at test)
    4. report the test half, and save results/<task>/verifier_rule.json for the gateway

Usage:
    uv run python router/fit_verifier.py            # ~30 min the first time (scores are saved)
    uv run python router/fit_verifier.py --report   # refit + report from saved scores
"""

import argparse
import json
from pathlib import Path

from stage2_router import is_test

ROOT = Path(__file__).resolve().parent.parent
TYPES = ["math", "mmlu", "trivia", "code"]


def load(task, name):
    p = ROOT / "results" / task / f"{name}.jsonl"
    return {r["id"]: r for r in map(json.loads, open(p))} if p.exists() else {}


def score_all():
    from mlx_lm import load as mlx_load

    from verifier_mlx import add_lora, p_correct, remember_scale, set_lora
    model, tok = mlx_load("mlx-community/Qwen3.5-4B-MLX-4bit")
    wrapped = add_lora(model)
    remember_scale(wrapped)
    for task in TYPES:
        local, done, older = load(task, "local"), load(task, "verifier_mac"), None
        todo = [k for k in local if k not in done]
        print(f"{task}: {len(todo)} answers to score")
        with open(ROOT / "results" / task / "verifier_mac.jsonl", "a") as f:
            for n, k in enumerate(todo):
                r = local[k]
                msgs = r.get("messages")
                if msgs is None:                     # early math rows didn't save their prompt
                    from tasks import TASKS
                    if older is None:
                        older = {it["id"]: it for it in TASKS[task]["load"](10**6)}
                    msgs = TASKS[task]["messages"](older[k])
                set_lora(wrapped, True)
                ft = p_correct(model, tok, msgs, r["output"])
                set_lora(wrapped, False)
                base = p_correct(model, tok, msgs, r["output"])
                f.write(json.dumps({"id": k, "finetuned": round(ft, 5), "base": round(base, 5)}) + "\n")
                if n % 25 == 0:
                    f.flush()
                    print(f"  {n}/{len(todo)}")


def stats(rows, handed):
    wrong = [not r["local_ok"] for r in rows]
    caught = sum(h and w for h, w in zip(handed, wrong))
    acc = sum(r["cloud_ok"] if h else r["local_ok"] for r, h in zip(rows, handed)) / len(rows)
    return dict(acc=acc, prec=caught / sum(handed) if any(handed) else 1.0,
                rec=caught / sum(wrong) if any(wrong) else 1.0, handoff=sum(handed) / len(rows))


def threshold_for_recall(rows, key, target):
    wrong = sorted(r[key] for r in rows if not r["local_ok"])
    k = max(1, round(target * len(wrong) + 1e-9))
    return wrong[k - 1] + 1e-6


def fit():
    print(f"{'':7s} {'signal':10s} {'threshold':>9s} {'accuracy':>9s} {'precision':>10s} {'recall':>7s} {'handed off':>11s}")
    for task in TYPES:
        local, cloud, sigs, ver = (load(task, n) for n in ["local", "cloud", "signals", "verifier_mac"])
        old = json.loads((ROOT / "results" / task / "router_config.json").read_text())
        lp = old["min_logprob"] if old["min_logprob"] is not None else float("-inf")
        rows = [dict(test=is_test(k), local_ok=local[k]["correct"], cloud_ok=cloud[k]["correct"],
                     n=sigs[k]["n_tokens"], lp=sigs[k]["min_logprob"], **ver[k])
                for k in local if k in cloud and k in sigs and k in ver]
        train = [r for r in rows if not r["test"]]
        test = [r for r in rows if r["test"]]
        old_rule = lambda r: r["n"] > old["cap_tokens"] or r["lp"] < lp
        target = stats(train, [old_rule(r) for r in train])["rec"]

        best = None
        for key in ["base", "finetuned"]:
            t = threshold_for_recall(train, key, target)
            tr = stats(train, [r[key] < t for r in train])
            if best is None or tr["prec"] > best[2]["prec"]:
                best = (key, t, tr)
        key, t, _ = best
        s = stats(test, [r[key] < t for r in test])
        o = stats(test, [old_rule(r) for r in test])
        print(f"{task:7s} {'old rule':10s} {'':>9s} {o['acc']:9.1%} {o['prec']:10.0%} {o['rec']:7.0%} {o['handoff']:11.0%}")
        print(f"{'':7s} {key:10s} {t:9.3f} {s['acc']:9.1%} {s['prec']:10.0%} {s['rec']:7.0%} {s['handoff']:11.0%}")
        (ROOT / "results" / task / "verifier_rule.json").write_text(json.dumps(
            dict(signal=key, threshold=round(t, 4), n_test=len(test),
                 **{f"test_{k}": round(v, 4) for k, v in s.items()}), indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    if not ap.parse_args().report:
        score_all()
    fit()


if __name__ == "__main__":
    main()
