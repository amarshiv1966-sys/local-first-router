"""Does knowing the question type help the router? Three versions on the same test questions:

    A  math's rule for every question             (what the gateway did before)
    B  each type's own rule, using the TRUE type  (the best case)
    C  each type's own rule, using the GUESSED type from classify.py (the realistic case)

Run fit_config.py for every task and classify.py first.

Usage:
    uv run python router/compare.py
"""

import json
from pathlib import Path

from classify import MIN_CONFIDENCE, TYPES
from stage2_router import is_test

RESULTS = Path("results")
FALLBACK = "math"       # rule used when the classifier is unsure


def load(task, name):
    with open(RESULTS / task / f"{name}.jsonl") as f:
        return {r["id"]: r for r in map(json.loads, f)}


def hands_off(sig, rule):
    lp = rule["min_logprob"] if rule["min_logprob"] is not None else float("-inf")
    return sig["n_tokens"] > rule["cap_tokens"] or sig["min_logprob"] < lp


def score(rows, rule_for):
    """rows: dicts with local_ok, cloud_ok, sig, task. rule_for(row) -> the rule to apply."""
    handed = [hands_off(r["sig"], rule_for(r)) for r in rows]
    acc = sum(r["cloud_ok"] if h else r["local_ok"] for r, h in zip(rows, handed)) / len(rows)
    wrong = [not r["local_ok"] for r in rows]
    caught = sum(h and w for h, w in zip(handed, wrong))
    prec = caught / sum(handed) if any(handed) else 1.0
    rec = caught / sum(wrong) if any(wrong) else 1.0
    return dict(acc=acc, prec=prec, rec=rec, handoff=sum(handed) / len(rows))


def main():
    rules = {t: json.loads((RESULTS / t / "router_config.json").read_text()) for t in TYPES}
    guesses = json.loads((RESULTS / "classifier_test.json").read_text())

    rows = []
    for t in TYPES:
        local, cloud, sigs = load(t, "local"), load(t, "cloud"), load(t, "signals")
        for k in local:
            if k in cloud and k in sigs and is_test(k):
                g = guesses[k]
                rows.append(dict(task=t, local_ok=local[k]["correct"], cloud_ok=cloud[k]["correct"],
                                 sig=sigs[k],
                                 guess=g["pred"] if g["conf"] >= MIN_CONFIDENCE else FALLBACK))

    versions = {
        "A  math rule for all":    lambda r: rules["math"],
        "B  own rule, true type":  lambda r: rules[r["task"]],
        "C  own rule, guessed":    lambda r: rules[r["guess"]],
    }

    print("rules in use (cap tokens / worst log-prob):")
    for t in TYPES:
        lp = rules[t]["min_logprob"]
        print(f"  {t:7s} {rules[t]['cap_tokens']:5d} / {'none' if lp is None else f'{lp:.2f}'}")
    wrong_type = sum(r["guess"] != r["task"] for r in rows)
    print(f"\nclassifier picked the wrong rule for {wrong_type} of {len(rows)} test questions\n")

    for scope in ["ALL"] + TYPES:
        part = rows if scope == "ALL" else [r for r in rows if r["task"] == scope]
        local = sum(r["local_ok"] for r in part) / len(part)
        cloud = sum(r["cloud_ok"] for r in part) / len(part)
        print(f"--- {scope} ({len(part)} test questions) | local {local:.1%} | cloud {cloud:.1%}")
        print(f"  {'version':24s} {'accuracy':>8s} {'precision':>9s} {'recall':>7s} {'handed off':>10s}")
        for name, rule_for in versions.items():
            s = score(part, rule_for)
            print(f"  {name:24s} {s['acc']:8.1%} {s['prec']:9.0%} {s['rec']:7.0%} {s['handoff']:10.0%}")
        print()


if __name__ == "__main__":
    main()
