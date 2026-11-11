"""Stage 2: can the local model tell when its own answer is probably wrong?

Usage (run Stage 1 first, for both local and cloud):
    uv run python router/stage2_router.py score     # reads token probabilities, ~1-3 min
    uv run python router/stage2_router.py report    # tables + results/router_curve.png

`score` does NOT generate answers again. It feeds each saved answer back into the
model in ONE forward pass and reads how probable the model found each token it wrote.
That is fast, because nothing is sampled one token at a time.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset

LOCAL_MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
RESULTS = Path("results")
SYSTEM = "Solve the math problem. Think step by step, then put only the final answer in \\boxed{}."
MAX_TOKENS = 2048
TOLERANCE = 0.01   # "as good as cloud" = within 1 percentage point of cloud-only accuracy


def load_rows(name):
    path = RESULTS / f"{name}.jsonl"
    with open(path) as f:
        return {r["id"]: r for r in map(json.loads, f)}


# ---------- score: token log-probabilities of each saved answer ----------

def score():
    import mlx.core as mx
    from mlx_lm import load

    model, tokenizer = load(LOCAL_MODEL)
    local = load_rows("local")
    problems = {q["unique_id"]: q["problem"]
                for q in load_dataset("HuggingFaceH4/MATH-500", split="test")}

    out_path = RESULTS / "signals.jsonl"
    done = set(load_rows("signals")) if out_path.exists() else set()

    for i, (qid, row) in enumerate(local.items()):
        if qid in done:
            continue
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": problems[qid]}]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        prompt_ids = tokenizer.encode(prompt)
        answer_ids = tokenizer.encode(row["output"], add_special_tokens=False)
        ids = prompt_ids + answer_ids

        x = mx.array(ids)[None]                          # (1, T)
        logits = model(x)[0, :-1]                        # position t predicts token t+1
        targets = mx.array(ids[1:])
        lse = mx.logsumexp(logits, axis=-1)              # log of softmax denominator
        picked = mx.take_along_axis(logits, targets[:, None], axis=-1).squeeze(-1)
        logprobs = (picked - lse)[len(prompt_ids) - 1:]  # keep only the answer's tokens
        lp = logprobs.astype(mx.float32).tolist()

        tail = lp[-50:]                                  # the end, where the answer is stated
        sig = dict(
            id=qid,
            mean_logprob=sum(lp) / len(lp),
            min_logprob=min(lp),
            tail_logprob=sum(tail) / len(tail),
            n_tokens=len(lp),
        )
        with open(out_path, "a") as f:
            f.write(json.dumps(sig) + "\n")
        print(f"[{i + 1}/{len(local)}] {len(lp)} tokens | mean logprob {sig['mean_logprob']:.3f}")


# ---------- report: how good is each signal as a router? ----------

# each signal turned into "confidence": higher = more likely the local answer is right
SIGNALS = {
    "mean token prob":   lambda s: s["mean_logprob"],
    "worst token prob":  lambda s: s["min_logprob"],
    "end-of-answer prob": lambda s: s["tail_logprob"],
    "shortness":         lambda s: -s["n_tokens"],
}


def auroc(conf, correct):
    """Chance that a random right answer gets higher confidence than a random wrong one.
    0.5 = no better than a coin flip, 1.0 = perfectly separates right from wrong."""
    pos = [c for c, ok in zip(conf, correct) if ok]
    neg = [c for c, ok in zip(conf, correct) if not ok]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def curve(order, local_ok, cloud_ok):
    """Hand off the k least-confident questions, for every k. Returns accuracy per k."""
    n = len(order)
    accs = []
    for k in range(n + 1):
        handed = set(order[:k])
        right = sum(cloud_ok[i] if i in handed else local_ok[i] for i in range(n))
        accs.append(right / n)
    return accs


def report():
    local, cloud, sigs = load_rows("local"), load_rows("cloud"), load_rows("signals")
    ids = [k for k in local if k in cloud and k in sigs]
    n = len(ids)
    local_ok = [local[k]["correct"] for k in ids]
    cloud_ok = [cloud[k]["correct"] for k in ids]
    capped = sum(sigs[k]["n_tokens"] >= MAX_TOKENS - 5 for k in ids)

    print(f"{n} questions | local {sum(local_ok) / n:.0%} | cloud {sum(cloud_ok) / n:.0%} | "
          f"{capped} local answers hit the {MAX_TOKENS}-token limit\n")

    curves = {}
    oracle = sorted(range(n), key=lambda i: local_ok[i])            # wrong answers first
    curves["perfect router"] = curve(oracle, local_ok, cloud_ok)
    rnd = list(range(n))
    random.Random(0).shuffle(rnd)
    curves["random"] = curve(rnd, local_ok, cloud_ok)

    print(f"{'signal':20s} {'AUROC':>6s}")
    for name, fn in SIGNALS.items():
        conf = [fn(sigs[k]) for k in ids]
        print(f"{name:20s} {auroc(conf, local_ok):6.3f}")
        order = sorted(range(n), key=lambda i: conf[i])            # least confident first
        curves[name] = curve(order, local_ok, cloud_ok)

    steps = [0, 10, 20, 30, 50, 100]
    print(f"\naccuracy when handing off X% of questions to the cloud")
    print(f"{'router':20s}" + "".join(f"{s:>6d}%" for s in steps))
    for name, accs in curves.items():
        print(f"{name:20s}" + "".join(f"{accs[round(s * n / 100)]:7.0%}" for s in steps))

    target = sum(cloud_ok) / n - TOLERANCE
    print(f"\nhandoffs needed to get within 1 point of cloud-only accuracy (>= {target:.1%}):")
    for name, accs in curves.items():
        k = next((k for k, a in enumerate(accs) if a >= target), None)
        print(f"  {name:20s} {'never' if k is None else f'{k / n:.0%} of questions'}")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 5))
    xs = [100 * k / n for k in range(n + 1)]
    for name, accs in curves.items():
        style = "--" if name in ("perfect router", "random") else "-"
        plt.plot(xs, [100 * a for a in accs], style, label=name)
    plt.axhline(100 * (target + TOLERANCE), color="gray", linewidth=0.8)
    plt.xlabel("% of questions handed off to the cloud (≈ cost)")
    plt.ylabel("accuracy (%)")
    plt.title("Local-first routing on MATH-500")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(RESULTS / "router_curve.png", dpi=150, bbox_inches="tight")
    print(f"\nsaved {RESULTS / 'router_curve.png'}")


# ---------- honest test: pick the threshold on one half, measure on the other ----------

def is_test(qid):
    """Fixed 50/50 split by question id: the same question always lands on the same side."""
    return int(hashlib.md5(qid.encode()).hexdigest(), 16) % 2 == 1


def pick_threshold(conf, local_ok, cloud_ok):
    """On the TRAIN half: the lowest handoff rate that gets within 1 point of cloud accuracy.
    Returns the confidence threshold: hand off anything below it."""
    n = len(conf)
    target = sum(cloud_ok) / n - TOLERANCE
    order = sorted(range(n), key=lambda i: conf[i])
    accs = curve(order, local_ok, cloud_ok)
    k = next((k for k, a in enumerate(accs) if a >= target), n)
    if k == 0:
        return float("-inf")
    if k == n:
        return float("inf")
    return (conf[order[k - 1]] + conf[order[k]]) / 2      # halfway between the two neighbours


def split_report():
    local, cloud, sigs = load_rows("local"), load_rows("cloud"), load_rows("signals")
    ids = [k for k in local if k in cloud and k in sigs]
    train = [k for k in ids if not is_test(k)]
    test = [k for k in ids if is_test(k)]

    def accuracy(qids, handed):
        return sum(cloud[k]["correct"] if k in handed else local[k]["correct"] for k in qids) / len(qids)

    cloud_test = sum(cloud[k]["correct"] for k in test) / len(test)
    local_test = sum(local[k]["correct"] for k in test) / len(test)
    print(f"\n=== honest test: threshold picked on {len(train)} questions, "
          f"measured on {len(test)} unseen ones ===")
    print(f"test set: local only {local_test:.1%} | cloud only {cloud_test:.1%}\n")
    print(f"{'signal':20s} {'handed off':>11s} {'accuracy':>9s} {'vs cloud':>9s}")

    for name, fn in SIGNALS.items():
        thr = pick_threshold([fn(sigs[k]) for k in train],
                             [local[k]["correct"] for k in train],
                             [cloud[k]["correct"] for k in train])
        handed = {k for k in test if fn(sigs[k]) < thr}
        acc = accuracy(test, handed)
        print(f"{name:20s} {len(handed) / len(test):11.0%} {acc:9.1%} {acc - cloud_test:+9.1%}")

    # early stop: the length rule only needs to see the FIRST `cap` tokens, so stop there
    thr = pick_threshold([SIGNALS["shortness"](sigs[k]) for k in train],
                         [local[k]["correct"] for k in train],
                         [cloud[k]["correct"] for k in train])
    cap = int(-thr) if thr not in (float("inf"), float("-inf")) else MAX_TOKENS
    full = sum(local[k]["secs"] for k in test)
    early = sum(local[k]["secs"] * min(1.0, cap / max(1, sigs[k]["n_tokens"])) for k in test)
    print(f"\nearly stop: hand off any local answer that passes {cap} tokens")
    print(f"  local time on the test set: {full / 60:.0f} min without early stop, "
          f"~{early / 60:.0f} min with it ({1 - early / full:.0%} saved, estimated)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["score", "report"])
    args = ap.parse_args()
    if args.step == "score":
        score()
    else:
        report()
        split_report()
