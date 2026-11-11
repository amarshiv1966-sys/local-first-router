"""Stage 1: how good is the local model, how good is the cloud model, on the same questions?

Usage:
    uv run python router/stage1_baseline.py local --n 100
    uv run python router/stage1_baseline.py cloud --n 100
    uv run python router/stage1_baseline.py report

Results are cached in results/*.jsonl, one line per question. Rerunning skips
questions already answered, so a run can be stopped and resumed (or continued
the next day when the free API limit resets).
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

LOCAL_MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
CLOUD_MODEL = "openai/gpt-oss-120b"
GROQ_URL = "https://api.groq.com/openai/v1"
MAX_TOKENS = 2048
RESULTS = Path("results")

SYSTEM = "Solve the math problem. Think step by step, then put only the final answer in \\boxed{}."


# ---------- data + checking ----------

def load_questions(n):
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    ds = ds.shuffle(seed=0)                     # fixed seed: same questions every run
    return list(ds.select(range(min(n, len(ds)))))


PRED_CONFIG = [LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()]


def is_correct(model_text, gold_answer):
    try:
        gold = parse(f"${gold_answer}$")
        pred = parse(model_text, extraction_config=PRED_CONFIG)
        return bool(verify(gold, pred))          # order matters: gold first
    except Exception:
        return False


def load_done(path):
    if not path.exists():
        return {}
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    return {r["id"]: r for r in rows}


def save_row(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


# ---------- local model (MLX) ----------

def run_local(questions):
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(LOCAL_MODEL)
    sampler = make_sampler(temp=0.0)            # greedy: repeatable results
    path = RESULTS / "local.jsonl"
    done = load_done(path)

    for i, q in enumerate(questions):
        if q["unique_id"] in done:
            continue
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": q["problem"]}]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False)               # ignored if the template has no thinking mode
        t0 = time.time()
        text = generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, sampler=sampler)
        secs = time.time() - t0
        row = dict(id=q["unique_id"], level=q["level"], subject=q["subject"],
                   correct=is_correct(text, q["answer"]), secs=round(secs, 2),
                   tokens=len(tokenizer.encode(text)), output=text)
        save_row(path, row)
        print(f"[{i + 1}/{len(questions)}] level {q['level']} | "
              f"{'OK ' if row['correct'] else 'BAD'} | {secs:.1f}s")


# ---------- cloud model (Groq) ----------

def run_cloud(questions):
    from openai import OpenAI, RateLimitError

    client = OpenAI(base_url=GROQ_URL, api_key=os.environ["GROQ_API_KEY"])
    path = RESULTS / "cloud.jsonl"
    done = load_done(path)

    for i, q in enumerate(questions):
        if q["unique_id"] in done:
            continue
        while True:
            try:
                t0 = time.time()
                resp = client.chat.completions.create(
                    model=CLOUD_MODEL,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": q["problem"]}],
                    temperature=0, max_tokens=MAX_TOKENS,
                    extra_body={"reasoning_effort": "low"},   # fewer tokens: free tier is ~200K/day
                )
                break
            except RateLimitError:
                print("rate limited, waiting 60s ...")
                time.sleep(60)
        secs = time.time() - t0
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        row = dict(id=q["unique_id"], level=q["level"], subject=q["subject"],
                   correct=is_correct(text, q["answer"]), secs=round(secs, 2),
                   prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
                   output=text)
        save_row(path, row)
        print(f"[{i + 1}/{len(questions)}] level {q['level']} | "
              f"{'OK ' if row['correct'] else 'BAD'} | {secs:.1f}s | {usage.completion_tokens} tok")
        time.sleep(2.1)                          # stay under 30 requests/minute


# ---------- report ----------

def report():
    for name in ["local", "cloud"]:
        rows = list(load_done(RESULTS / f"{name}.jsonl").values())
        if not rows:
            print(f"{name}: no results yet")
            continue
        acc = sum(r["correct"] for r in rows) / len(rows)
        avg_secs = sum(r["secs"] for r in rows) / len(rows)
        print(f"\n{name}: {acc:.1%} correct on {len(rows)} questions, {avg_secs:.1f}s avg")
        by_level = defaultdict(list)
        for r in rows:
            by_level[r["level"]].append(r["correct"])
        for lvl in sorted(by_level):
            v = by_level[lvl]
            print(f"  level {lvl}: {sum(v) / len(v):.0%} ({len(v)} questions)")

    # the question the whole project is about: where does local fail but cloud succeed?
    local = load_done(RESULTS / "local.jsonl")
    cloud = load_done(RESULTS / "cloud.jsonl")
    both = [k for k in local if k in cloud]
    if both:
        cases = defaultdict(int)
        for k in both:
            cases[(local[k]["correct"], cloud[k]["correct"])] += 1
        print(f"\non the {len(both)} questions both answered:")
        print(f"  both right           : {cases[(True, True)]}   (keep local, save money)")
        print(f"  local wrong, cloud right: {cases[(False, True)]}   (should hand off)")
        print(f"  local right, cloud wrong: {cases[(True, False)]}")
        print(f"  both wrong           : {cases[(False, False)]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("who", choices=["local", "cloud", "report"])
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.who == "report":
        report()
    else:
        qs = load_questions(args.n)
        run_local(qs) if args.who == "local" else run_cloud(qs)
