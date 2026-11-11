"""Stage 1: how good is the local model, how good is the cloud model, on the same questions?

Usage:
    uv run python router/stage1_baseline.py local --task math   --n 500
    uv run python router/stage1_baseline.py cloud --task math   --n 500
    uv run python router/stage1_baseline.py local --task mmlu   --n 300
    uv run python router/stage1_baseline.py local --task trivia --n 300
    uv run python router/stage1_baseline.py report              # every task

Tasks are defined in router/tasks.py. Results are cached in results/<task>/*.jsonl,
one line per question. Rerunning skips questions already answered, so a run can be
stopped and resumed (or continued the next day when the free API limit resets).
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from tasks import TASKS

LOCAL_MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
CLOUD_MODEL = "openai/gpt-oss-120b"
GROQ_URL = "https://api.groq.com/openai/v1"
MAX_TOKENS = 2048
RESULTS = Path("results")


def load_done(path):
    if not path.exists():
        return {}
    with open(path) as f:
        return {r["id"]: r for r in map(json.loads, f)}


def save_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


# ---------- local model (MLX) ----------

def run_local(task, items):
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(LOCAL_MODEL)
    sampler = make_sampler(temp=0.0)            # greedy: repeatable results
    path = RESULTS / task / "local.jsonl"
    done = load_done(path)

    for i, item in enumerate(items):
        if item["id"] in done:
            continue
        messages = TASKS[task]["messages"](item)
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                               tokenize=False, enable_thinking=False)
        t0 = time.time()
        text = generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, sampler=sampler)
        secs = time.time() - t0
        ok = TASKS[task]["check"](text, item)
        save_row(path, dict(id=item["id"], group=item["group"], correct=ok, secs=round(secs, 2),
                            tokens=len(tokenizer.encode(text)), messages=messages, output=text))
        print(f"[{i + 1}/{len(items)}] {item['group']} | {'OK ' if ok else 'BAD'} | {secs:.1f}s")


# ---------- cloud model (Groq) ----------

def run_cloud(task, items):
    from openai import OpenAI, RateLimitError

    client = OpenAI(base_url=GROQ_URL, api_key=os.environ["GROQ_API_KEY"])
    path = RESULTS / task / "cloud.jsonl"
    done = load_done(path)

    for i, item in enumerate(items):
        if item["id"] in done:
            continue
        messages = TASKS[task]["messages"](item)
        while True:
            try:
                t0 = time.time()
                resp = client.chat.completions.create(
                    model=CLOUD_MODEL, messages=messages, temperature=0, max_tokens=MAX_TOKENS,
                    extra_body={"reasoning_effort": "low"})   # fewer tokens: free tier is ~200K/day
                break
            except RateLimitError:
                print("rate limited, waiting 60s ...")
                time.sleep(60)
        secs = time.time() - t0
        text = resp.choices[0].message.content or ""
        u = resp.usage
        ok = TASKS[task]["check"](text, item)
        save_row(path, dict(id=item["id"], group=item["group"], correct=ok, secs=round(secs, 2),
                            prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
                            output=text))
        print(f"[{i + 1}/{len(items)}] {item['group']} | {'OK ' if ok else 'BAD'} | "
              f"{secs:.1f}s | {u.completion_tokens} tok")
        time.sleep(2.1)                          # stay under 30 requests/minute


# ---------- report ----------

def report_task(task):
    print(f"\n========== {task} ==========")
    local = load_done(RESULTS / task / "local.jsonl")
    cloud = load_done(RESULTS / task / "cloud.jsonl")
    for name, rows in [("local", local), ("cloud", cloud)]:
        rows = list(rows.values())
        if not rows:
            print(f"{name}: no results yet")
            continue
        acc = sum(r["correct"] for r in rows) / len(rows)
        avg = sum(r.get("secs", 0) for r in rows) / len(rows)
        print(f"{name}: {acc:.1%} correct on {len(rows)} questions, {avg:.1f}s avg")
        groups = defaultdict(list)
        for r in rows:
            groups[r.get("group", f"level {r.get('level')}")].append(r["correct"])
        if len(groups) > 1:
            for g in sorted(groups):
                v = groups[g]
                print(f"  {g:22s} {sum(v) / len(v):4.0%} ({len(v)})")

    both = [k for k in local if k in cloud]
    if both:
        cases = defaultdict(int)
        for k in both:
            cases[(local[k]["correct"], cloud[k]["correct"])] += 1
        print(f"on the {len(both)} questions both answered:")
        print(f"  both right              : {cases[(True, True)]}")
        print(f"  local wrong, cloud right: {cases[(False, True)]}   (should hand off)")
        print(f"  local right, cloud wrong: {cases[(True, False)]}")
        print(f"  both wrong              : {cases[(False, False)]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("who", choices=["local", "cloud", "report"])
    ap.add_argument("--task", choices=list(TASKS), default="math")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    if args.who == "report":
        for t in TASKS:
            if (RESULTS / t).exists():
                report_task(t)
    else:
        items = TASKS[args.task]["load"](args.n)
        run_local(args.task, items) if args.who == "local" else run_cloud(args.task, items)
