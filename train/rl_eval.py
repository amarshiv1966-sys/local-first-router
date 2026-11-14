"""Stage 4 evaluation: the RL model on the held-out TriviaQA questions from Stages 1-3.

For each test question the RL model (greedy) either answers or says HANDOFF.
    handed off -> the cloud model's saved answer counts
    answered   -> its own answer is checked
Compared with the Stage 3 fine-tuned verifier at the SAME handoff rate: which gets more right?

Usage (inside .venv-gpu, from the repo folder):
    python train/rl_eval.py                                  # the RL model
    python train/rl_eval.py --lora outputs/rl_warmstart_lora # the warm start alone
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "router"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                    # noqa: E402
from unsloth import FastLanguageModel           # noqa: E402

from rl_handoff import RL_SYSTEM, is_handoff    # noqa: E402
from stage2_router import is_test               # noqa: E402
from tasks import trivia_check, trivia_load     # noqa: E402

R = ROOT / "results" / "trivia"


def load(name):
    return {r["id"]: r for r in map(json.loads, open(R / f"{name}.jsonl"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default="outputs/rl_handoff_lora")
    args = ap.parse_args()
    local, cloud, ver = load("local"), load("cloud"), load("verifier")
    # TriviaQA lists some questions several times (once per evidence document): keep each id once
    seen, items = set(), []
    for it in trivia_load(10**6):
        k = it["id"]
        if k in local and k in cloud and is_test(k) and k not in seen:
            seen.add(k)
            items.append(it)

    model, tok = FastLanguageModel.from_pretrained(str(ROOT / args.lora),
                                                   max_seq_length=1024, load_in_4bit=True)
    tok = getattr(tok, "tokenizer", tok)
    FastLanguageModel.for_inference(model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = []
    out_path = R / f"{Path(args.lora).name}.jsonl"
    with open(out_path, "w") as f:
        for s in range(0, len(items), 16):
            batch = items[s:s + 16]
            prompts = [tok.apply_chat_template([{"role": "system", "content": RL_SYSTEM},
                                                {"role": "user", "content": it["question"]}],
                                               add_generation_prompt=True, tokenize=False, enable_thinking=False)
                       for it in batch]
            enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=64, do_sample=False, pad_token_id=tok.pad_token_id)
            for it, row in zip(batch, gen[:, enc["input_ids"].shape[1]:]):
                text = tok.decode(row, skip_special_tokens=True)
                h = is_handoff(text)
                r = dict(id=it["id"], output=text, handoff=h, rl_ok=(not h) and trivia_check(text, it))
                rows.append(r)
                f.write(json.dumps(r) + "\n")

    n = len(rows)
    handed = sum(r["handoff"] for r in rows)
    answered = [r for r in rows if not r["handoff"]]
    final = sum(cloud[r["id"]]["correct"] if r["handoff"] else r["rl_ok"] for r in rows) / n
    base_wrong = sum(not local[r["id"]]["correct"] for r in rows if r["handoff"])
    print(f"{n} test questions")
    print(f"  RL model hands off {handed / n:.0%}; when it answers it is right "
          f"{sum(r['rl_ok'] for r in answered) / max(1, len(answered)):.0%} of the time")
    print(f"  of its handoffs, {base_wrong / max(1, handed):.0%} were questions the base model got wrong "
          f"(precision vs the Stage 1 answers)")

    # Stage 3 verifier with the same number of handoffs: hand off the lowest-scoring questions
    order = sorted(rows, key=lambda r: ver[r["id"]]["p_correct"])
    vh = {r["id"] for r in order[:handed]}
    v_final = sum(cloud[k]["correct"] if k in vh else local[k]["correct"] for k in (r["id"] for r in rows)) / n
    v_prec = sum(not local[k]["correct"] for k in vh) / max(1, handed)
    base = sum(local[r["id"]]["correct"] for r in rows) / n
    cl = sum(cloud[r["id"]]["correct"] for r in rows) / n
    print(f"\n  {'version':38s}{'accuracy':>9s}{'precision':>10s}{'handed off':>11s}")
    print(f"  {'local only':38s}{base:9.1%}{'':>10s}{0:11.0%}")
    print(f"  {'Stage 3 verifier, same handoff count':38s}{v_final:9.1%}{v_prec:10.0%}{handed / n:11.0%}")
    print(f"  {Path(args.lora).name:38s}{final:9.1%}{base_wrong / max(1, handed):10.0%}{handed / n:11.0%}")
    print(f"  {'always cloud':38s}{cl:9.1%}{'':>10s}{1:11.0%}")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
