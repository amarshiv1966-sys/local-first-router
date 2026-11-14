"""Stage 4: teach the model to say HANDOFF by itself, with reinforcement learning (GRPO).

No separate judge: the model either answers or replies "HANDOFF", in one pass.
Reward per sampled reply:
    correct answer  +1
    wrong answer    -1
    HANDOFF         +C   (default 0.3)
So handing off pays only when the model's chance of being right is below (1 + C) / 2 = 65%.
The model has to learn, from its own trial and error, which questions it gets wrong.

GRPO in one line: for each question, sample several replies, score them, and push the model
toward the replies that scored above the group's average (no value network needed).

Started on facts (TriviaQA): short answers keep RL affordable on an 8 GB GPU, and facts are
where the Stage 3 verifier was weakest (AUROC 0.83).

Warm start: RL can only reinforce what the model already does sometimes. Out of the box it
never says HANDOFF (0% in a test run), so GRPO had nothing to push toward. So we first show it
a few hundred examples (supervised): its own wrong Stage 3 answers become "HANDOFF", its right
ones stay as they were. Then RL refines WHEN to hand off. The warm-start model is saved on its
own too, so the evaluation can show what RL added on top of it.

Usage (inside .venv-gpu, from the repo folder):
    python train/rl_handoff.py --warmup 500 --steps 200
    python train/rl_handoff.py --warmup 500 --steps 0     # warm start only
Output: outputs/rl_warmstart_lora/, outputs/rl_handoff_lora/
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "router"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from unsloth import FastLanguageModel           # noqa: E402  (must come before trl)
from datasets import Dataset                    # noqa: E402
from trl import GRPOConfig, GRPOTrainer         # noqa: E402

from sources import trivia_items                # noqa: E402
from tasks import TRIVIA_SYSTEM, trivia_check   # noqa: E402
from verifier_common import MODEL               # noqa: E402

RL_SYSTEM = (TRIVIA_SYSTEM + " If you are not confident that your answer is correct, "
             "reply with only the word HANDOFF instead, and a stronger model will answer.")


def is_handoff(text):
    return "HANDOFF" in text and "answer:" not in text.lower()


def warm_start(model, tok, n):
    import torch
    rows = [json.loads(l) for l in open(ROOT / "data" / "trivia.jsonl")]
    random.Random(0).shuffle(rows)
    rows = rows[:n]
    print(f"warm start on {len(rows)} examples: {sum(not r['correct'] for r in rows)} wrong -> HANDOFF")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)
    model.train()
    accum, run = 8, 0.0
    for i, r in enumerate(rows):
        prompt = tok.apply_chat_template([{"role": "system", "content": RL_SYSTEM}, r["messages"][-1]],
                                         add_generation_prompt=True, tokenize=False, enable_thinking=False)
        target = (r["output"] if r["correct"] else "HANDOFF") + tok.eos_token
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        t_ids = tok(target, add_special_tokens=False)["input_ids"][:64]
        ids = torch.tensor([p_ids + t_ids], device="cuda")
        labels = torch.tensor([[-100] * len(p_ids) + t_ids], device="cuda")   # learn the reply only
        loss = model(input_ids=ids, labels=labels).loss / accum
        loss.backward()
        run += loss.item()
        if (i + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad()
            if (i + 1) % (accum * 10) == 0:
                print(f"  warm start {i + 1}/{len(rows)}  loss {run / 10:.3f}")
                run = 0.0
    out = ROOT / "outputs" / "rl_warmstart_lora"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--c", type=float, default=0.3, help="reward for HANDOFF")
    ap.add_argument("--gens", type=int, default=8, help="replies sampled per question")
    ap.add_argument("--warmup", type=int, default=500, help="supervised warm-start examples (0 = none)")
    args = ap.parse_args()

    model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length=1024, load_in_4bit=True)
    tok = getattr(tok, "tokenizer", tok)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=0)

    if args.warmup:
        warm_start(model, tok, args.warmup)
        if not args.steps:
            return

    items = trivia_items(1000)          # the same TriviaQA *train* questions as Stage 3
    prompts = [tok.apply_chat_template([{"role": "system", "content": RL_SYSTEM},
                                        {"role": "user", "content": it["messages"][-1]["content"]}],
                                       add_generation_prompt=True, tokenize=False, enable_thinking=False)
               for it in items]
    ds = Dataset.from_dict({"prompt": prompts, "gold": [it["gold"] for it in items]})

    stats = {"n": 0, "handoff": 0, "right": 0}

    def reward(completions, gold, **_):
        out = []
        for text, g in zip(completions, gold):
            text = text if isinstance(text, str) else text[0]["content"]
            if is_handoff(text):
                r = args.c
                stats["handoff"] += 1
            elif trivia_check(text, {"gold": g}):
                r = 1.0
                stats["right"] += 1
            else:
                r = -1.0
            stats["n"] += 1
            out.append(r)
        if stats["n"] >= 400:                      # a running summary every ~50 questions
            n = stats["n"]
            print(f"    last {n} replies: {stats['handoff'] / n:.0%} HANDOFF, "
                  f"{stats['right'] / n:.0%} right, {(n - stats['handoff'] - stats['right']) / n:.0%} wrong")
            stats.update(n=0, handoff=0, right=0)
        return out

    cfg = GRPOConfig(
        output_dir=str(ROOT / "outputs" / "rl_run"),
        learning_rate=5e-6, beta=0.04, temperature=0.8,
        num_generations=args.gens, per_device_train_batch_size=args.gens,
        gradient_accumulation_steps=4,                  # 4 questions per update
        max_prompt_length=256, max_completion_length=64,
        max_steps=args.steps, logging_steps=5, save_steps=50, report_to="none",
        lr_scheduler_type="cosine", warmup_ratio=0.05,
    )
    trainer = GRPOTrainer(model=model, processing_class=tok, reward_funcs=[reward],
                          args=cfg, train_dataset=ds)
    trainer.train()
    out = ROOT / "outputs" / "rl_handoff_lora"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
