"""Stage 3, step 2: fine-tune the verifier with LoRA on the data from gen_data.py.

The training signal is a 2-way choice at one token: "Yes" (my answer was right) vs "No".
Loss = cross-entropy between those two logits and the true label, weighted so right and
wrong answers count equally. Only the LoRA weights (about 1% of the model) are trained.

Usage (inside .venv-gpu, from the repo folder):
    python train/train_verifier.py              # 1 epoch
    python train/train_verifier.py --epochs 2

Output: outputs/verifier_lora/   (the LoRA weights, a few tens of MB)
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                    # noqa: E402
from unsloth import FastLanguageModel           # noqa: E402

from verifier_common import MODEL, auroc, verifier_prompt, yes_no_ids, yes_no_logits   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=16)        # examples per weight update
    args = ap.parse_args()

    rows = [json.loads(l) for f in sorted((ROOT / "data").glob("*.jsonl")) for l in open(f)]
    random.Random(0).shuffle(rows)
    n_val = len(rows) // 10
    val, train = rows[:n_val], rows[n_val:]
    right = sum(r["correct"] for r in train)
    print(f"{len(train)} training examples ({right / len(train):.0%} right), {len(val)} for validation")
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r["correct"])
    for t, c in by_task.items():
        print(f"  {t:7s} {len(c):5d} examples, {sum(c) / len(c):.0%} right")

    model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length=3072, load_in_4bit=True)
    tok = getattr(tok, "tokenizer", tok)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=0)
    yes_id, no_id = yes_no_ids(tok)

    # weight the rarer label up, so the model can't win by always saying the common one
    w_right, w_wrong = len(train) / (2 * right), len(train) / (2 * (len(train) - right))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = args.epochs * len(train) // args.accum
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / total))

    def evaluate():
        model.eval()
        scores = []
        with torch.no_grad():
            for r in val:
                lg = yes_no_logits(model, tok, verifier_prompt(tok, r["messages"], r["output"]), yes_id, no_id)
                scores.append(torch.softmax(lg, 0)[0].item())
        model.train()
        return auroc(scores, [r["correct"] for r in val])

    print(f"validation AUROC before training: {evaluate():.3f}")
    model.train()
    step, t0, run_loss = 0, time.time(), 0.0
    for epoch in range(args.epochs):
        random.Random(epoch).shuffle(train)
        for i, r in enumerate(train):
            lg = yes_no_logits(model, tok, verifier_prompt(tok, r["messages"], r["output"]), yes_id, no_id)
            target = torch.tensor(0 if r["correct"] else 1, device=lg.device)
            loss = torch.nn.functional.cross_entropy(lg[None], target[None])
            loss = loss * (w_right if r["correct"] else w_wrong) / args.accum
            loss.backward()
            run_loss += loss.item()
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                if step % 10 == 0:
                    left = (total - step) * (time.time() - t0) / step / 60
                    print(f"  step {step}/{total}  loss {run_loss / 10:.3f}  ~{left:.0f} min left")
                    run_loss = 0.0
        print(f"epoch {epoch + 1}: validation AUROC {evaluate():.3f}")

    out = ROOT / "outputs" / "verifier_lora"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
