"""Stage 3, step 1: run the 4B model on NEW questions and label every answer right or wrong.

These labels are what the verifier learns from: "given a question and my own answer, was I right?"

Usage (inside .venv-gpu, from the repo folder):
    python train/gen_data.py --task trivia --n 1000
    python train/gen_data.py --task code   --n 500
    python train/gen_data.py --task mmlu   --n 1000
    python train/gen_data.py --task math   --n 1000

Output: data/<task>.jsonl, one line per question. Safe to stop with Ctrl+C and run again:
it skips questions that are already done.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "router"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                    # noqa: E402
from unsloth import FastLanguageModel           # noqa: E402

from sources import SOURCES                     # noqa: E402

MODEL = "unsloth/Qwen3.5-4B"      # same model as on the Mac (there: MLX 4-bit; here: bitsandbytes 4-bit)
MAX_NEW = 1024                    # answers longer than this almost always fail anyway


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=SOURCES)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=MAX_NEW, help="longest answer, in tokens")
    args = ap.parse_args()

    out_path = ROOT / "data" / f"{args.task}.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out_path)} if out_path.exists() else set()
    items = [it for it in SOURCES[args.task](args.n) if it["id"] not in done]
    print(f"{args.task}: {len(done)} already done, {len(items)} to go")
    if not items:
        return

    model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length=2048, load_in_4bit=True)
    FastLanguageModel.for_inference(model)
    tok = getattr(tok, "tokenizer", tok)                  # some models return a processor
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    prompts = [tok.apply_chat_template(it["messages"], add_generation_prompt=True, tokenize=False,
                                       enable_thinking=False) for it in items]
    order = sorted(range(len(items)), key=lambda i: len(prompts[i]))   # similar lengths batch well
    t0, n_done, n_right = time.time(), 0, 0
    with open(out_path, "a") as f:
        for s in range(0, len(order), args.batch):
            idx = order[s:s + args.batch]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for i, row in zip(idx, new):
                n_tok = int((row != tok.pad_token_id).sum())
                text = tok.decode(row, skip_special_tokens=True)
                it = items[i]
                ok = bool(it["check"](text))
                f.write(json.dumps(dict(id=it["id"], task=it["task"], messages=it["messages"],
                                        output=text, tokens=n_tok, correct=ok)) + "\n")
                n_done += 1
                n_right += ok
            f.flush()
            torch.cuda.empty_cache()                      # hand unused memory back between batches
            rate = n_done / (time.time() - t0)
            print(f"  {n_done}/{len(items)}  right so far {n_right / n_done:.0%}  "
                  f"{rate * 60:.0f} questions/min  ~{(len(items) - n_done) / rate / 60:.0f} min left")


if __name__ == "__main__":
    main()
