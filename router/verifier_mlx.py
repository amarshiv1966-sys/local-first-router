"""The Stage 3 verifier on the Mac: the 4060-trained LoRA, loaded into the MLX model.

The LoRA was trained with PEFT/Unsloth (PyTorch). Each adapted layer has two small matrices:
    PEFT:  lora_A (r x in),  lora_B (out x r),   output += (alpha/r) * B @ A @ x
    MLX:   lora_a (in x r),  lora_b (r x out),   output += scale * x @ lora_a @ lora_b
So converting = transposing both and wrapping the same layers in mlx_lm's LoRALinear.

The answering model and the verifier share ONE copy of the 4B weights: the LoRA is switched
off while answering (scale 0) and on while judging. No second model in memory.

Usage:
    uv run python router/verifier_mlx.py     # checks the Mac scores match the 4060's
"""

import json
import re
from pathlib import Path

import mlx.core as mx
from mlx_lm.tuner.lora import LoRALinear

ROOT = Path(__file__).resolve().parent.parent
LORA_DIR = ROOT / "outputs" / "verifier_lora"
QUESTION = "Look at your answer above. Is it correct? Reply with only Yes or No."
MAX_ANSWER_TOKENS = 1024
KEY = re.compile(r"layers\.(\d+)\.(self_attn|mlp|linear_attn)\.(\w+)\.lora_([AB])\.weight$")


def text_layers(model):
    lm = getattr(model, "language_model", model)
    return getattr(lm, "model", lm).layers


def add_lora(model, lora_dir=LORA_DIR):
    """Wrap every layer the LoRA trained with mlx_lm's LoRALinear. Returns the wrapped layers."""
    cfg = json.loads((lora_dir / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    pairs = {}
    for k, v in mx.load(str(lora_dir / "adapter_model.safetensors")).items():   # handles bfloat16
        if "visual" in k:
            continue
        m = KEY.search(k)
        if m:
            layer, block, proj, ab = m.groups()
            pairs.setdefault((int(layer), block, proj), {})[ab] = v
    layers, wrapped = text_layers(model), []
    for (i, block, proj), w in sorted(pairs.items()):
        parent = getattr(layers[i], block)
        lora = LoRALinear.from_base(getattr(parent, proj), r=cfg["r"], scale=scale, dropout=0.0)
        lora.lora_a = w["A"].T.astype(mx.float16)
        lora.lora_b = w["B"].T.astype(mx.float16)
        setattr(parent, proj, lora)
        wrapped.append(lora)
    mx.eval([l.parameters() for l in wrapped])
    print(f"verifier LoRA: {len(wrapped)} layers adapted (scale {scale})")
    return wrapped


def set_lora(wrapped, on):
    for l in wrapped:
        l.scale = l._on_scale if on else 0.0


def remember_scale(wrapped):
    for l in wrapped:
        l._on_scale = l.scale


def verifier_prompt(tok, messages, output):
    ids = tok.encode(output, add_special_tokens=False)
    if len(ids) > MAX_ANSWER_TOKENS:
        output = "...\n" + tok.decode(ids[-MAX_ANSWER_TOKENS:])
    msgs = [*messages, {"role": "assistant", "content": output}, {"role": "user", "content": QUESTION}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)


def p_correct(model, tok, messages, output):
    """P("Yes") / (P("Yes") + P("No")) for 'is your answer above correct?'"""
    yes = tok.encode("Yes", add_special_tokens=False)[0]
    no = tok.encode("No", add_special_tokens=False)[0]
    ids = tok.encode(verifier_prompt(tok, messages, output), add_special_tokens=False)
    logits = model(mx.array(ids)[None])[0, -1]
    two = mx.softmax(mx.stack([logits[yes], logits[no]]).astype(mx.float32))
    return two[0].item()


def main():
    """Score 10 saved answers per type on the Mac and compare with the 4060's scores."""
    import random

    from mlx_lm import load
    model, tok = load("mlx-community/Qwen3.5-4B-MLX-4bit")
    wrapped = add_lora(model)
    remember_scale(wrapped)
    diffs = []
    for task in ["math", "mmlu", "trivia", "code"]:
        local = {r["id"]: r for r in map(json.loads, open(ROOT / "results" / task / "local.jsonl"))}
        gpu = {r["id"]: r["p_correct"] for r in map(json.loads, open(ROOT / "results" / task / "verifier.jsonl"))}
        ids = [k for k in gpu if "messages" in local.get(k, {})]
        random.Random(0).shuffle(ids)
        print(f"--- {task}")
        for k in ids[:10]:
            r = local[k]
            p = p_correct(model, tok, r["messages"], r["output"])
            diffs.append(abs(p - gpu[k]))
            print(f"  Mac {p:.3f}   4060 {gpu[k]:.3f}   truly {'right' if r['correct'] else 'wrong'}")
    print(f"\nmean difference Mac vs 4060: {sum(diffs) / len(diffs):.3f}  (small = conversion is right)")


if __name__ == "__main__":
    main()
