"""Shared pieces for the Stage 3 verifier: the prompt it sees and how its score is read.

The verifier is the same 4B model plus a small LoRA add-on. It is shown a question, its
own earlier answer, and then asked: "Is it correct? Yes or No". Its score is
    P(correct) = P("Yes") / (P("Yes") + P("No"))   at the next token.
"""

import torch

MODEL = "unsloth/Qwen3.5-4B"
QUESTION = "Look at your answer above. Is it correct? Reply with only Yes or No."
MAX_ANSWER_TOKENS = 1024        # longer answers are cut, keeping the END (where the final answer is)


def verifier_prompt(tok, messages, output):
    ids = tok(output, add_special_tokens=False)["input_ids"]
    if len(ids) > MAX_ANSWER_TOKENS:
        output = "...\n" + tok.decode(ids[-MAX_ANSWER_TOKENS:])
    msgs = [*messages, {"role": "assistant", "content": output},
            {"role": "user", "content": QUESTION}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                   enable_thinking=False)


def yes_no_ids(tok):
    return (tok("Yes", add_special_tokens=False)["input_ids"][0],
            tok("No", add_special_tokens=False)["input_ids"][0])


def yes_no_logits(model, tok, prompt, yes_id, no_id):
    """The two logits [Yes, No] at the next token (for training and scoring)."""
    enc = tok(prompt, return_tensors="pt").to("cuda")
    try:
        logits = model(**enc, logits_to_keep=1).logits[0, -1]       # saves memory
    except TypeError:
        logits = model(**enc).logits[0, -1]
    return torch.stack([logits[yes_id], logits[no_id]]).float()


def auroc(scores, labels):
    """Chance that a random correct answer scores higher than a random wrong one."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))
