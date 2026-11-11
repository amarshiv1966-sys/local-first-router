"""Every question type the router is tested on, in one place.

Each task knows three things:
    load(n)              -> n questions, always the same ones (fixed shuffle seed)
    messages(item)       -> the chat prompt for that question
    check(text, item)    -> was the model's answer correct?

Adding a new question type = adding one entry to TASKS at the bottom.
"""

import re
import string

from datasets import load_dataset


# ---------- math: MATH-500, answers checked with Math-Verify ----------

MATH_SYSTEM = "Solve the math problem. Think step by step, then put only the final answer in \\boxed{}."


def math_load(n):
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test").shuffle(seed=0)
    return [dict(id=q["unique_id"], group=f"level {q['level']}", question=q["problem"], gold=q["answer"])
            for q in ds.select(range(min(n, len(ds))))]


def math_messages(item):
    return [{"role": "system", "content": MATH_SYSTEM},
            {"role": "user", "content": item["question"]}]


def math_check(text, item):
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
    try:
        gold = parse(f"${item['gold']}$")
        pred = parse(text, extraction_config=[LatexExtractionConfig(boxed_match_priority=0),
                                              ExprExtractionConfig()])
        return bool(verify(gold, pred))           # order matters: gold first
    except Exception:
        return False


# ---------- knowledge + reasoning: MMLU-Pro, multiple choice with up to 10 options ----------

MMLU_SYSTEM = ("Answer the multiple-choice question. Think step by step, "
               "then finish with a final line of the form 'Answer: X' where X is the option letter.")
LETTERS = "ABCDEFGHIJ"


def mmlu_load(n):
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test").shuffle(seed=0)
    return [dict(id=f"mmlu/{q['question_id']}", group=q["category"], question=q["question"],
                 options=q["options"], gold=q["answer"])
            for q in ds.select(range(min(n, len(ds))))]


def mmlu_messages(item):
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(item["options"]))
    return [{"role": "system", "content": MMLU_SYSTEM},
            {"role": "user", "content": f"{item['question']}\n\n{opts}"}]


def mmlu_check(text, item):
    # "answer" in any case, but the option itself must be a CAPITAL letter, so "the answer is a ..." isn't read as A
    found = re.findall(r"(?i:answer)\s*(?i:is)?\s*[:\-]?\s*\(?\**([A-J])\b", text)
    if not found:
        found = re.findall(r"\\boxed\{\(?([A-J])\)?\}", text)
    return bool(found) and found[-1].upper() == item["gold"]    # the LAST stated answer counts


# ---------- facts: TriviaQA, short answers matched against a list of accepted aliases ----------

TRIVIA_SYSTEM = ("Answer the question with a short factual answer. "
                 "Finish with a final line of the form 'Answer: <your answer>'.")


def trivia_load(n):
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation").shuffle(seed=0)
    return [dict(id=f"trivia/{q['question_id']}", group="trivia", question=q["question"],
                 gold=q["answer"]["normalized_aliases"])
            for q in ds.select(range(min(n, len(ds))))]


def trivia_messages(item):
    return [{"role": "system", "content": TRIVIA_SYSTEM},
            {"role": "user", "content": item["question"]}]


def normalize(s):
    """The standard TriviaQA cleanup: lowercase, drop punctuation and a/an/the, squeeze spaces."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def trivia_check(text, item):
    found = re.findall(r"answer\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if not found:
        return False
    pred = normalize(found[-1])
    if pred in item["gold"]:                                  # exact match with an accepted alias
        return True
    # lenient: a short answer that contains an alias as whole words ("the river nile" vs "nile")
    return len(pred.split()) <= 6 and any(
        re.search(rf"\b{re.escape(a)}\b", pred) for a in item["gold"] if a)


# ---------- registry ----------

TASKS = {
    "math":   dict(load=math_load,   messages=math_messages,   check=math_check),
    "mmlu":   dict(load=mmlu_load,   messages=mmlu_messages,   check=mmlu_check),
    "trivia": dict(load=trivia_load, messages=trivia_messages, check=trivia_check),
}
