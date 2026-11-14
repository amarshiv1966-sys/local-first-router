"""Training questions for Stage 3: NEW questions, none of them from the evaluation sets.

    math    MATH train split (MATH-500 is drawn from the test split, so no overlap)
    mmlu    MMLU-Pro questions after the first 300 we evaluated on
    trivia  TriviaQA train split (we evaluated on validation)
    code    MBPP (we evaluated on HumanEval)

Each source returns items with: id, task, messages, and a check(text) that says if an answer is right.
"""

import hashlib

from datasets import concatenate_datasets, load_dataset

from sandbox import run_code
from tasks import (MATH_SYSTEM, TRIVIA_SYSTEM, extract_code, math_check, mmlu_check,
                   mmlu_messages, trivia_check)

CODE_SYSTEM = ("Write the Python function. Reply with the complete function, including any "
               "imports, in a single ```python code block.")


def last_boxed(s):
    """Contents of the last \\boxed{...} in a worked solution (handles nested braces)."""
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    i += len("\\boxed{")
    depth, j = 1, i
    while j < len(s) and depth:
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        j += 1
    return s[i:j - 1] if depth == 0 else None


def short_id(prefix, text):
    return f"{prefix}/{hashlib.md5(text.encode()).hexdigest()[:10]}"


def math_items(n):
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train").shuffle(seed=1)
    out = []
    for q in ds:
        gold = last_boxed(q["solution"])
        if gold:
            item = dict(id=short_id("mathtrain", q["problem"]), task="math", gold=gold,
                        messages=[{"role": "system", "content": MATH_SYSTEM},
                                  {"role": "user", "content": q["problem"]}])
            item["check"] = lambda text, it=item: math_check(text, it)
            out.append(item)
        if len(out) == n:
            break
    return out


def mmlu_items(n):
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test").shuffle(seed=0)
    ds = ds.select(range(300, min(300 + n, len(ds))))        # skip the 300 used for evaluation
    out = []
    for q in ds:
        item = dict(id=f"mmlu/{q['question_id']}", task="mmlu", gold=q["answer"],
                    question=q["question"], options=q["options"])
        item["messages"] = mmlu_messages(item)
        item["check"] = lambda text, it=item: mmlu_check(text, it)
        out.append(item)
    return out


def trivia_items(n):
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train").shuffle(seed=1)
    out = []
    for q in ds.select(range(min(n, len(ds)))):
        item = dict(id=f"trivia/{q['question_id']}", task="trivia",
                    gold=q["answer"]["normalized_aliases"],
                    messages=[{"role": "system", "content": TRIVIA_SYSTEM},
                              {"role": "user", "content": q["question"]}])
        item["check"] = lambda text, it=item: trivia_check(text, it)
        out.append(item)
    return out


def code_items(n):
    parts = load_dataset("google-research-datasets/mbpp", "full")
    ds = concatenate_datasets([parts[s] for s in parts]).shuffle(seed=1)
    out = []
    for q in ds.select(range(min(n, len(ds)))):
        prompt = f"{q['text']}\nYour code should pass this test:\n{q['test_list'][0]}"
        item = dict(id=f"mbpp/{q['task_id']}", task="code",
                    messages=[{"role": "system", "content": CODE_SYSTEM},
                              {"role": "user", "content": prompt}])

        def check(text, q=q):
            program = "\n".join([q["test_setup_code"] or "", extract_code(text), *q["test_list"]])
            return run_code(program, timeout=10)["passed"]
        item["check"] = check
        out.append(item)
    return out


SOURCES = {"math": math_items, "mmlu": mmlu_items, "trivia": trivia_items, "code": code_items}
