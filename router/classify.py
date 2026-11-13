"""Guess a question's type (math, mmlu, trivia, code) from its text alone.

TF-IDF + logistic regression, trained on the questions we already ran. Uses the same
50/50 split as the router (stage2_router.is_test), so the router's test questions
are never seen by the classifier during training.

Usage:
    uv run python router/classify.py     # trains, reports accuracy, saves results/classifier.pkl
"""

import json
import pickle
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from stage2_router import is_test

RESULTS = Path("results")
TYPES = ["math", "mmlu", "trivia", "code"]
MIN_CONFIDENCE = 0.6           # below this, the gateway treats the question as "general"


def question_text(row, task, older):
    """The user's question for one saved row. Early math rows didn't save their prompt."""
    if "messages" in row:
        return row["messages"][-1]["content"]
    if task not in older:
        from tasks import TASKS
        older[task] = {it["id"]: it["question"] for it in TASKS[task]["load"](10**6)}
    return older[task][row["id"]]


def load_questions():
    rows, older = [], {}
    for task in TYPES:
        with open(RESULTS / task / "local.jsonl") as f:
            for r in map(json.loads, f):
                rows.append(dict(id=r["id"], task=task, text=question_text(r, task, older)))
    return rows


def build_model():
    return make_pipeline(
        # words AND character pieces: "def", ">>>", "A." and "\frac" are all strong hints
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=4.0),
    )


def main():
    rows = load_questions()
    train = [r for r in rows if not is_test(r["id"])]
    test = [r for r in rows if is_test(r["id"])]
    model = build_model().fit([r["text"] for r in train], [r["task"] for r in train])

    probs = model.predict_proba([r["text"] for r in test])
    labels = list(model.classes_)
    pred = [labels[p.argmax()] for p in probs]
    conf = [p.max() for p in probs]
    right = sum(p == r["task"] for p, r in zip(pred, test))
    unsure = sum(c < MIN_CONFIDENCE for c in conf)
    print(f"trained on {len(train)} questions, tested on {len(test)} unseen ones")
    print(f"accuracy: {right / len(test):.1%}   (unsure, below {MIN_CONFIDENCE:.0%}: {unsure})\n")

    print("confusion (rows = true type, columns = guessed type)")
    print(f"{'':8s}" + "".join(f"{t:>8s}" for t in labels))
    for t in labels:
        c = Counter(p for p, r in zip(pred, test) if r["task"] == t)
        print(f"{t:8s}" + "".join(f"{c[g]:8d}" for g in labels))

    # questions written the way a real user might, NOT in the datasets' formatting
    print("\nhand-written questions (no dataset formatting):")
    for q in ["What is 15% of 240?",
              "Solve for x: 3x + 7 = 22",
              "Who painted the Mona Lisa?",
              "What is the capital of Australia?",
              "Write a Python function that reverses a linked list.",
              "How do I remove duplicates from a list in Python?",
              "Which of these is a prime number? A. 21 B. 27 C. 29 D. 33",
              "Why does increasing interest rates usually reduce inflation?"]:
        p = model.predict_proba([q])[0]
        best = labels[p.argmax()]
        print(f"  {best:7s} {p.max():4.0%}  {q}")

    with open(RESULTS / "classifier.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(RESULTS / "classifier_test.json", "w") as f:
        json.dump({r["id"]: dict(true=r["task"], pred=p, conf=round(float(c), 3))
                   for r, p, c in zip(test, pred, conf)}, f)
    print(f"\nsaved {RESULTS / 'classifier.pkl'}")


if __name__ == "__main__":
    main()
