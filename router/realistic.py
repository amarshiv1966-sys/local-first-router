"""Which type detector works on questions written the way real people ask them?

Three detectors, scored on 40 hand-written questions (10 per type) that look nothing
like the benchmark datasets:

    old      classifier.pkl from classify.py (trained only on dataset questions)
    retrain  the same classifier, plus a few plain-English examples (EXTRA below)
    llm      ask the local 4B model to name the type, no training at all

The EXTRA examples and the TEST questions are different questions, so "retrain" is
never tested on what it was trained on.

Usage:
    uv run python router/realistic.py          # normal questions
    uv run python router/realistic.py --hard   # tricky ones
"""

import pickle
import time
from collections import Counter
from pathlib import Path

from classify import MIN_CONFIDENCE, TYPES, build_model, load_questions
from stage2_router import is_test

RESULTS = Path("results")

# plain-English training examples, added to the dataset questions for "retrain"
EXTRA = {
    "math": ["What is 20% of 350?", "Find x if 5x - 3 = 17.", "What is the derivative of x^3?",
             "How many ways can 4 people sit in a row?", "Simplify (x^2 - 9)/(x - 3).",
             "What is the area of a circle with radius 7?", "Is 91 a prime number?",
             "What is the sum of the angles in a hexagon?"],
    "code": ["Write a Python function to check if a string is a palindrome.",
             "How do I sort a dictionary by value in Python?",
             "Fix this code: for i in range(10) print(i)",
             "Write a function that returns the nth Fibonacci number.",
             "How can I read a CSV file in Python?", "Implement binary search in Python.",
             "Why does my Python loop never end?", "Write code to merge two sorted lists."],
    "mmlu": ["Which gas do plants absorb? A. Oxygen B. Nitrogen C. Carbon dioxide D. Helium",
             "Which is a mammal?\nA) Shark\nB) Dolphin\nC) Trout\nD) Octopus",
             "The main cause of inflation is: (A) high demand (B) low prices (C) more savings (D) none",
             "Pick the correct option: Newton's second law says A. F=ma B. E=mc^2 C. V=IR D. PV=nRT",
             "Which organ makes insulin? A. Liver B. Pancreas C. Kidney D. Heart",
             "Which of these is not a programming language? A. Python B. Java C. HTML D. C++",
             "Choose one: the speed of sound is fastest in A) air B) water C) steel D) vacuum",
             "Which country has the largest population? Options: A. India B. USA C. Brazil D. Japan"],
    "trivia": ["Who wrote Pride and Prejudice?", "What is the capital of Canada?",
               "Which planet is known as the red planet?", "Who discovered penicillin?",
               "In which year did World War II end?", "What is the largest ocean on Earth?",
               "Who was the first person to walk on the moon?", "What currency is used in Japan?"],
}

# the test: different questions, phrased like a real user
TEST = {
    "math": ["What is 15% of 240?", "Solve for x: 3x + 7 = 22", "What's the integral of 2x dx?",
             "If a train goes 60 km/h for 2.5 hours, how far does it travel?",
             "How many diagonals does an octagon have?", "What is the LCM of 12 and 18?",
             "Expand (a + b)^2", "What is the probability of rolling two sixes with two dice?",
             "A rectangle has perimeter 30 and length 9. What is its width?",
             "What is log base 2 of 64?"],
    "code": ["Write a Python function that reverses a linked list.",
             "How do I remove duplicates from a list in Python?",
             "Write a function to count vowels in a string.",
             "What's wrong with this: def f(x) return x*2",
             "How do I make an HTTP request in Python?",
             "Write a Python class for a stack with push and pop.",
             "Give me code to find the maximum element in an array.",
             "How do I convert a string to an integer in Python?",
             "Write a function that checks whether two strings are anagrams.",
             "Implement quicksort in Python."],
    "mmlu": ["Which of these is a prime number? A. 21 B. 27 C. 29 D. 33",
             "What is H2O commonly called? A) Salt B) Water C) Sugar D) Acid",
             "Which part of the cell makes energy? (A) Nucleus (B) Ribosome (C) Mitochondria (D) Wall",
             "The Treaty of Versailles ended which war? A. WWI B. WWII C. Cold War D. Korean War",
             "Which is a renewable energy source?\nA. Coal\nB. Solar\nC. Oil\nD. Natural gas",
             "Supply and demand: if demand rises and supply stays fixed, price will A. fall B. rise C. stay D. vanish",
             "Which is the smallest unit of memory? A. Byte B. Bit C. Kilobyte D. Nibble",
             "Ohm's law relates: A) voltage, current, resistance B) force, mass C) heat, work D) none",
             "Which blood type is the universal donor? Options: A. A+ B. AB+ C. O- D. B-",
             "Who proposed the theory of relativity? (A) Newton (B) Einstein (C) Bohr (D) Tesla"],
    "trivia": ["Who painted the Mona Lisa?", "What is the capital of Australia?",
               "Which band released the album Abbey Road?", "What is the tallest mountain in the world?",
               "Who is the author of the Harry Potter books?", "What is the chemical symbol for gold?",
               "Which country hosted the 2016 Summer Olympics?", "How many players are on a football team?",
               "What language is spoken in Brazil?", "Who invented the telephone?"],
}

# tricky questions: mixed signals, numbers in facts, code words in math, options without letters...
HARD = {
    "math": ["My code outputs 7 for 3 + 4 * 1. What should 3 + 4 * 2 give?",
             "A Python list has 12 items. How many ways can I pick 3 of them?",
             "Who wins more often: rolling a 6 on one die, or a total of 7 on two dice?",
             "What year comes 1,000 days after Jan 1, 2020?",
             "The capital is $5000 at 8% compound interest. What is it after 3 years?",
             "If an algorithm takes n^2 steps, how many steps for n = 1500?",
             "Options are A=3, B=5, C=7. What is A*B + C?",
             "How many seconds are in a leap year?",
             "x is twice y and their sum is 27. What is x?",
             "Which is bigger, 2^100 or 100^2, and by how much roughly?"],
    "code": ["What does this print? print(sum(range(10)))",
             "Is it better to use a list or a set to check membership in Python?",
             "I need a regex that matches Indian phone numbers.",
             "Explain what *args means in a Python function.",
             "Convert this to a list comprehension: out=[]; for x in a: out.append(x*x)",
             "My recursion hits maximum depth on n=5000. How do I fix it?",
             "What's the time complexity of Python's sorted()?",
             "Write SQL to get the second highest salary from an employees table.",
             "How do I compute the factorial of 50 in Python without overflow?",
             "Who created Python, and show me hello world in it."],
    "mmlu": ["Which sorting algorithm is stable? A. Quicksort B. Heapsort C. Merge sort D. Selection sort",
             "What is 7 x 8? A. 54 B. 56 C. 58 D. 64",
             "Who wrote Hamlet?\nA. Dickens\nB. Shakespeare\nC. Austen\nD. Tolstoy",
             "What does print(2**3) output? (A) 6 (B) 8 (C) 9 (D) 23",
             "True or false style: pick one. (A) The sun is a planet (B) The sun is a star",
             "Choose the derivative of sin x: a) cos x b) -cos x c) tan x d) -sin x",
             "Big-O of binary search? i. O(n) ii. O(log n) iii. O(1) iv. O(n log n)",
             "Which year did India gain independence? 1) 1945 2) 1947 3) 1950 4) 1952",
             "Select the noble gas from: A. Oxygen, B. Argon, C. Chlorine, D. Sodium",
             "Best answer: a function calling itself is called A. iteration B. recursion C. lambda D. closure"],
    "trivia": ["How many bones are in the adult human body?",
               "In what year was the Python programming language first released?",
               "Who proved Fermat's Last Theorem?",
               "What is the value of pi to 2 decimal places?",
               "Which company makes the C++ compiler MSVC?",
               "What is the boiling point of water in Fahrenheit?",
               "Who is known as the father of computers?",
               "How many squares are on a chessboard's surface, counting only the 64 small ones? Just the number.",
               "What does the A in ASCII stand for?",
               "Which mathematician is the Fibonacci sequence named after?"],
}

LLM_PROMPT = ("Classify the user's question into exactly one type:\n"
              "math - a math problem to calculate or solve\n"
              "code - writing, fixing or explaining program code\n"
              "mmlu - a multiple-choice question with lettered options\n"
              "trivia - a short factual question (who, what, where, when)\n"
              "Reply with only the type name.")


def classifier_detector(model):
    labels = list(model.classes_)

    def guess(q):
        p = model.predict_proba([q])[0]
        return labels[p.argmax()] if p.max() >= MIN_CONFIDENCE else "unsure"
    return guess


def llm_detector():
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    model, tok = load("mlx-community/Qwen3.5-4B-MLX-4bit")
    sampler = make_sampler(temp=0.0)

    def guess(q):
        prompt = tok.apply_chat_template([{"role": "system", "content": LLM_PROMPT},
                                          {"role": "user", "content": q}],
                                         add_generation_prompt=True, tokenize=False,
                                         enable_thinking=False)
        out = generate(model, tok, prompt, max_tokens=5, sampler=sampler).strip().lower()
        return next((t for t in TYPES if t in out), "unsure")
    return guess


def evaluate(name, guess, test):
    t0 = time.time()
    rows = [(t, q, guess(q)) for t in TYPES for q in test[t]]
    ms = (time.time() - t0) / len(rows) * 1000
    right = sum(t == g for t, _, g in rows)
    print(f"--- {name}: {right}/{len(rows)} right ({right / len(rows):.0%}), {ms:.0f} ms per question")
    for t in TYPES:
        c = Counter(g for tt, _, g in rows if tt == t)
        print(f"  {t:7s} {c[t]:2d}/{len(test[t])}   wrong guesses: {dict((k, v) for k, v in c.items() if k != t) or '-'}")
    for t, q, g in rows:
        if g != t:
            print(f"    {t:7s} -> {g:7s} {q[:70]!r}")
    print()
    return right


def main():
    old = pickle.loads((RESULTS / "classifier.pkl").read_bytes())

    # retrain: same training split as classify.py, plus EXTRA
    train = [r for r in load_questions() if not is_test(r["id"])]
    X = [r["text"] for r in train] + [q for t in TYPES for q in EXTRA[t]]
    y = [r["task"] for r in train] + [t for t in TYPES for _ in EXTRA[t]]
    new = build_model().fit(X, y)

    import sys
    test = HARD if "--hard" in sys.argv else TEST
    print(f"test set: {'HARD' if test is HARD else 'normal'} ({sum(map(len, test.values()))} questions)\n")
    scores = {"old": evaluate("old classifier", classifier_detector(old), test),
              "retrain": evaluate("retrained classifier", classifier_detector(new), test),
              "llm": evaluate("local model asked", llm_detector(), test)}

    with open(RESULTS / "classifier_retrained.pkl", "wb") as f:
        pickle.dump(new, f)
    print("scores (out of 40):", scores)
    print(f"saved {RESULTS / 'classifier_retrained.pkl'}")


if __name__ == "__main__":
    main()
