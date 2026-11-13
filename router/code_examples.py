"""Check a code answer against the examples written in the question itself.

Most coding questions show a few examples, e.g.
    >>> special_factorial(4)
    288
or  check_if_last_char_is_a_letter("apple pie") -> False

The local answer is run against those examples in the sandbox. This uses NO hidden
tests: only what a real user's question would contain. If the answer fails its own
examples, it is almost certainly wrong -> hand off.

Usage:
    uv run python router/code_examples.py     # how well this works as a handoff rule
"""

import doctest
import json
import re
from pathlib import Path

from sandbox import run_code
from tasks import extract_code

ARROW = re.compile(r"^\s*([A-Za-z_]\w*\(.*\))\s*(?:==>|=>|->|➞|==|=|returns?|should return)\s*(.+?)\s*$")


def extract_examples(question):
    """[(call, expected), ...] taken from the question text."""
    found = []
    for ex in doctest.DocTestParser().get_examples(question):
        if ex.want.strip():
            found.append((ex.source.strip(), ex.want.strip()))
    if not found:
        defined = set(re.findall(r"def\s+(\w+)\s*\(", question))
        for line in question.splitlines():
            m = ARROW.match(line)
            if m and m.group(1).split("(")[0] in defined:
                found.append((m.group(1), m.group(2).rstrip(".,")))
    return found


def check_examples(answer, question):
    """True = passes every example, False = fails one, None = question has no usable examples."""
    examples = extract_examples(question)
    if not examples:
        return None
    code = extract_code(answer)
    if not re.search(r"def\s+\w+\s*\(", code):          # only a body: put it under the signature
        code = question + code
    checks = []
    for call, want in examples:
        checks.append(f"""
try:
    _want = eval({want!r})
except Exception:
    _want = _SKIP
if _want is not _SKIP:
    assert _eq({call}, _want), {call!r}""")
    program = code + """

_SKIP = object()
def _eq(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return abs(a - b) < 1e-6
    return a == b
""" + "\n".join(checks) + "\n"
    return run_code(program, timeout=10)["passed"]


def main():
    from stage2_router import is_test
    R = Path("results/code")
    load = lambda n: {r["id"]: r for r in map(json.loads, open(R / f"{n}.jsonl"))}
    local, cloud, sigs = load("local"), load("cloud"), load("signals")
    rule = json.loads((R / "router_config.json").read_text())

    def old_rule(k):
        lp = rule["min_logprob"] if rule["min_logprob"] is not None else float("-inf")
        return sigs[k]["n_tokens"] > rule["cap_tokens"] or sigs[k]["min_logprob"] < lp

    rows = []
    for k, r in local.items():
        if k in cloud and k in sigs and is_test(k):
            q = r["messages"][-1]["content"]
            rows.append(dict(id=k, local_ok=r["correct"], cloud_ok=cloud[k]["correct"],
                             ex=check_examples(r["output"], q), old=old_rule(k)))

    n_ex = sum(r["ex"] is not None for r in rows)
    print(f"{len(rows)} test questions, {n_ex} have usable examples in the question")
    fails = [r for r in rows if r["ex"] is False]
    print(f"local answer failed its own examples: {len(fails)}, of which actually wrong: "
          f"{sum(not r['local_ok'] for r in fails)}")
    passes = [r for r in rows if r["ex"] is True]
    print(f"passed its own examples: {len(passes)}, of which actually wrong: "
          f"{sum(not r['local_ok'] for r in passes)}\n")

    versions = {
        "old rule (length + log-prob)":   lambda r: r["old"],
        "examples only":                  lambda r: r["ex"] is False,
        "examples, old rule if none":     lambda r: r["ex"] is False if r["ex"] is not None else r["old"],
        "examples OR old rule":           lambda r: r["ex"] is False or r["old"],
        "always cloud":                   lambda r: True,
    }
    wrong = sum(not r["local_ok"] for r in rows)
    print(f"local {sum(r['local_ok'] for r in rows)/len(rows):.1%} | cloud "
          f"{sum(r['cloud_ok'] for r in rows)/len(rows):.1%} | local wrong on {wrong}")
    print(f"  {'version':30s} {'accuracy':>8s} {'precision':>9s} {'recall':>7s} {'handed off':>10s}")
    for name, f in versions.items():
        h = [f(r) for r in rows]
        acc = sum(r["cloud_ok"] if x else r["local_ok"] for r, x in zip(rows, h)) / len(rows)
        caught = sum(x and not r["local_ok"] for r, x in zip(rows, h))
        prec = caught / sum(h) if any(h) else 1.0
        rec = caught / wrong if wrong else 1.0
        print(f"  {name:30s} {acc:8.1%} {prec:9.0%} {rec:7.0%} {sum(h)/len(rows):10.0%}")


if __name__ == "__main__":
    main()
