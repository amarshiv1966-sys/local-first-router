"""Stage 3, step 3: does the fine-tuned verifier spot wrong answers better than the old signals?

Scored on the SAME questions and answers as Stages 1-2 (the Mac runs in results/<task>/),
which the verifier never saw in training. The rule's threshold is chosen on the training half
and measured on the test half, exactly like fit_config.py.

Fair comparison: the verifier's threshold is set so it catches as many wrong answers (recall)
on the training half as the old rule does. Then we compare how many handoffs each needs.

Two verifiers are compared, so the gain from fine-tuning is shown separately from the gain
from simply asking the model:
    base       the plain 4B model, asked "Is it correct?"   (no training)
    verifier   the same model + the LoRA from train_verifier.py

Usage (inside .venv-gpu, from the repo folder):
    python train/eval_verifier.py --base       # score with the untrained model
    python train/eval_verifier.py              # score with the fine-tuned model, then report
    python train/eval_verifier.py --report     # report only (works without a GPU)
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "router"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage2_router import is_test          # noqa: E402
from verifier_common import auroc          # noqa: E402

TYPES = ["math", "mmlu", "trivia", "code"]
R = ROOT / "results"


def load(task, name):
    path = R / task / f"{name}.jsonl"
    return {r["id"]: r for r in map(json.loads, open(path))} if path.exists() else {}


def score_all(base):
    import torch
    from unsloth import FastLanguageModel

    from verifier_common import MODEL, verifier_prompt, yes_no_ids, yes_no_logits
    name = "verifier_base" if base else "verifier"
    model, tok = FastLanguageModel.from_pretrained(MODEL if base else str(ROOT / "outputs" / "verifier_lora"),
                                                   max_seq_length=3072, load_in_4bit=True)
    tok = getattr(tok, "tokenizer", tok)
    model.eval()
    yes_id, no_id = yes_no_ids(tok)
    for task in TYPES:
        local = load(task, "local")
        done = load(task, name)
        older = None
        with open(R / task / f"{name}.jsonl", "a") as f:
            for n, (k, r) in enumerate(local.items()):
                if k in done:
                    continue
                msgs = r.get("messages")
                if msgs is None:                          # early math rows didn't save the prompt
                    if older is None:
                        from tasks import TASKS
                        older = {it["id"]: it for it in TASKS[task]["load"](10**6)}
                    msgs = TASKS[task]["messages"](older[k])
                with torch.no_grad():
                    lg = yes_no_logits(model, tok, verifier_prompt(tok, msgs, r["output"]), yes_id, no_id)
                p = torch.softmax(lg, 0)[0].item()
                f.write(json.dumps({"id": k, "p_correct": round(p, 5)}) + "\n")
                if n % 50 == 0:
                    print(f"  {task}: {n}/{len(local)}")
                    f.flush()


def stats(rows, handed):
    wrong = [not r["local_ok"] for r in rows]
    caught = sum(h and w for h, w in zip(handed, wrong))
    acc = sum(r["cloud_ok"] if h else r["local_ok"] for r, h in zip(rows, handed)) / len(rows)
    prec = caught / sum(handed) if any(handed) else 1.0
    rec = caught / sum(wrong) if any(wrong) else 1.0
    return acc, prec, rec, sum(handed) / len(rows)


def threshold_for_recall(rows, score, target):
    """Lowest threshold whose recall on these rows reaches the target."""
    wrong_scores = sorted(score(r) for r in rows if not r["local_ok"])
    if not wrong_scores:
        return 0.0
    k = max(1, round(target * len(wrong_scores) + 1e-9))
    return wrong_scores[k - 1] + 1e-9                     # hand off if score < threshold


def report():
    print("AUROC on the test half (higher = better at telling right from wrong answers)")
    print(f"{'':8s}{'fine-tuned':>11s}{'untrained':>10s}{'length':>8s}{'min log-prob':>13s}")
    tables = []
    for task in TYPES:
        local, cloud, sigs, ver, base = (load(task, n) for n in
                                         ["local", "cloud", "signals", "verifier", "verifier_base"])
        rule = json.loads((R / task / "router_config.json").read_text())
        lp = rule["min_logprob"] if rule["min_logprob"] is not None else float("-inf")
        rows = [dict(id=k, test=is_test(k), local_ok=local[k]["correct"], cloud_ok=cloud[k]["correct"],
                     p=ver[k]["p_correct"], b=base.get(k, {}).get("p_correct", float("nan")),
                     n=sigs[k]["n_tokens"], lp=sigs[k]["min_logprob"])
                for k in local if k in cloud and k in sigs and k in ver]
        if not rows:
            print(f"{task:8s} (no verifier scores yet)")
            continue
        train = [r for r in rows if not r["test"]]
        test = [r for r in rows if r["test"]]
        y = [r["local_ok"] for r in test]
        print(f"{task:8s}{auroc([r['p'] for r in test], y):11.3f}{auroc([r['b'] for r in test], y):10.3f}"
              f"{auroc([-r['n'] for r in test], y):8.3f}{auroc([r['lp'] for r in test], y):13.3f}")

        old = lambda r: r["n"] > rule["cap_tokens"] or r["lp"] < lp
        target = stats(train, [old(r) for r in train])[2]
        thr = threshold_for_recall(train, lambda r: r["p"], target)
        capped = [r for r in train if r["n"] <= rule["cap_tokens"]]
        thr2 = threshold_for_recall(capped, lambda r: r["p"], target) if capped else thr
        has_base = all(r["b"] == r["b"] for r in rows)          # nan != nan
        thr_b = threshold_for_recall(train, lambda r: r["b"], target) if has_base else None
        versions = {
            "old rule (length + log-prob)": old,
            f"fine-tuned (P<{thr:.2f})": lambda r: r["p"] < thr,
            f"cap {rule['cap_tokens']} OR verifier": lambda r: r["n"] > rule["cap_tokens"] or r["p"] < thr2,
            "always cloud": lambda r: True,
        }
        if has_base:
            versions = {**dict(list(versions.items())[:1]),
                        f"untrained (P<{thr_b:.2f})": lambda r: r["b"] < thr_b,
                        **dict(list(versions.items())[1:])}
        tables.append((task, len(test), [(name, stats(test, [f(r) for r in test])) for name, f in versions.items()]))

    for task, n, lines in tables:
        print(f"\n--- {task}: {n} test questions")
        print(f"  {'version':32s}{'accuracy':>9s}{'precision':>10s}{'recall':>8s}{'handed off':>11s}")
        for name, (acc, prec, rec, h) in lines:
            print(f"  {name:32s}{acc:9.1%}{prec:10.0%}{rec:8.0%}{h:11.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="skip scoring, only print the report")
    ap.add_argument("--base", action="store_true", help="score with the untrained model")
    args = ap.parse_args()
    if not args.report:
        score_all(args.base)
    report()


if __name__ == "__main__":
    main()
