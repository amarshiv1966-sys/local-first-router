"""Pick the gateway's routing rule from Stage 1-2 data, test it honestly, save it.

Rule:  hand off to the cloud if EITHER
         - the local answer passes `cap` tokens        (checked while generating -> early stop)
         - its worst token log-prob is below `min_lp`  (checked when the answer finishes)

Two things trade off against each other:
    handoffs   = cloud cost          (fewer is cheaper)
    local time = how long users wait (a lower cap stops lost answers sooner)
For each cap, we find the best threshold, then show the whole trade-off table.

Usage:
    uv run python router/fit_config.py            # picks the balanced option
    uv run python router/fit_config.py --cap 600  # or choose a cap yourself from the table
"""

import argparse
import json

from stage2_router import MAX_TOKENS, RESULTS, TOLERANCE, is_test, load_rows

CAPS = list(range(300, MAX_TOKENS, 100)) + [MAX_TOKENS]


def handoffs(rows, cap, min_lp):
    return {k for k, s in rows.items() if s["n_tokens"] > cap or s["min_logprob"] < min_lp}


def accuracy(ids, handed, local, cloud):
    return sum(cloud[k]["correct"] if k in handed else local[k]["correct"] for k in ids) / len(ids)


def precision_recall(handed, rows, local):
    """precision: share of handoffs that were NEEDED (local was wrong).
    recall:    share of local mistakes that got handed off (caught)."""
    wrong = {k for k in rows if not local[k]["correct"]}
    caught = len(handed & wrong)
    return (caught / len(handed) if handed else 1.0), (caught / len(wrong) if wrong else 1.0)


def local_time(rows, cap, local):
    """Local seconds if every answer is cut off at `cap` tokens (assumes time grows evenly with tokens)."""
    return sum(local[k]["secs"] * min(1.0, cap / max(1, s["n_tokens"])) for k, s in rows.items())


def best_threshold(rows, cap, local, cloud, target):
    """For a fixed cap: the threshold with the fewest handoffs that still reaches the accuracy target."""
    best = None
    for lp in [float("-inf")] + sorted({s["min_logprob"] for s in rows.values()}):
        h = handoffs(rows, cap, lp)
        if accuracy(rows, h, local, cloud) >= target and (best is None or len(h) < best[0]):
            best = (len(h), lp)
    return None if best is None else best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, help="pick this cap instead of the balanced one")
    args = ap.parse_args()

    local, cloud, sigs = load_rows("local"), load_rows("cloud"), load_rows("signals")
    ids = [k for k in local if k in cloud and k in sigs]
    train = {k: sigs[k] for k in ids if not is_test(k)}
    test = {k: sigs[k] for k in ids if is_test(k)}
    target = sum(cloud[k]["correct"] for k in train) / len(train) - TOLERANCE
    full_train = local_time(train, MAX_TOKENS, local)
    full_test = local_time(test, MAX_TOKENS, local)

    print(f"target on train: within 1 point of cloud (>= {target:.1%})")
    print(f"test: {len(test)} unseen questions | local only "
          f"{sum(local[k]['correct'] for k in test) / len(test):.1%} | cloud only "
          f"{sum(cloud[k]['correct'] for k in test) / len(test):.1%}\n")
    print(f"{'cap':>5} {'threshold':>9} | {'TEST: precision':>15} {'recall':>6} {'handed off':>10} "
          f"{'accuracy':>8} {'time saved':>10}")

    options = []
    for cap in CAPS:
        lp = best_threshold(train, cap, local, cloud, target)
        if lp is None:
            continue
        h_train = handoffs(train, cap, lp)
        # balanced score on TRAIN only: share handed off + share of local time spent
        score = len(h_train) / len(train) + local_time(train, cap, local) / full_train
        h = handoffs(test, cap, lp)
        prec, rec = precision_recall(h, test, local)
        row = dict(cap=cap, lp=lp, score=score, precision=prec, recall=rec,
                   handoff=len(h) / len(test), acc=accuracy(test, h, local, cloud),
                   saved=1 - local_time(test, cap, local) / full_test)
        options.append(row)
        lp_txt = "none" if lp == float("-inf") else f"{lp:.2f}"
        print(f"{cap:>5} {lp_txt:>9} | {prec:>15.0%} {rec:>6.0%} {row['handoff']:>10.0%} "
              f"{row['acc']:>8.1%} {row['saved']:>10.0%}")

    if args.cap:
        pick = next(o for o in options if o["cap"] == args.cap)
    else:
        pick = min(options, key=lambda o: o["score"])          # chosen on train, never on test
    lp = pick["lp"]
    wrong = sum(not local[k]["correct"] for k in test) / len(test)
    print(f"\nperfect router: precision 100%, recall 100%, hands off exactly the {wrong:.0%} "
          f"local gets wrong")
    print(f"\nchosen: cap {pick['cap']} tokens, threshold "
          f"{'none' if lp == float('-inf') else f'{lp:.2f}'} -> on unseen questions: "
          f"precision {pick['precision']:.0%}, recall {pick['recall']:.0%}, "
          f"{pick['acc']:.1%} accuracy, {pick['handoff']:.0%} handed off, "
          f"{pick['saved']:.0%} local time saved")

    config = dict(cap_tokens=pick["cap"], min_logprob=None if lp == float("-inf") else lp,
                  test_precision=pick["precision"], test_recall=pick["recall"],
                  test_accuracy=pick["acc"], test_handoff_rate=pick["handoff"],
                  test_local_time_saved=pick["saved"], n_test=len(test))
    with open(RESULTS / "router_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"saved {RESULTS / 'router_config.json'}  (restart the gateway to use it)")


if __name__ == "__main__":
    main()
