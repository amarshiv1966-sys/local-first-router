# Local-first LLM router

A 4B model answers on my laptop. When it's likely to be wrong, the question goes to a 120B cloud model instead. The interesting part is the router: **can a small model tell when its own answer is wrong?**

Built on an M5 MacBook Air (16 GB) with [MLX](https://github.com/ml-explore/mlx). Cloud model: `gpt-oss-120b` on Groq.

![Cost vs accuracy](results/math/router_curve.png)

## Highlights

- **Four question types, one router:** math, multiple choice (MMLU-Pro), facts (TriviaQA) and code (HumanEval), 1,263 questions, every answer checked automatically.
- **Each type gets its own handoff rule**, fitted on one half of the questions and measured on the other. Accuracy on all held-out questions rises from 82.6% (one rule for everything) to 85.8%.
- **The type is detected by the local model itself:** 40/40 on plain hand-written questions and 32/40 on deliberately tricky ones. A TF-IDF classifier scored 97% on benchmark questions but only 15/40 and 12/40 on these, because it had learned the datasets' formatting, not their meaning, so it was dropped.
- **Code is checked by running it** against the examples written in the question. Precision of handoffs goes from 25% to **100%**, and handoffs drop from 56% to **13%**.
- **The model judges its own answers (Stage 3).** Asked "is your answer above correct?", the untrained 4B model already separates right from wrong answers (AUROC 0.77–0.96). A LoRA fine-tuned on 2,400 of its own labelled answers raises that on three of four types, most on facts (0.77 → 0.83) and MMLU-Pro (0.90 → 0.93). [Model on Hugging Face](https://huggingface.co/Godspirate/qwen3.5-4b-self-verifier).
- **RL to say HANDOFF (Stage 4).** GRPO with reward +1 right / −1 wrong / +0.3 handoff, after a supervised warm start, teaches the model to hand off *before* answering. On facts it comes within about 2 points of the separate verifier at the same handoff rate, in a single pass.

## Results across question types

Same two models, four kinds of questions, every answer checked automatically.

| Type | Questions | Local | Cloud | Perfect router | Local time |
| --- | --- | --- | --- | --- | --- |
| Math (MATH-500) | 500 | 78.2% | 93.6% | 95.4% | 42 s |
| Knowledge (MMLU-Pro) | 300 | 66.3% | 78.7% | 83.0% | 35 s |
| Facts (TriviaQA) | 299 | 44.8% | 73.6% | 76.3% | 3.6 s |
| Code (HumanEval) | 164 | 81.7% | 95.7% | 97.0% | 6.2 s |

Code answers are run against HumanEval's unit tests in a sandbox (`router/sandbox.py`).

**How well each confidence signal spots a wrong local answer (AUROC):**

| Signal | Math | MMLU-Pro | Facts | Code |
| --- | --- | --- | --- | --- |
| Answer length | **0.94** | **0.81** | 0.62 | 0.66 |
| Lowest token probability | 0.87 | 0.75 | **0.77** | 0.71 |
| Probability near the end | 0.65 | 0.65 | 0.67 | **0.76** |
| Answers that hit the 2,048-token limit | 21% | 20% | 4% | 2% |

**What this shows:** answer length is a strong warning sign where the model rambles when it's lost (math, MMLU-Pro). For facts and code, wrong answers are as short and confident as right ones, and every simple signal is weak. The 4B model is also far worse at facts (45%) than at code (82%): facts live in parameters, and a 4B model holds far fewer than a 120B one.

## Routing on held-out questions

Each rule: hand off if the answer passes a token cap (checked while generating, so lost answers stop early) or if any token's log-probability falls below a threshold. Both are chosen on the training half and measured on the test half.

- **Precision:** share of handoffs that were needed (the local answer was wrong).
- **Recall:** share of wrong local answers that were caught.

| Type | Test questions | Rule | Accuracy | Precision | Recall | Handed off |
| --- | --- | --- | --- | --- | --- | --- |
| Math | 254 | 600 tokens / −1.5 | 95.3% | 30% | 100% | 60% |
| MMLU-Pro | 142 | 1,100 tokens | 76.8% | 68% | 68% | 35% |
| Facts | 146 | 300 tokens / −0.63 | 73.3% | 59% | 96% | 91% |
| Code (confidence rule) | 86 | 300 tokens / −0.88 | 94.2% | 25% | 75% | 56% |
| **Code (run its examples)** | 86 | fails an example in the question | 93.0% | **100%** | 69% | **13%** |

**All four types together (628 test questions):**

| Version | Accuracy | Precision | Recall | Handed off |
| --- | --- | --- | --- | --- |
| Math's rule for every question | 82.6% | 43% | 77% | 56% |
| Each type's own rule | **85.8%** | 45% | **88%** | 61% |

**What this shows:**
- Per-type rules help mostly by handing off **more** where the local model is weak (facts: 61% → 73% accuracy), not by spotting mistakes better: overall precision only moves from 43% to 45%.
- Confidence signals guess; running the answer checks. For code, every answer that failed its own examples was really wrong. The catch: examples only cover a few cases, so 4 wrong answers passed them (recall 69%).
- For math, facts and multiple choice there is nothing to run, so they still depend on confidence signals. Making those more precise is the job of fine-tuning (next).

### Question-type detection

The gateway needs to know a question's type to pick its rule. Tested on 80 hand-written questions (`router/realistic.py`) phrased the way people actually ask, not like the benchmarks:

| Detector | Plain (40) | Tricky (40) |
| --- | --- | --- |
| TF-IDF + logistic regression, trained on benchmark questions | 15 | 12 |
| Same, plus 32 plain-English examples | 21 | 11 |
| **Local 4B model, asked for the type** (~0.2 s) | **40** | **32** |

Most of the model's misses on the tricky set are debatable ("what's the time complexity of sorted()?" → fact) and land on the fact rule, which hands off almost everything. So a wrong type usually costs an extra cloud call, not a wrong answer.

## Stage 3: a fine-tuned self-verifier

**Data.** The 4B model answered 2,400 *new* questions (MATH train, unused MMLU-Pro, TriviaQA train, MBPP), none from the test sets. Each answer was checked automatically: 51% right, 49% wrong.

**Training.** LoRA (rank 16, 0.5% of the weights) on an RTX 4060 laptop GPU, 30 minutes. The model sees a question and its own answer, then "Is it correct? Yes or No". The loss is cross-entropy on just that Yes/No choice, with right and wrong answers weighted equally. Its score is P(Yes).

**How well it spots wrong answers (AUROC, held-out test questions):**

| | Fine-tuned | Untrained (just asked) | Best old signal |
| --- | --- | --- | --- |
| Math | **0.975** | 0.957 | 0.973 (length) |
| MMLU-Pro | **0.929** | 0.902 | 0.779 (length) |
| Facts | **0.834** | 0.766 | 0.774 (log-prob) |
| Code | 0.895 | **0.902** | 0.715 (log-prob) |

**In the gateway** (thresholds refitted on the Mac's own scores, `router/fit_verifier.py`; set on the training half to catch as many mistakes as the old rule, then measured on the test half):

| Type | Rule | Accuracy | Precision | Handed off |
| --- | --- | --- | --- | --- |
| Math | old | 95.3% | 30% | 60% |
| | verifier | 94.5% | 33% | 54% |
| MMLU-Pro | old | 76.8% | 68% | 35% |
| | verifier (untrained) | 75.4% | 78% | 28% |
| Facts | old | 73.3% | 59% | 91% |
| | verifier | 71.9% | 66% | 79% |
| Code, no examples | old | 94.2% | 25% | 56% |
| | verifier | 96.5% | 42% | 38% |

**What this shows:** most of the gain over Stage 2 comes from *asking* the model at all; fine-tuning adds most where the old signals were weakest (facts). The verifier hands off less and more precisely everywhere, but on math, MMLU-Pro and facts it costs 0.8–1.4 points of accuracy, a real trade at these test sizes. Facts stay hardest: a model can't spot a fact it doesn't know by rereading its own answer.

The LoRA was trained in PyTorch and runs on the Mac in MLX (`router/verifier_mlx.py` transposes the matrices into mlx-lm's LoRA layers). It shares the answering model's weights: off while answering, on only while judging.

## Stage 4: RL to hand off before answering (facts)

GRPO samples 8 replies per question, scores them (+1 right, −1 wrong, +0.3 for "HANDOFF") and pushes toward the above-average ones. Out of the box the model *never* said HANDOFF (0 of 416 replies), so there was nothing to reinforce. A supervised warm start on 500 of its own answers (wrong ones relabelled HANDOFF) fixed that; then 200 GRPO steps (~70 min on the 4060).

146 held-out TriviaQA questions, compared with the Stage 3 verifier handing off the same number of questions:

| | Accuracy | Precision | Handed off |
| --- | --- | --- | --- |
| Local only | 43.8% | | 0% |
| Verifier | 72.6% | 67% | 79% |
| Warm start only | 71.9% | 61% | 79% |
| Verifier | 68.5% | 76% | 58% |
| Warm start + RL | 66.4% | 74% | 58% |
| Always cloud | 74.0% | | 100% |

RL made the model more selective than the warm start (hands off 79% → 58%, precision 61% → 74%), but the separate verifier stays about 2 points ahead. RL's advantage is cost: it decides in 1–3 tokens *before* answering, while the verifier needs the full answer plus a second pass. [RL model on Hugging Face](https://huggingface.co/Godspirate/qwen3.5-4b-rl-handoff).

## How it works

```mermaid
flowchart LR
    Q[Question] --> C{In cache?}
    C -- yes --> A[Answer]
    C -- no --> T[Local model names the type<br/>math / mmlu / facts / code]
    T --> L[Local 4B model answers<br/>token by token]
    L -- code: fails the question's examples --> H[Cloud 120B model]
    L -- otherwise --> V{Verifier:<br/>is my answer correct?}
    V -- P below the type's threshold --> H
    V -- confident --> A
    H --> A
```

The gateway (`gateway/app.py`) is a FastAPI server with an OpenAI-style endpoint and a small demo page:

- **Type first:** the local model names the question's type (or the caller passes `task`), which picks the routing rule.
- **Local first:** runs the 4B model with MLX and watches each token as it's generated.
- **Code is executed:** a code answer is run in the sandbox against the examples in the question.
- **Early stop:** if the answer passes the token cap, it stops and hands off instead of waiting.
- **Cache:** repeated questions are answered from SQLite at no cost.
- **Cost tracking:** cloud calls are priced at Groq's rates ($0.15 / $0.60 per million input / output tokens); local answers log what the cloud would have cost.
- **Transparency:** every response says where the answer came from and why.

```json
"router": {
  "task": "math",
  "task_source": "detected by local model",
  "route": "cloud",
  "reason": "local answer passed 600 tokens",
  "local_tokens": 601,
  "cost_usd": 0.000076,
  "latency_s": 13.82
}
```

## Run it

Needs an Apple Silicon Mac, [uv](https://docs.astral.sh/uv/), and a free [Groq API key](https://console.groq.com).

```bash
uv sync
export GROQ_API_KEY="your-key"

# reproduce the results
uv run python router/stage1_baseline.py local --task math --n 500
uv run python router/stage1_baseline.py cloud --task math --n 500
uv run python router/stage2_router.py score  --task math
uv run python router/stage2_router.py report --task math
uv run python router/fit_config.py --task math
# (repeat with --task mmlu / trivia / code)

uv run python router/compare.py              # one rule vs per-type rules
uv run python router/code_examples.py        # code: run answers against the question's examples
uv run python router/realistic.py --hard     # question-type detection

# start the gateway, then open http://127.0.0.1:8000 for the demo page
# (or /docs to try the raw API)
uvx hf download Godspirate/qwen3.5-4b-self-verifier --local-dir outputs/verifier_lora   # Stage 3 verifier
uv run uvicorn gateway.app:app --port 8000
```

| File | What it does |
| --- | --- |
| `router/tasks.py` | Question types: how each is loaded, prompted and checked (math, MMLU-Pro, TriviaQA, code) |
| `router/sandbox.py` | Runs model-written code against unit tests, with a timeout and resource limits (optional Docker mode) |
| `router/stage1_baseline.py` | Runs both models on a task and checks answers |
| `router/stage2_router.py` | Scores confidence signals; AUROC, cost–accuracy curves, held-out test |
| `router/fit_config.py` | Picks a type's rule on the training split, reports precision/recall on the test split |
| `router/compare.py` | One rule for everything vs each type's own rule |
| `router/code_examples.py` | Extracts the examples from a coding question and runs the answer against them |
| `router/realistic.py` | Hand-written plain and tricky questions for testing type detection |
| `router/verifier_mlx.py` | Loads the 4060-trained verifier LoRA into the MLX model |
| `router/fit_verifier.py` | Scores every answer with the verifier on the Mac and fits each type's threshold |
| `train/` | Run on an RTX 4060 (WSL2, Unsloth): `gen_data.py` makes labelled answers, `train_verifier.py` / `eval_verifier.py` (Stage 3), `rl_handoff.py` / `rl_eval.py` (Stage 4) |
| `gateway/app.py` | The API server |
| `gateway/static/index.html` | Demo page: ask a question, see where it was answered and why |
| `results/<task>/` | Raw outputs, signals, chosen rule, chart, per question type |

## Limitations

- **Benchmarks, not real traffic.** Rules were fitted on benchmark questions; real users ask differently, as the type-detection test showed.
- **The cloud is faster here.** Groq answers in about 2 s; the 4B model on a MacBook Air takes 4–43 s depending on the type. The case for local-first is cost and privacy, not speed.
- **Handoffs improve the odds, not certainty.** The cloud model is wrong too: 6% on math, 26% on facts.
- **Small test splits** (86–254 questions per type; 146 for Stage 4). Differences of a point or two are within noise.
- **The type-detection questions were written by me**, 80 of them. Treat 32/40 as indicative, not precise.
- **Training data leans short.** Generation runs shortest questions first and was stopped early for MMLU-Pro and math, so long questions are under-represented in the verifier's training data.
- **Two quantizations.** The verifier was trained on a bitsandbytes 4-bit model and runs on an MLX 4-bit one; scores differ by about 0.06 on average, which is why thresholds were refitted on the Mac.
- **Stage 4 is facts only.**
- **Time savings from early stop are estimated** from token counts, assuming time grows evenly with tokens.
- **The default code sandbox is not a security boundary.** It uses a separate process, a timeout and a throwaway folder; on macOS the memory limit isn't enforced. `SANDBOX=docker` runs code in a container with no network, a read-only file system and a memory limit.

## Next

1. **RL on all four types**, with the real dollar cost of a cloud call in the reward.
2. **Larger test sets with confidence intervals**, and a direct reproduction of [UCCI (2026)](https://arxiv.org/abs/2605.18796) on the same data.
3. **Execution checks for math**: have the model write a short program that verifies its own answer, the way code is checked now.
