"""Gateway v1: one API, local-first answers, cloud handoff when the local model looks lost.

Run (from src/llm_project):
    uv run uvicorn gateway.app:app --port 8000

Endpoints:
    GET  /                      demo page: ask a question, see where it was answered and why
    GET  /chat                  chat page: conversations with memory, every reply still routed
    POST /api/chat              send a message in a conversation (history is kept in SQLite)
    GET  /api/conversations     list conversations; /api/conversations/{id} for one (DELETE removes it)
    POST /v1/chat/completions   OpenAI-style request -> answer + routing info
    GET  /stats                 totals: % local, cloud spend, money saved
    GET  /health                is the server up?
"""

import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ---------- settings ----------

LOCAL_MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
CLOUD_MODEL = "openai/gpt-oss-120b"
GROQ_URL = "https://api.groq.com/openai/v1"
PRICE_IN, PRICE_OUT = 0.15, 0.60       # $ per 1M tokens, Groq gpt-oss-120b (Oct 2026 price page)
MAX_TOKENS = 2048

ROOT = Path(__file__).resolve().parent.parent           # the repo's top folder
sys.path.insert(0, str(ROOT / "router"))
from tasks import CODE_SYSTEM, MATH_SYSTEM, MMLU_SYSTEM, TRIVIA_SYSTEM   # noqa: E402

# the instruction added for each question type (the same ones used in the evaluation)
SYSTEMS = {
    "math": MATH_SYSTEM,
    "mmlu": MMLU_SYSTEM,
    "trivia": TRIVIA_SYSTEM,
    "code": CODE_SYSTEM,
    "general": "You are a helpful assistant. Answer clearly and concisely.",
}

# one routing rule per question type, each fitted by router/fit_config.py --task <type>
RULES = {t: json.loads((ROOT / "results" / t / "router_config.json").read_text())
         for t in ["math", "mmlu", "trivia", "code"]
         if (ROOT / "results" / t / "router_config.json").exists()}
FALLBACK_RULE = "math"                                    # for "general" questions

# Stage 3 verifier rules (router/fit_verifier.py): after answering, the model is asked "is your
# answer above correct?"; hand off if P(yes) < threshold. "general" borrows MMLU's rule.
VRULES = {t: json.loads((ROOT / "results" / t / "verifier_rule.json").read_text())
          for t in ["math", "mmlu", "trivia", "code"]
          if (ROOT / "results" / t / "verifier_rule.json").exists()}
VRULES.setdefault("general", VRULES.get("mmlu"))

# question-type detection: ask the local model (router/realistic.py: 40/40 on hand-written
# questions, vs 21/40 for the TF-IDF classifier, which learned dataset formatting)
from realistic import LLM_PROMPT                          # noqa: E402
from code_examples import check_examples                  # noqa: E402
TYPES = ["math", "mmlu", "trivia", "code"]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ---------- startup: load the local model once ----------

from mlx_lm import generate, load, stream_generate                 # noqa: E402
from mlx_lm.sample_utils import make_sampler              # noqa: E402
from openai import OpenAI                                 # noqa: E402

model, tokenizer = load(LOCAL_MODEL)

# the verifier LoRA shares the model's weights: OFF while answering, ON only while judging
from verifier_mlx import LORA_DIR, add_lora, p_correct, remember_scale, set_lora   # noqa: E402
lora_layers = None
if VRULES.get("math") and (LORA_DIR / "adapter_model.safetensors").exists():
    lora_layers = add_lora(model)
    remember_scale(lora_layers)
    set_lora(lora_layers, False)
sampler = make_sampler(temp=0.0)
local_lock = threading.Lock()          # one GPU, one model: run one local generation at a time
cloud = OpenAI(base_url=GROQ_URL, api_key=os.environ["GROQ_API_KEY"])

db = sqlite3.connect(LOG_DIR / "cache.db", check_same_thread=False)
db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT)")
db.execute("CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY, title TEXT, created REAL)")
db.execute("""CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, conversation INTEGER,
              role TEXT, content TEXT, router TEXT, created REAL)""")
db_lock = threading.Lock()

app = FastAPI(title="Local-first LLM router")


# ---------- request / response shapes ----------

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str = "auto"                # "auto" = route; "local" or "cloud" = force one side
    task: str = "auto"                 # math, mmlu, trivia, code, general, or "auto" = guess it


# ---------- the two sides ----------

def detect_type(question):
    """Ask the local model which type this is. Returns (type, how it was decided)."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": LLM_PROMPT}, {"role": "user", "content": question}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False)
    with local_lock:
        out = generate(model, tokenizer, prompt, max_tokens=5, sampler=sampler).strip().lower()
    found = next((t for t in TYPES if t in out), None)
    return (found, "detected by local model") if found else ("general", f"unclear ({out!r})")


def run_local(messages, cap):
    """Generate locally, token by token. Returns (text, tokens, worst_logprob, gave_up)."""
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                           tokenize=False, enable_thinking=False)
    text, n, worst = "", 0, 0.0
    with local_lock:
        for r in stream_generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS, sampler=sampler):
            text += r.text
            n += 1
            worst = min(worst, r.logprobs[r.token].item())     # this token's log-probability
            if n > cap:                                         # early stop: probably lost
                return text, n, worst, True
    return text, n, worst, False


def verify(messages, answer, signal):
    """P(the local answer is correct), from the fine-tuned ("finetuned") or plain ("base") model."""
    with local_lock:
        set_lora(lora_layers, signal == "finetuned")
        try:
            return p_correct(model, tokenizer, messages, answer)
        finally:
            set_lora(lora_layers, False)


def run_cloud(messages):
    resp = cloud.chat.completions.create(
        model=CLOUD_MODEL, messages=messages, temperature=0, max_tokens=MAX_TOKENS,
        extra_body={"reasoning_effort": "low"})
    u = resp.usage
    cost = (u.prompt_tokens * PRICE_IN + u.completion_tokens * PRICE_OUT) / 1e6
    return resp.choices[0].message.content or "", u.prompt_tokens, u.completion_tokens, cost


# ---------- the endpoint ----------

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    return route([m.model_dump() for m in req.messages], req.task, req.model)


def route(messages, task_req="auto", mode="auto"):
    """The whole router: detect the type, answer locally, check, hand off if needed."""
    if task_req == "auto":
        task, how = detect_type(messages[-1]["content"])
    else:
        task, how = (task_req if task_req in SYSTEMS else "general"), "given"
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": SYSTEMS[task]})
    rule = RULES.get(task, RULES[FALLBACK_RULE])
    cap, min_lp = rule["cap_tokens"], rule["min_logprob"]      # min_lp None = no probability check

    key = hashlib.sha256(json.dumps([mode, messages]).encode()).hexdigest()
    with db_lock:
        hit = db.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
    if hit:
        out = json.loads(hit[0])
        out["router"]["cached"] = True
        log(out["router"] | {"cost_usd": 0.0})
        return out

    t0 = time.time()
    info = {"cached": False, "task": task, "task_source": how, "local_tokens": 0, "cost_usd": 0.0}

    use_cloud = mode == "cloud"
    if not use_cloud:
        # code: no early stop, the answer gets run against the question's own examples instead
        examples_ok, vrule = None, VRULES.get(task) if lora_layers else None
        if task == "code" or vrule:
            # no early stop: the finished answer gets checked (examples run, or the verifier asked)
            text, n, worst, _ = run_local(messages, MAX_TOKENS)
            gave_up = n > cap
            if task == "code" and mode != "local":
                examples_ok = check_examples(text, messages[-1]["content"])
        else:
            text, n, worst, gave_up = run_local(messages, cap)
        info.update(local_tokens=n, worst_logprob=round(worst, 3))
        if mode == "local":
            info.update(route="local", reason="forced local")
        elif examples_ok is False:
            use_cloud, info["reason"] = True, "local code failed the examples in the question"
        elif examples_ok is True:
            info.update(route="local", reason="local code passed the examples in the question")
        elif vrule:
            p = verify(messages, text, vrule["signal"])
            info["p_correct"] = round(p, 3)
            if p < vrule["threshold"]:
                use_cloud, info["reason"] = True, f"verifier: P(correct) {p:.2f} < {vrule['threshold']:.2f}"
            else:
                info.update(route="local", reason=f"verifier: P(correct) {p:.2f}")
        elif gave_up:
            use_cloud, info["reason"] = True, f"local answer passed {cap} tokens"
        elif min_lp is not None and worst < min_lp:
            use_cloud, info["reason"] = True, f"low confidence (worst token {worst:.2f} < {min_lp:.2f})"
        else:
            info.update(route="local", reason="confident")

    if use_cloud:
        text, p_tok, c_tok, cost = run_cloud(messages)
        info.update(route="cloud", cost_usd=round(cost, 6), cloud_tokens=p_tok + c_tok)
        info.setdefault("reason", "forced cloud")
    else:
        # what this answer WOULD have cost in the cloud (rough: same answer length)
        p_tok = len(tokenizer.encode(json.dumps(messages)))
        info["saved_usd"] = round((p_tok * PRICE_IN + info["local_tokens"] * PRICE_OUT) / 1e6, 6)

    info["latency_s"] = round(time.time() - t0, 2)
    out = {"object": "chat.completion",
           "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop"}],
           "router": info}
    with db_lock:
        db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(out)))
        db.commit()
    log(info)
    return out


# ---------- chat: conversations with memory ----------

HISTORY_TOKENS = 6000      # older turns are dropped so the 4B model's prompt stays fast


class ChatTurn(BaseModel):
    text: str
    conversation_id: int | None = None
    model: str = "auto"
    task: str = "auto"


def history(conv):
    with db_lock:
        rows = db.execute("SELECT role, content, router FROM messages WHERE conversation = ? ORDER BY id",
                          (conv,)).fetchall()
    return [{"role": r, "content": c, "router": json.loads(x) if x else None} for r, c, x in rows]


@app.post("/api/chat")
def chat_turn(turn: ChatTurn):
    now = time.time()
    conv = turn.conversation_id
    if conv is None:
        with db_lock:
            conv = db.execute("INSERT INTO conversations (title, created) VALUES (?, ?)",
                              (turn.text[:60], now)).lastrowid
            db.commit()
    # earlier turns, newest kept first until the token budget runs out
    past, budget = [], HISTORY_TOKENS - len(tokenizer.encode(turn.text))
    for m in reversed(history(conv)):
        budget -= len(tokenizer.encode(m["content"]))
        if budget < 0:
            break
        past.insert(0, {"role": m["role"], "content": m["content"]})
    out = route(past + [{"role": "user", "content": turn.text}], turn.task, turn.model)
    reply = out["choices"][0]["message"]["content"]
    with db_lock:
        db.execute("INSERT INTO messages (conversation, role, content, router, created) VALUES (?,?,?,?,?)",
                   (conv, "user", turn.text, None, now))
        db.execute("INSERT INTO messages (conversation, role, content, router, created) VALUES (?,?,?,?,?)",
                   (conv, "assistant", reply, json.dumps(out["router"]), time.time()))
        db.commit()
    return {"conversation_id": conv, "reply": reply, "router": out["router"],
            "history_turns_used": len(past) // 2}


@app.get("/api/conversations")
def conversations():
    with db_lock:
        rows = db.execute("SELECT id, title, created FROM conversations ORDER BY created DESC").fetchall()
    return [{"id": i, "title": t, "created": c} for i, t, c in rows]


@app.get("/api/conversations/{conv}")
def conversation(conv: int):
    return {"id": conv, "messages": history(conv)}


@app.delete("/api/conversations/{conv}")
def delete_conversation(conv: int):
    with db_lock:
        db.execute("DELETE FROM messages WHERE conversation = ?", (conv,))
        db.execute("DELETE FROM conversations WHERE id = ?", (conv,))
        db.commit()
    return {"deleted": conv}


@app.get("/chat")
def chat_page():
    return FileResponse(Path(__file__).parent / "static" / "chat.html")


def log(info):
    with open(LOG_DIR / "requests.jsonl", "a") as f:
        f.write(json.dumps({"time": time.time()} | info) + "\n")


@app.get("/stats")
def stats():
    path = LOG_DIR / "requests.jsonl"
    rows = [json.loads(line) for line in open(path)] if path.exists() else []
    if not rows:
        return {"requests": 0}
    local_n = sum(r.get("route") == "local" for r in rows)
    return {
        "requests": len(rows),
        "answered_locally": f"{local_n / len(rows):.0%}",
        "cache_hits": sum(r.get("cached", False) for r in rows),
        "cloud_spend_usd": round(sum(r.get("cost_usd", 0) for r in rows), 6),
        "saved_usd_estimate": round(sum(r.get("saved_usd", 0) for r in rows if not r.get("cached")), 6),
        "rules": {t: {"cap_tokens": r["cap_tokens"], "min_logprob": r["min_logprob"]} for t, r in RULES.items()},
        "verifier_rules": {t: {"signal": r["signal"], "threshold": r["threshold"]}
                           for t, r in VRULES.items() if r} if lora_layers else None,
    }


@app.get("/")
def demo_page():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "local_model": LOCAL_MODEL, "cloud_model": CLOUD_MODEL}
