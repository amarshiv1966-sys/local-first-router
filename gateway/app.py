"""Gateway v1: one API, local-first answers, cloud handoff when the local model looks lost.

Run (from src/llm_project):
    uv run uvicorn gateway.app:app --port 8000

Endpoints:
    GET  /                      demo page: ask a question, see where it was answered and why
    POST /v1/chat/completions   OpenAI-style request -> answer + routing info
    GET  /stats                 totals: % local, cloud spend, money saved
    GET  /health                is the server up?
"""

import hashlib
import json
import os
import sqlite3
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
DEFAULT_SYSTEM = "Solve the math problem. Think step by step, then put only the final answer in \\boxed{}."

ROOT = Path(__file__).resolve().parent.parent           # src/llm_project
CONFIG = json.loads((ROOT / "results" / "math" / "router_config.json").read_text())   # math-only rule for now
CAP = CONFIG["cap_tokens"]
MIN_LP = CONFIG["min_logprob"]                           # None = no probability check
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ---------- startup: load the local model once ----------

from mlx_lm import load, stream_generate                 # noqa: E402
from mlx_lm.sample_utils import make_sampler              # noqa: E402
from openai import OpenAI                                 # noqa: E402

model, tokenizer = load(LOCAL_MODEL)
sampler = make_sampler(temp=0.0)
local_lock = threading.Lock()          # one GPU, one model: run one local generation at a time
cloud = OpenAI(base_url=GROQ_URL, api_key=os.environ["GROQ_API_KEY"])

db = sqlite3.connect(LOG_DIR / "cache.db", check_same_thread=False)
db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT)")
db_lock = threading.Lock()

app = FastAPI(title="Local-first LLM router")


# ---------- request / response shapes ----------

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str = "auto"                # "auto" = route; "local" or "cloud" = force one side


# ---------- the two sides ----------

def run_local(messages):
    """Generate locally, token by token. Returns (text, tokens, worst_logprob, gave_up)."""
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                           tokenize=False, enable_thinking=False)
    text, n, worst = "", 0, 0.0
    with local_lock:
        for r in stream_generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS, sampler=sampler):
            text += r.text
            n += 1
            worst = min(worst, r.logprobs[r.token].item())     # this token's log-probability
            if n > CAP:                                         # early stop: probably lost
                return text, n, worst, True
    return text, n, worst, False


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
    messages = [m.model_dump() for m in req.messages]
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM})

    key = hashlib.sha256(json.dumps([req.model, messages]).encode()).hexdigest()
    with db_lock:
        hit = db.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
    if hit:
        out = json.loads(hit[0])
        out["router"]["cached"] = True
        log(out["router"] | {"cost_usd": 0.0})
        return out

    t0 = time.time()
    info = {"cached": False, "local_tokens": 0, "cost_usd": 0.0}

    use_cloud = req.model == "cloud"
    if not use_cloud:
        text, n, worst, gave_up = run_local(messages)
        info.update(local_tokens=n, worst_logprob=round(worst, 3))
        if req.model == "local":
            info.update(route="local", reason="forced local")
        elif gave_up:
            use_cloud, info["reason"] = True, f"local answer passed {CAP} tokens"
        elif MIN_LP is not None and worst < MIN_LP:
            use_cloud, info["reason"] = True, f"low confidence (worst token {worst:.2f} < {MIN_LP:.2f})"
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
        "rule": {"cap_tokens": CAP, "min_logprob": MIN_LP},
    }


@app.get("/")
def demo_page():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "local_model": LOCAL_MODEL, "cloud_model": CLOUD_MODEL}
