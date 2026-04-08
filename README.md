<div align="center">

# 🧠 VibeForge AI

**Multi-agent personal assistant that actually works.**

4 API keys. 12 models. 48 concurrent slots. One answer.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

</div>

---

## What is this?

I got tired of rate limits killing my Gemini calls halfway through complex tasks. So I built a system that spreads work across multiple API keys and models simultaneously, picks the right model for each subtask based on how hard it is, and falls back gracefully when things go wrong (because they always do).

The idea is simple: you throw a complex question at it, a Planner breaks it into pieces, a pool of agents work on those pieces concurrently across all available keys/models, and a Judge stitches everything together into one coherent answer. The whole thing is grounded in a local knowledge base via RAG so the agents don't just hallucinate random stuff.

### The interesting bits

- **Semaphore-based concurrency** — not `asyncio.gather` and pray. Actual mutual exclusion over 48 resource slots with a HashMap tracking every allocation in real-time.
- **Criticality-based routing** — a task rated 9/10 gets Gemini 3.x. A boring summarization task gets Gemma 3 1B. The Planner decides, and the pool routes accordingly. If the top-tier model is rate-limited, the system auto-demotes down through the tiers instead of just crashing.
- **Three separate RAG stages** — BM25 for the Planner (keyword matching is actually better for planning docs), Vector search for the Executor (semantic similarity for technical content), and a Hybrid BM25+Vector with Reciprocal Rank Fusion for the Judge (because quality assessment needs both exact matches AND semantic understanding).
- **Two-level cache** — L1 is a straight SHA-256 hash lookup in SQLite (exact match, instant). L2 uses ChromaDB cosine similarity (so "what's the weather in NYC" and "NYC weather today" hit the same cache entry). Threshold is 0.90 to avoid false positives.

---

## Architecture

```
You ask something
        │
        ▼
 ┌─ Cache check ──────── HIT? → return immediately
 │      │ MISS
 │      ▼
 │   Planner (BM25 RAG)
 │   breaks query into N independent tasks
 │      │
 │      ▼
 │   Executor (Vector RAG)
 │   N agents run concurrently through the resource pool
 │   Semaphore(48) controls access, criticality picks the model tier
 │      │
 │      ▼
 │   Judge (Hybrid RAG)
 │   reads all agent outputs, synthesizes final answer
 │      │
 │      ▼
 └── Save to SQLite + ChromaDB + Cache
```

### Model tiers

| Tier | Models | When it's used |
|------|--------|----------------|
| T0 | Gemini 3.1 Flash Lite, Gemini 3 Flash | Hard stuff — reasoning, complex analysis |
| T1 | Gemini 2.5 Flash / Lite | Middle ground — search grounding, general tasks |
| T2 | Gemma 4 31B / 26B | Decent — thinking tasks with generous rate limits |
| T3 | Gemma 3 (27B → 1B, six variants) | Workhorse — high throughput, repetitive subtasks |

The Planner assigns criticality 1–10 to each subtask. 9-10 starts at T0, 7-8 at T1, and so on. If a tier is exhausted, we automatically try the next one down.

---

## Project layout

```
.
├── core/                     # all the actual logic lives here
│   ├── llm.py                # builds genai clients from env or BYOK keys
│   ├── resource_pool.py      # the semaphore pool — this is where the magic happens
│   ├── planner.py            # query → structured agent plan (JSON schema output)
│   ├── executor.py           # runs agents concurrently through the pool
│   ├── judge.py              # synthesizes everything into a final answer
│   ├── rag_engine.py         # BM25, Vector, and Hybrid retrievers
│   ├── knowledge_base_loader.py
│   ├── storage.py            # SQLite history + ChromaDB embeddings
│   ├── cache.py              # L1 hash + L2 semantic
│   └── tools.py              # placeholder for future tool integrations
│
├── api/                      # FastAPI backend for serving as a service
│   ├── server.py             # CORS, lifespan hooks, route mounting
│   ├── models.py             # pydantic schemas
│   ├── dependencies.py       # singleton init (pool, storage, cache)
│   └── routes/               # query, health, history, cache endpoints
│
├── ui/                       # streamlit frontend
│   ├── app.py                # 4 tabs: Chat, Agents, History, Metrics
│   ├── styles/custom.css     # dark theme, glassmorphism
│   └── components/           # sidebar, chat, agent_viz, metrics
│
├── cli/main.py               # the original REPL interface
├── tests/test_rag.py         # RAG integration tests (7 test groups)
├── knowledge_base/           # txt files organized by stage (planning/technical/evaluation/policy)
├── util/                     # dev utilities (model checker, reference data)
├── data/                     # runtime — SQLite db + ChromaDB (gitignored)
├── run.py                    # unified launcher
├── prompts.txt               # collection of stress-test prompts
└── requirements.txt
```

---

## Setup

**You'll need:** Python 3.11+ and at least one Gemini API key from [AI Studio](https://aistudio.google.com/apikey).

```bash
git clone https://github.com/Swarno-Coder/VibeForge.git
cd VibeForge

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

Copy the env template and drop in your keys:

```bash
cp .env.example .env
```

```env
NUM_KEYS=4
GEMINI_API_KEY1=AIzaSy...
GEMINI_API_KEY2=AIzaSy...
GEMINI_API_KEY3=AIzaSy...
GEMINI_API_KEY4=AIzaSy...
```

Works fine with just 1 key (you get 12 slots instead of 48). More keys = more concurrency.

---

## Running it

Three modes, one launcher:

```bash
# Web UI — opens at localhost:8501
python run.py ui

# REST API — opens at localhost:8000, Swagger docs at /docs
python run.py api

# CLI — interactive terminal
python run.py cli
```

### The UI

Dark themed Streamlit app with four tabs:
- **Chat** — type your query, watch the pipeline execute in real-time
- **Agents** — see which agents were spawned, what models they got, how many retries
- **History** — scroll through past conversations
- **Metrics** — resource pool utilization, cache hit rates, storage counts

There's a BYOK panel in the sidebar — you can paste in your own Gemini keys without touching the `.env` file. Useful for sharing the deployment with others.

### The API

Standard REST. Main endpoint:

```bash
# basic query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare EU vs US AI regulation in 2026"}'

# with your own keys (BYOK)
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "...", "api_keys": {"1": "AIzaSy..."}}'
```

Other endpoints: `GET /api/health`, `GET /api/history`, `GET /api/history/{id}`, `GET /api/cache/stats`, `DELETE /api/cache`.

Full Swagger docs at `http://localhost:8000/docs`.

### CLI commands

| Command | What it does |
|---------|-------------|
| `/history` | print recent conversations |
| `/cachestats` | show L1/L2 hit rates |
| `/clearcache` | wipe the cache |
| `/exit` | quit |

---

## Knowledge base

Drop `.txt` files into the subdirectories under `knowledge_base/`:

```
knowledge_base/
├── planning/      # Planner reads these via BM25
├── technical/     # Executor reads these via vector search
├── evaluation/    # Judge reads these via hybrid search
└── policy/        # shared across executor + judge
```

Files get chunked automatically (500 words, 50 word overlap). The system also feeds its own answers back into the knowledge base over time — so it gets better the more you use it.

---

## Tests

```bash
# unit tests — no API keys needed, tests RAG retrieval logic
python -m tests.test_rag

# full pipeline e2e — needs valid keys in .env
python -m tests.test_rag --e2e
```

7 test groups covering knowledge base loading, BM25/Vector/Hybrid retrieval accuracy, context formatting, singleton initialization, and (optionally) full pipeline execution.

---

## How the retry logic actually works

This was the hardest part to get right. Here's what happens when an agent tries to call a model:

1. Agent calls `pool.acquire(name, criticality)` → semaphore blocks if all 48 slots are in use
2. Pool finds a free slot in the agent's target tier (based on criticality)
3. Agent makes the API call
4. If it works → great, release the slot, done
5. If 429/quota error → blacklist **all keys for that model** for 60 seconds, retry
6. If JSON/parse error → blacklist **just that slot** for 60 seconds, retry
7. After 2 failures on the same tier → auto-demote to the next tier down
8. After 12 total failures → give up, return error message
9. `pool.release()` always runs (it's in a `finally` block)

The allocation HashMap (`slot_id → {agent, model, key, tier, time}`) lets you see exactly what's happening at any moment — which agent is using which key on which model.

---

## Dependencies

| Package | Why |
|---------|-----|
| `google-genai` | Gemini SDK |
| `rich` | pretty terminal output |
| `pydantic` | schema validation |
| `python-dotenv` | env file parsing |
| `chromadb` | vector store for RAG + semantic cache |
| `sentence-transformers` | embedding model (all-MiniLM-L6-v2) |
| `fastapi` + `uvicorn` | REST API |
| `streamlit` | web UI |

---

## Contributing

Fork it, make a branch, open a PR. Keep commits clean. If you're touching `resource_pool.py`, write a test — that code is load-bearing.

## License

MIT — see [LICENSE](LICENSE).
