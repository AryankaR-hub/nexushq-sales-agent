# NexusHQ Sales Agent API

> Persistent AI Sales Assistant — cross-session memory, real tool use, structured self-evaluation.

**Live URL:** `https://sales-agent-production.up.railway.app`

---

## Architecture Overview

```
User Request
     │
     ▼
POST /chat/{user_id}
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                        ChatService                          │
│                                                             │
│  1. get_or_create_session(user_id, session_id?)             │
│       └─► SQLite via AbstractMemoryBackend                  │
│                                                             │
│  2. save_message(role=user, content=...)                    │
│                                                             │
│  3. SalesAgent.run(user_id, session_id, message)            │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Agentic Tool-Use Loop                  │    │
│  │                                                     │    │
│  │  Claude → tool_use blocks → dispatch callables      │    │
│  │                                                     │    │
│  │  get_user_memory(user_id)                           │    │
│  │    └─► queries UserMemory table → returns facts     │    │
│  │                                                     │    │
│  │  search_catalog(query)                              │    │
│  │    └─► keyword search over catalog.json             │    │
│  │                                                     │    │
│  │  flag_for_human(user_id, reason)  [if needed]       │    │
│  │    └─► inserts FlaggedEvent row                     │    │
│  │                                                     │    │
│  │  tool results → Claude → final text response        │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  4. EvalService.score(user_msg, response, tools, catalog)   │
│       └─► Separate LLM call → structured JSON eval block    │
│                                                             │
│  5. save_message(role=assistant, content, eval_data)        │
│                                                             │
│  6. _extract_and_store_facts() → UserMemory rows            │
│                                                             │
│  7. _maybe_compress_memory() → summarize if > threshold     │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
ChatResponse { response, eval, tools_called, session_id }
```

---

## Database Schema

```
sessions          messages                user_memory        flagged_events
─────────         ────────                ───────────        ──────────────
id (uuid)    ←── session_id              id                 id
user_id           user_id                user_id            user_id
created_at        role                   fact_type          session_id
updated_at        content                content            reason
                  groundedness           source_session_id  confidence_score
                  relevance              created_at         resolved
                  confidence             updated_at         created_at
                  flagged
                  eval_reasoning
                  tools_called (JSON)
```

---

## Memory Design

### What's stored and why

Every assistant turn extracts **atomic facts** from the conversation via a dedicated LLM call:

```
"User is evaluating Enterprise plan for a 200-person company"
"User's main concern is HIPAA compliance"
"User already uses Okta for SSO"
```

These facts are written to the `user_memory` table and retrieved as a tool call (`get_user_memory`) on every subsequent turn — regardless of session. This means context survives session boundaries without re-sending history in the request body.

### Why atomic facts over raw history

Raw history grows unboundedly. Atomic facts are:
- Compact: one row per insight, not one row per token
- Queryable: could be filtered by fact type, date, or session
- Compressible: when fact count exceeds threshold, an LLM summarizes them into one `summary` row

### Memory compression (bonus)

When a user accumulates ≥ 15 atomic facts, the service auto-compresses them into a single summary paragraph using `MEMORY_SUMMARIZATION_PROMPT`. This prevents unbounded growth while preserving context quality.

### Why the memory layer is abstracted

`app/memory/base.py` defines `AbstractMemoryBackend` — an ABC with typed method signatures. `SQLiteMemoryBackend` implements it. `app/memory/factory.py` has one import:

```python
from app.memory.sqlite_backend import SQLiteMemoryBackend  # ← only line to change
```

**To swap to Postgres:** Change `DATABASE_URL` to `postgresql+asyncpg://...` — SQLAlchemy handles the rest.  
**To swap to Mem0:** Implement `AbstractMemoryBackend` in `mem0_backend.py`, change the factory import.

### At scale

At production scale, atomic-fact extraction would move to a background task queue (Celery/ARQ), facts would be embedded and stored in a vector DB (pgvector, Pinecone) for semantic retrieval, and the `get_user_memory` tool would do a similarity search over embeddings instead of a full-table SELECT.

---

## Eval Design

### How self-scoring works

After every agent response, `EvalService.score()` makes a **separate LLM call** with a structured prompt asking the model to return JSON with:

| Score | Meaning |
|---|---|
| `groundedness` | Is the answer grounded in catalog data, not hallucinated? |
| `relevance` | Does the answer directly address what was asked? |
| `confidence` | How confident is the model in the accuracy? |
| `flagged` | Should a human review this? Auto-true if any score < 0.60 |
| `reasoning` | One-sentence explanation |

### Limitations

1. **Positivity bias** — the same model that generated the answer also evaluates it. It tends to score itself generously.
2. **No ground truth** — without reference answers, "groundedness" is the model's belief, not a fact check.
3. **Latency** — adds ~0.5–1s per response (second API call).

### What we'd replace it with at scale

- **Separate judge model** — use a different model family (e.g. GPT-4 as judge for Claude responses) to reduce self-serving bias.
- **Prometheus / G-Eval** — open-source eval frameworks with rubric-based scoring.
- **RAG faithfulness check** — compare response claims against retrieved catalog chunks using cosine similarity.
- **Human-in-the-loop** — sample 5% of flagged responses for human annotation, use those as training signal.

---

## Cross-Session Memory Demo

These two `curl` commands demonstrate memory persistence across separate API calls (no shared state between calls):

### Call 1 — Establish context (Session A)

```bash
curl -s -X POST https://sales-agent-production.up.railway.app/chat/demo-user-001 \
  -H "Content-Type: application/json" \
  -d '{"message": "Hi, we are a 300-person company evaluating enterprise options. We need SSO with Okta and HIPAA compliance. What is your Enterprise pricing?"}' \
  | python3 -m json.tool
```

Expected: Agent calls `get_user_memory` (empty) + `search_catalog("enterprise SSO HIPAA pricing")`, returns Enterprise plan details, stores facts: company size, SSO requirement, HIPAA requirement.

### Call 2 — New session, memory already loaded

```bash
curl -s -X POST https://sales-agent-production.up.railway.app/chat/demo-user-001 \
  -H "Content-Type: application/json" \
  -d '{"message": "Does that plan include audit logs? And does it work with our Okta setup?"}' \
  | python3 -m json.tool
```

Expected: Agent calls `get_user_memory` (returns: "evaluating Enterprise, 300 people, Okta, HIPAA") — **no re-statement of context needed**. Agent knows "that plan" = Enterprise. Confirms audit logs + Okta SSO compatibility.

**The `session_id` is intentionally omitted in Call 2** — a new session is created, but the user's memory persists in the DB and is injected via tool use.

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat/{user_id}` | Send message, get response + eval |
| `GET` | `/chat/{user_id}/history` | Full conversation history |
| `DELETE` | `/chat/{user_id}/memory` | GDPR-style memory wipe |
| `GET` | `/chat/{user_id}/evals` | Aggregated eval stats (bonus) |
| `GET` | `/catalog` | Full product/pricing catalog |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Interactive Swagger UI |

### Sample response — `POST /chat/{user_id}`

```json
{
  "response": "Our Enterprise plan is $499/month and includes SSO (SAML 2.0 / OIDC, compatible with Okta), audit logs, and HIPAA + SOC 2 Type II compliance — exactly what you mentioned needing. For a 300-person company, this is the right tier. Would you like to discuss custom annual pricing or a trial setup?",
  "eval": {
    "groundedness": 0.94,
    "relevance": 0.96,
    "confidence": 0.91,
    "flagged": false,
    "reasoning": "Response sourced directly from catalog. User context (300-person company, Okta, HIPAA) applied correctly from memory. No hallucination risk detected."
  },
  "tools_called": ["get_user_memory", "search_catalog"],
  "session_id": "3f9a1b2c-...",
  "user_id": "demo-user-001"
}
```

---

## Local Setup

### 1. Clone and install

```bash
git clone https://github.com/your-username/nexushq-sales-agent
cd nexushq-sales-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY
```

### 3. Run

```bash
uvicorn main:app --reload
```

API available at `http://localhost:8000`  
Docs at `http://localhost:8000/docs`

### 4. Run tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

---

## Railway Deployment

### First deploy

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Create project and deploy
railway init
railway up

# Set environment variable
railway variables set ANTHROPIC_API_KEY=your_key_here
```

The `railway.toml` configures the start command, health check path, and restart policy automatically.

### Persistent SQLite on Railway

Railway volumes keep the SQLite DB persistent across deploys:
```bash
railway volume create --name sales-agent-db --mount-path /app
```
Then update `DATABASE_URL` to `sqlite+aiosqlite:////app/sales_agent.db`.

---

## Project Structure

```
├── main.py                        # FastAPI app, lifespan, middleware
├── catalog.json                   # Mock SaaS product catalog
├── requirements.txt
├── Procfile                       # Railway start command
├── railway.toml                   # Railway deployment config
├── .env.example
└── app/
    ├── config.py                  # Settings (pydantic-settings)
    ├── api/
    │   └── routes.py              # Route handlers only — no logic
    ├── agents/
    │   ├── agent.py               # Agentic tool-use loop
    │   └── prompts.py             # All prompts versioned here
    ├── memory/
    │   ├── base.py                # AbstractMemoryBackend (swap-safe)
    │   ├── sqlite_backend.py      # SQLite / Postgres implementation
    │   └── factory.py             # One-line backend swap
    ├── tools/
    │   ├── catalog_tool.py        # search_catalog + tool definition
    │   ├── memory_tool.py         # get_user_memory + tool definition
    │   └── flag_tool.py           # flag_for_human + tool definition
    ├── services/
    │   ├── chat_service.py        # Orchestrates agent + memory + eval
    │   └── eval_service.py        # LLM self-scoring
    ├── models/
    │   └── schemas.py             # Pydantic request/response schemas
    └── db/
        ├── base.py                # Async engine, session factory, init_db
        └── models.py              # SQLAlchemy ORM models
```

---

## Tech Stack

- **FastAPI** — async web framework
- **SQLAlchemy 2.0** (async) — ORM + migrations-ready
- **aiosqlite** — async SQLite driver (swappable to asyncpg for Postgres)
- **Anthropic Python SDK** — Claude claude-sonnet-4-20250514 for agent + eval
- **Pydantic v2** — schema validation
- **Railway** — hosting
