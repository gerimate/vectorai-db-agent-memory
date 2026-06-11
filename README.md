# Python Agent with Persistent Memory - Actian VectorAI DB

A conversational Python agent that remembers things **across sessions** using
[Actian VectorAI DB](https://www.actian.com/vectorai/) as a local,
on-premises vector memory store.

---

## Architecture

```
memory-layer-clean/
├── requirements.txt
├── docker-compose.yml     # VectorAI DB container
├── memory/
│   ├── __init__.py
│   ├── store.py           # MemoryStore - wraps VectorAI DB client
│   └── embeddings.py      # sentence-transformers embeddings (fully local)
├── agent/
│   ├── __init__.py
│   └── agent.py           # Agent - LLM loop with memory recall/store
└── main.py                # CLI entry point
```

**Two memory tiers:**

| Tier | Storage | Scope |
|---|---|---|
| Short-term | In-process `list` | Current session conversation, lost on exit |
| Long-term | VectorAI DB vector store | Persists across all sessions |

Each user/agent exchange is embedded and stored as an *episodic* memory.
On every new turn, the top 5 memories are retrieved by hybrid score and
injected into the system prompt.

---

## Hybrid Recall Scoring

Memories are ranked by a weighted composite score rather than raw cosine
similarity alone:

| Weight | Factor |
|---|---|
| 0.60 | Cosine similarity to the query vector |
| 0.20 | Importance score of the memory |
| 0.15 | Recency (exponential decay, half-life 168 h) |
| 0.05 | Access frequency (capped at 10 accesses) |

The default `score_threshold` in `agent/agent.py` is `0.50`, and only memories
with `importance >= 0.5` are included.

---

## Quick Start

### Prerequisites

The embedding model (`BAAI/bge-small-en-v1.5`) must be downloaded before the
first run. The code loads it with `local_files_only=True`, so it will not
download automatically at runtime:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
```

### 1. Start VectorAI DB

```bash
docker compose up -d
```

Ports:

| Port | Purpose |
|---|---|
| 6573 | REST API |
| 6574 | gRPC (SDK default) |
| 6575 | Browser UI - http://localhost:6575 |

### 2. Set up Python environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run

```bash
python main.py
```

The startup check will verify both VectorAI DB (gRPC on port 6574) and Ollama
are reachable before entering the chat loop.

---

## CLI Commands

| Input | Effect |
|---|---|
| Any text | Chat with the agent |
| `remember: <fact>` | Store an explicit high-importance fact |
| `/count` | Show total memories stored in VectorAI DB |
| `/session` | Show current session ID |
| `quit` | Exit |

---

## Memory Schema

Each stored point in the `agent_memory` collection:

| Field | Type | Description |
|---|---|---|
| `content` | `str` | Raw text of the memory |
| `session_id` | `str` | Session identifier |
| `memory_type` | `str` | `"episode"` or `"fact"` |
| `timestamp` | `float` | Unix timestamp at storage time |
| `importance` | `float` | 0.0-1.0 relevance weight |
| `access_count` | `int` | Number of times this memory has been recalled |
| `last_accessed` | `float` | Unix timestamp of the most recent recall |

---

## Exploration Angles

These can be toggled in `memory/store.py` -> `MemoryStore.recall()`:

### Memory decay
Filter by recency to exclude old memories:
```python
past_memories = self.memory.recall(query_vector=query_vec, max_age_days=30)
```

### Importance gating
Raise the threshold to surface only high-confidence memories:
```python
past_memories = self.memory.recall(query_vector=query_vec, min_importance=0.7)
```

### Memory type filtering
Retrieve only explicitly stored facts:
```python
past_memories = self.memory.recall(query_vector=query_vec, memory_type="fact")
```

### k tuning
Change the recall limit in `agent/agent.py`:
```python
past_memories = self.memory.recall(query_vector=query_vec, limit=3)   # tighter focus
past_memories = self.memory.recall(query_vector=query_vec, limit=10)  # broader context
```

---

## RAG vs Memory

The architecture is structurally identical to a Retrieval-Augmented
Generation (RAG) pipeline:

```
Embed query -> Search vector store -> Inject results into prompt -> Generate
```

The only difference is the **source of the stored vectors**:

- **RAG**: documents, PDFs, knowledge bases - indexed ahead of time.
- **Agent memory**: the agent's own past conversations - generated at runtime.

Vector databases are not a RAG-exclusive technology. They are a
general-purpose **semantic store**. As LLM agents run longer (multi-session,
multi-task), keeping memories in a vector store (cheap, unbounded, fuzzy
retrieval) is often preferable to keeping everything in-context (expensive,
bounded by window size). This is the same trade-off RAG was invented to solve,
applied to the agent's own history rather than external knowledge.
