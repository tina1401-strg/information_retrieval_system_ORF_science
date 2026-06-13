# ORF Science IR Bot

A hybrid information-retrieval and question-answering system over [science.orf.at](https://science.orf.at) articles, usable as a plain terminal tool or as a Signal messenger bot.

Every query is routed by a local LLM — either returning a ranked list of articles or generating a grounded natural-language answer using retrieval-augmented generation (RAG).

---

## What it does

1. **Scrapes** article metadata and full text from science.orf.at, storing title, description, body (markdown), URL, and date.
2. **Stores** articles in a local SQLite database, skipping duplicates and refreshing entries older than 10 hours.
3. **Indexes** articles across two retrieval surfaces using a hybrid pipeline:
   - **Sparse (BM25s)** with field weighting (3× title, 2× description, 1× body) and German compound-word splitting via `charsplit` (e.g. *Medizinnobelpreis* → *Nobelpreis + Medizin*)
   - **Dense (E5-large)** using `intfloat/multilingual-e5-large` with `query:` / `passage:` prefixes, chunked at paragraph level (each chunk encoded as *title + paragraph*), aggregated to document level via penalized pooling
   - **Score-based fusion** combining both signals: `fused = dense + (sparse / 28)³` — a cubic nonlinearity that preserves score magnitudes and amplifies strong sparse matches without discarding confidence information
   - **Temporal recency boost** applied to both scores for article retrieval (not QA)
4. **Routes queries** with a local LLM that classifies intent and extracts date filters (e.g. `seit 2024` → `[2024-01-01 : *]`).
5. **Answers questions** via a RAG pipeline with knowledge-graph augmentation (see below).

---

## How it works

### Query routing

Every query passes through a local LLM that:
- Classifies it as **article retrieval** or **question answering**
- Extracts any **time references** into a structured date filter
- Passes the cleaned query to the appropriate pipeline

### Article retrieval pipeline

1. BM25s sparse scores (field-weighted, compound-split tokens)
2. E5-large dense scores (chunk-level cosine similarity → penalized pooling to document level)
3. Temporal recency boost applied to both
4. Score-based fusion: `dense + (sparse / 28)³`
5. Date mask from LLM router applied post-fusion
6. Top N articles returned if fused score ≥ threshold (default `0.8`); otherwise a "no results" message

### Question answering pipeline (RAG)

Runs two branches in parallel, then merges:

**Branch A — Entity → Knowledge Graph**
1. [GLiNER](https://github.com/urchade/GLiNER) extracts named entities from the query (persons, organisations, locations, concepts, scientific terms, etc.)
2. Entities are matched against a NetworkX knowledge graph of `subject → relation → object` triples mined from the article corpus
3. Matching facts are formatted as structured context

**Branch B — Hybrid chunk retrieval**
1. Same BM25s + E5 fusion as article retrieval, but over **paragraph-level QA chunks** (no pooling needed, no recency boost)
2. Top chunks selected and labelled with their source: `[Quelle: title | url]`

**Merge → generation**
- Chunks and KG facts are assembled into a single context block
- A local LLM generates a grounded answer (≤ 512 tokens) from the combined context

### Knowledge graph

Built offline via LLM-based triple extraction over chunked article bodies. Triples are deduplicated per update batch and stored as a NetworkX `DiGraph` (pickle). Each edge carries source article provenance (ID, URL, date). Queried at runtime by fuzzy entity matching against node names.

### Preprocessing

All text — at index build time and at query time — passes through:
- **Compound splitting** (`charsplit`, confidence-gated, recursive): splits German compound words for BM25s
- **`query:` / `passage:` prefixes** for E5 embeddings as required by the model family

---

## Requirements

- Python 3.11–3.12
- Java 25+ (for signal-cli; a bundled JDK works fine)
- `signal-cli` ≥ 0.14.3 (bot mode only — earlier versions have a breaking protocol incompatibility with Signal's server as of June 2026)
- CUDA 12-compatible GPU recommended for embedding and LLM inference

Install Python dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

---

## Usage

### Terminal mode

```bash
./run_retrieval.sh
```

Prompts for queries on stdin. Type a query in German and press Enter. The LLM router decides whether to return articles or answer the question directly.

Special command: `surprise me [N]` — returns N random articles.

### Signal bot mode

```bash
./run_retrieval.sh --bot
```

1. Scan the displayed QR code with your Signal app to link the device.
2. Send messages starting with `hey bot <query>` in any DM or group chat.
3. Article retrieval results are returned as Signal link previews (title, description, image); QA answers as plain text.
4. On exit (Ctrl-C), all Signal data is wiped from RAM and the linked device is cleaned up.

> **Note on signal-cli:** The bot requires signal-cli ≥ 0.14.3 and Java 25. Set the correct JDK path in `run_retrieval.sh` if your system Java is older.

---

## Scripts

### Bulk-download articles

```bash
cd scripts
python download_data.py
```

Fetches articles sequentially from ID `3200001` and writes them to `data/articles.db`. Stops after **100 consecutive missing IDs** (404 / skipped). Progress is printed per article (`[OK]` / `[SKIP]` / `[ERROR]`).

Run this before the first use of `run_retrieval.sh` or `probe.py`.

To backfill a specific ID range (e.g. after the warning *"Database has not been updated in over 30 days"*), set `START_ID` and `END_ID` in the script, then delete the stale index caches before restarting:

```bash
rm data/articles_intfloat_multilingual_e5_large_chunkemb.pkl
rm data/articles_bm25s.pkl
# also delete QA caches if present
rm data/qa_*.pkl
```

> Failing to delete the caches after changing the article set will cause a shape mismatch error between the cached embeddings/indices and the database.

---

## Indexing & caching

| Cache file | Contents | Update strategy |
|---|---|---|
| `articles_bm25s.pkl` | BM25s index (scipy sparse) | Full rebuild on update |
| `articles_intfloat_multilingual_e5_large_chunkemb.pkl` | Article chunk embeddings (NumPy, n_chunks × 1024) | Incremental append |
| `qa_bm25s.pkl` | QA-surface BM25s index | Full rebuild on update |
| `qa_chunkemb.pkl` | QA chunk embeddings | Incremental append |
| `kg.pkl` | Knowledge graph (NetworkX DiGraph) | Incremental, deduplicated per batch |

> **Important:** All caches for the same surface (article or QA) must be in sync. If one is deleted or rebuilt, delete the other too, or you will get a shape mismatch at startup.

---

## Fusion formula

```
fused = dense_score + (sparse_score / sparse_norm) ** sparse_curve
```

- `sparse_norm = 28` — maps BM25s scores onto the cosine-similarity scale (a sparse score of 28 contributes exactly 1.0)
- `sparse_curve = 3` — cubic nonlinearity: weak lexical matches contribute almost nothing; strong ones boost sharply

This is a deliberate alternative to Reciprocal Rank Fusion (RRF): score magnitudes are preserved as confidence signals rather than discarded by rank-based fusion.

---

## Query language

| Pattern | Effect |
|---|---|
| `hey bot <query>` | Triggers the bot (Signal mode) |
| `surprise me [N]` | Returns N random articles (terminal mode) |
| `seit 2024` / `vor 2020` | Extracted as date filter by LLM router |
| Any question | Routed to RAG pipeline |
| Any topic | Routed to article retrieval |