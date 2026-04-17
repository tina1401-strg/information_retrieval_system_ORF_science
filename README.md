# ORF Science IR Bot

A hybrid information-retrieval system over [science.orf.at](https://science.orf.at) articles, usable as a plain terminal tool or as a Signal messenger bot.

## What it does

1. **Scrapes** article metadata and full text from new science.orf.at articles.
2. **Stores** articles in a local SQLite database, skipping duplicates and updating only when the data is older than 10 hours.
3. **Indexes** articles with a hybrid retrieval pipeline:
   - **BM25s** sparse retrieval with German compound-word splitting (via `charsplit`)
   - **Dense embeddings** using `intfloat/multilingual-e5-large` with chunked article bodies
   - **Both models** get a temporal recency boost each.
   - **Reciprocal Rank Fusion** to combine both scores with weighted BM25 influence and BM25s score treshold > 9.0
4. **Expands queries** with a local LLM (`qwen2.5:72b` via Ollama): abbreviations are expanded, time references are extracted

## Requirements

- Python 3.11–3.12
- [Ollama](https://ollama.com) running locally with `qwen2.5:72b` pulled
- `signal-cli` 0.14.0 (only for bot mode)
- CUDA 12 compatible GPU recommended (for embedding inference)

Install Python dependencies with [uv](https://github.com/astral-sh/uv):

## Usage

### Terminal mode

```bash
./run_retrieval.sh
```

Starts the retrieval system and prompts for queries on stdin. Type a query in German (or mixed language) and press Enter. Special command: `surprise me [N]` returns N random articles.

### Signal bot mode

```bash
./run_retrieval.sh --bot +43XXXXXXXXX
```

1. Scan the displayed QR code with your Signal app to link the device.
2. The bot listens for messages beginning with `hey bot <query>` in any DM or group chat.
3. Results are sent back as Signal link previews with title, description, and image.
4. On exit (Ctrl-C), all Signal data is wiped from RAM and the account is unregistered.

## Scripts

### Bulk-download articles

```bash
cd scripts
python download_data.py
```

Fetches articles sequentially starting from ID `3200001` and writes them to `data/articles.db`. Stops automatically after **100 consecutive IDs** that return no valid content (404 / skipped). Progress is printed for every article (`[OK]` / `[SKIP]` / `[ERROR]`). Needs to be run before first usage of run_retrieval.sh or probe.py.

### Probe Retrieval System

```bash
cd scripts
python probe.py
```

Interactive REPL for inspecting retrieval internals. 

- Query expansion and date-filter extraction run in **verbose/probe mode**, printing intermediate LLM output.
- Prefix a 7-digit article ID with `#` (e.g. `Klimawandel #3201234`) to include a specific article as a probe target — its rank and scores are outputted.
- Press `Ctrl-C` to exit.

## Query language

Queries are processed by the LLM before retrieval:

- **Abbreviations** are expanded in-place, e.g. `KI` → `KI Künstliche Intelligenz`
- **Time references** are extracted into a date filter, e.g. `seit 2024` → `[01-01-2024:*]`
- The cleaned query is then passed to the hybrid retrieval pipeline

If no relevant articles are found above the confidence threshold, the bot replies with a "no results" message.

## Configuration

All tuneable constants live in [src/config.py](src/config.py):

| Constant | Default | Description |
|---|---|---|
| `CONCURRENCY` | 5 | Parallel HTTP requests during scraping |
| `TIMEOUT` | 15 | HTTP timeout in seconds |
| `MODEL_NAME` | `intfloat/multilingual-e5-large` | Sentence embedding model |
| `LLM_MODEL` | `qwen2.5:72b` | Ollama model for query expansion |
| `DB_PATH` | `data/articles.db` | SQLite database location |
| `INDEX_CACHE` | `data/articles_bm25s.pkl` | BM25s index cache |
| `EMB_CACHE` | `data/articles_intfloat_...pkl` | Embedding cache |
