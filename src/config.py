from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# ---- Database ----
DB_PATH = str(_ROOT / "data" / "articles.db")
LAST_UPDATE_FILE = str(_ROOT / "data" / "last_update.txt")

# ---- Fetcher ----
BASE_URL          = "https://science.orf.at/stories/{}/"
DEFAULT_IMAGE_URL = "https://orf.at/mojo/1_4_1/storyserver//news/common/images/og-fallback-news.png"
CONCURRENCY       = 5
TIMEOUT           = 15
SKIP_RE_PATTERNS  = [
    r"Seite nicht gefunden",
    r"404",
    r"nicht (mehr )?verfügbar",
]
SKIP_TEXT_PATTERNS = [
    "test embed",
    "karten test",
    "Zum Inhalt [AK+1]\n/\nZur ORF.at-Navigation [AK+3]\nFernsehen",
]

# ---- Retrieval ----
INDEX_CACHE = _ROOT / "data" / "articles_bm25s.pkl"
EMB_CACHE   = _ROOT / "data" / "articles_intfloat_multilingual_e5_large_chunkemb.pkl"
MODEL_NAME  = "intfloat/multilingual-e5-large"

# ---- LLM / Query expansion ----
LLM_MODEL   = "qwen2.5:72b"
PROMPT_PATH = _ROOT / "prompt" / "prompt_abbrev_time.txt"
