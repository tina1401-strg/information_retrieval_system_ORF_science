from pathlib import Path
import subprocess
import os
import warnings
import os
import logging
import pickle

# ── GPU ───────────────────────────────────────────────────────────────────────
def _get_best_gpu(min_free_mb: int = 60000) -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    gpus = []
    for line in result.stdout.strip().split("\n"):
        idx, free = [x.strip() for x in line.split(",")]
        free_mb = int(free)
        if free_mb >= min_free_mb:
            gpus.append((free_mb, int(idx)))
    gpus.sort(reverse=True)
    if not gpus:
        raise RuntimeError(f"No GPU with at least {min_free_mb} MB free found!")
    return gpus[0][1]

DEVICE = f"cuda:{_get_best_gpu()}"

# ---- Paths ----
_ROOT = Path(__file__).resolve().parent.parent
QUERY_PROMPT_PATH = _ROOT / "prompts" / "query_prompt.txt"
QA_PROMPT_PATH  = _ROOT / "prompts" / "qa_prompt.txt"
KG_PATH = _ROOT / "data" / "articles_knowledge_graph.pkl"
DB_PATH = str(_ROOT / "data" / "articles.db")
LAST_UPDATE_FILE = str(_ROOT / "data" / "last_update.txt")
INDEX_CACHE = _ROOT / "data" / "articles_bm25s.pkl"
EMB_CACHE   = _ROOT / "data" / "articles_intfloat_multilingual_e5_large_chunkemb.pkl"
QA_EMB_CACHE    = _ROOT / "data" / "articles_intfloat_multilingual_e5_large_chunkemb_qa.pkl"
QA_INDEX_CACHE         = _ROOT / "data" / "articles_bm25s_qa.pkl"

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

# ---- Models ----
EMBED_MODEL  = "intfloat/multilingual-e5-large"
#LLM_MODEL    = "Qwen/Qwen2.5-7B-Instruct"
LLM_MODEL = "google/gemma-3-12B-it"
KG_MODEL = "Babelscape/mrebel-large"
KG_MAX_LENGTH = 256
KG_BATCH_SIZE = 16
NER_MODEL = "urchade/gliner_multi-v2.1"

# ---- Runtime ----

def set_online():
        os.environ["TRANSFORMERS_OFFLINE"]      = "0"
        os.environ["HF_HUB_OFFLINE"]           = "0"

def set_offline():
        os.environ["TRANSFORMERS_OFFLINE"]      = "1"
        os.environ["HF_HUB_OFFLINE"]           = "1"

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
warnings.filterwarnings("ignore") 
os.environ["TOKENIZERS_PARALLELISM"] = "false"      
os.environ["TRANSFORMERS_VERBOSITY"] = "error"       
os.environ["DATASETS_VERBOSITY"]     = "error"      
logging.getLogger("transformers").setLevel(logging.ERROR) 
logging.getLogger("huggingface_hub").setLevel(logging.ERROR) 
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# ---- Loading ----
def load_pickle(file):
    with open(file, "rb") as f:
        return pickle.load(f)

def save_pickle(obj, file) -> None:
    with open(file, "wb") as f:
        pickle.dump(obj, f)


