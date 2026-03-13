import glob
import tempfile
import os
import requests

DEFAULT_IMAGE_URL = "https://orf.at/mojo/1_4_1/storyserver//news/common/images/og-fallback-news.png"
DB_PATH = "../data/processed/articles.db"
LAST_UPDATE_FILE = "../data/processed/last_update.txt"
# ---- CONSTANTS for Fetcher ----
BASE_URL    = "https://science.orf.at/stories/{}/"
CONCURRENCY = 5
TIMEOUT     = 15
SKIP_RE_PATTERNS = [
    r"Seite nicht gefunden",
    r"404",
    r"nicht (mehr )?verfügbar",
]
SKIP_TEXT_PATTERNS = [
    "test embed",
    "karten test",
    "Zum Inhalt [AK+1]\n/\nZur ORF.at-Navigation [AK+3]\nFernsehen"
]

def clear_images():
    for f in glob.glob(tempfile.gettempdir() + "/tmp*.jpg"):
        try:
            os.unlink(f)
        except:
            pass

def load_image(image_url):
    image_path = None
    if image_url and image_url != DEFAULT_IMAGE_URL:
        img_resp = requests.get(image_url, timeout=5)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        tmp.write(img_resp.content)
        tmp.close()
        image_path = tmp.name

    return image_path