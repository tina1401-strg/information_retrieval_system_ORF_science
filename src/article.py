import re
import asyncio
import httpx
import requests
from bs4 import BeautifulSoup
import html as html_lib
import trafilatura
from file_utilities import SKIP_RE_PATTERNS, SKIP_TEXT_PATTERNS, TIMEOUT, CONCURRENCY, BASE_URL

class Article:

    @staticmethod
    def is_invalid(text: str) -> bool:
        if not text:
            return True
        for pattern in SKIP_RE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        for pattern in SKIP_TEXT_PATTERNS:
            if text == pattern:
                return True
        return False

    def scrape_story_ids(self, max_pages: int = 15) -> list[int]:
        """Scrape story IDs from ORF science listing pages."""
        all_ids = []
        for page_num in range(1, max_pages + 1):
            url = f"https://science.orf.at/?page={page_num}"
            response = requests.get(url, timeout=TIMEOUT)
            soup = BeautifulSoup(response.text, "html.parser")

            ids = []
            for h2 in soup.find_all("h2"):
                a = h2.find("a")
                if a and "stories" in a["href"]:
                    story_id = a["href"].split("/stories/")[1].strip("/")
                    ids.append(int(story_id))

            if not ids:  # BUG FIX: moved outside inner loop so it actually stops
                print(f"No more stories at page {page_num}, stopping.")
                break

            all_ids.extend(ids)
            print(f"Page {page_num}: found {len(ids)} stories")

        print(f"Total IDs scraped: {len(all_ids)}")
        return all_ids

    async def _fetch_story(self, client: httpx.AsyncClient, sem: asyncio.Semaphore, story_id: int) -> dict | None:
        """Fetch and parse a single story. Internal use only."""
        url = BASE_URL.format(story_id)
        async with sem:
            try:
                response = await client.get(url, timeout=TIMEOUT, follow_redirects=True)
                html = response.text
            except Exception as e:
                print(f"[ERROR] {story_id}: {e}")
                return None

        soup = BeautifulSoup(html, "html.parser")

        title_tag = soup.find("title")
        title = html_lib.unescape(title_tag.text) if title_tag else ""

        date_tag = soup.find("meta", attrs={"name": "dc.date"})
        date = date_tag["content"] if date_tag else ""

        desc_tag = soup.find("meta", attrs={"name": "description"})
        description = desc_tag["content"] if desc_tag else ""

        og_image = soup.find("meta", property="og:image")
        image_url = og_image["content"] if og_image else ""

        result = trafilatura.extract(html, output_format="markdown", no_fallback=False)

        if self.is_invalid(title) or self.is_invalid(result):
            print(f"[SKIP] {story_id} — {title or 'no title'}")
            return None

        print(f"[OK]   {story_id} — {title} — {date}")
        return {
            "id":          story_id,
            "url":         url,
            "title":       title,
            "date":        date,
            "description": description,
            "markdown":    result,
            "image_url":   image_url,
        }

    async def fetch_stories(self, story_ids: list[int]) -> list[dict]:
        """Fetch multiple stories concurrently. Returns list of valid article dicts."""
        sem = asyncio.Semaphore(CONCURRENCY)
        async with httpx.AsyncClient() as client:
            tasks = [self._fetch_story(client, sem, sid) for sid in story_ids]
            results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
