import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx
from fetcher import Fetcher
from db_handler import DBHandler

DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "articles.db")
START_ID = 3200001
MAX_CONSECUTIVE_INVALID = 100
CONCURRENCY = 5


async def main():
    fetcher = Fetcher()
    db = DBHandler(DB_PATH)
    sem = asyncio.Semaphore(CONCURRENCY)

    consecutive_invalid = 0
    story_id = START_ID

    async with httpx.AsyncClient() as client:
        while consecutive_invalid < MAX_CONSECUTIVE_INVALID:
            result = await fetcher._fetch_story(client, sem, story_id)
            if result is None:
                consecutive_invalid += 1
            else:
                consecutive_invalid = 0
                db._insert_articles([result])
            story_id += 1

    print(
        f"Stopped at ID {story_id - 1} after {MAX_CONSECUTIVE_INVALID} "
        "consecutive invalid IDs."
    )
    db.close()


if __name__ == "__main__":
    asyncio.run(main())
