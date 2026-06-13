import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx
from fetcher import Fetcher
from db_handler import DBHandler

DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "articles.db")
START_ID = 3200001
END_ID = None
#START_ID = 3235000
#END_ID = 3235853
MAX_CONSECUTIVE_INVALID = 100
CONCURRENCY = 5


async def main():
    fetcher = Fetcher()
    db = DBHandler(DB_PATH)
    sem = asyncio.Semaphore(CONCURRENCY)

    existing_ids = set(db.get_existing_ids()) 

    if END_ID:
        ids_to_fetch = [i for i in range(START_ID, END_ID + 1) if i not in existing_ids]
        print(f"Fetching {len(ids_to_fetch)} new IDs (skipping {END_ID - START_ID + 1 - len(ids_to_fetch)} existing)")

        async with httpx.AsyncClient() as client:
            tasks = [fetcher.fetch_story(client, sem, sid) for sid in ids_to_fetch]
            results = await asyncio.gather(*tasks)

        new_articles = [r for r in results if r is not None]
        if new_articles:
            db.insert_articles(new_articles)
        print(f"Inserted {len(new_articles)} new articles.")

    else:
        consecutive_invalid = 0
        story_id = START_ID
        async with httpx.AsyncClient() as client:
            while consecutive_invalid < MAX_CONSECUTIVE_INVALID:
                result = await fetcher.fetch_story(client, sem, story_id)
                if result is None:
                    consecutive_invalid += 1
                else:
                    consecutive_invalid = 0
                    if result.id not in existing_ids:
                        db._insert_articles([result])
                story_id += 1

        print(f"Stopped at ID {story_id - 1} after {MAX_CONSECUTIVE_INVALID} consecutive invalid IDs.")

    db.close()


if __name__ == "__main__":
    asyncio.run(main())
