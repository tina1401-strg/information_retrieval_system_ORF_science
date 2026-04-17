import sys
sys.path.insert(0, "../src")

from retrieval_system import RetrievalSystem
from db_handler import DBHandler
from config import DB_PATH
from query_utils import process_query
import asyncio
import os
import subprocess
import re

def get_best_gpu(min_free_mb: int = 10000) -> int:
    """Returns the GPU index with the most free VRAM, above min_free_mb threshold."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    
    best_gpu = None
    best_free = 0
    
    for line in result.stdout.strip().split("\n"):
        idx, free, used, total = [x.strip() for x in line.split(",")]
        free_mb = int(free)
        print(f"GPU {idx}: {free_mb} MB free / {total} MB total")
        
        if free_mb > best_free and free_mb >= min_free_mb:
            best_free = free_mb
            best_gpu = int(idx)
    
    if best_gpu is None:
        raise RuntimeError(f"No GPU with at least {min_free_mb} MB free found!")
    
    print(f"→ Selected GPU {best_gpu} ({best_free} MB free)")
    return best_gpu


os.environ["CUDA_VISIBLE_DEVICES"] = str(get_best_gpu())

async def handle_query(query, retrieval, db,):
    articles2check = re.findall(r'#(\d{7})', query)
    cleaned_query = re.sub(r'#\d{7}', '', query).strip()
    expanded_query, date_filter = process_query(cleaned_query, probe=True)
    if date_filter == "null" or date_filter == "None":
        date_filter = None
    import traceback
    try:
        retrieval.probe_hybrid(expanded_query, date_filter, db, articles2check)
    except Exception:
        traceback.print_exc()

    #for rank, a in enumerate(articles, start=1):
    #    print(f"Rank {rank}:\n{a['date']}\n{a['url']}\n{a['title']}\n{a['description']}\n")

async def main():
    db = DBHandler(DB_PATH)
    await db.update()
    retrieval = RetrievalSystem.from_db(db)
    
    while True:
        try:
            query = input("Enter query: ").strip()
            if not query:
                continue
            await handle_query(query, retrieval, db)
        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print("Error:", e)
            continue

    db.close()

if __name__ == "__main__":
    asyncio.run(main())