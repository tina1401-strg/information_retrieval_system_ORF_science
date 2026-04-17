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

import asyncio
import re
import argparse
from db_handler import DBHandler
from retrieval_system import RetrievalSystem
from signal_connector import SignalConnector
from config import DB_PATH
from query_utils import handle_query

async def terminal_loop(retrieval, db):
    """Plain terminal loop."""
    while True:
        try:
            query = input("Enter query: ").strip()
            if not query:
                continue
            await handle_query(query, retrieval, db, cli=None)
        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print("Error:", e)
            continue

async def bot_loop(cli, retrieval, db):
    """Signal bot loop."""
    while True:
        try:
            recipient_data = await cli.receive()
            if not recipient_data:
                continue

            text = recipient_data.get("text")
            source = recipient_data.get("source")

            if not text or not source:
                continue

            if text.lower().startswith("hey bot"):
                query = re.sub(r'^hey bot[\s,!.?:]*', '', text, flags=re.IGNORECASE).strip()
                if not query:
                    continue
                await handle_query(query, retrieval, db, cli, recipient_data)

        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print("Error:", e)
            continue

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bot",
        action="store_true",
        help="Run as Signal bot. Ensure signal-cli is running first:\n"
             "signal-cli -u <YOUR_NUMBER> daemon --tcp 127.0.0.1:7583"
    )
    return parser.parse_args()

async def main():
    args = parse_args()
    db = DBHandler(DB_PATH)
    await db.update()
    retrieval = RetrievalSystem.from_db(db)

    if args.bot:
        cli = SignalConnector()
        print("Bot ready to receive queries...")
        await cli.connect()
        await bot_loop(cli, retrieval, db)
    else:
        await terminal_loop(retrieval, db)

    db.close()

if __name__ == "__main__":
    asyncio.run(main())
