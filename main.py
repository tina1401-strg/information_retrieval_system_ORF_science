import asyncio
import re
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from db_handler import DBHandler
from retrieval_system import RetrievalSystem
from signal_connector import SignalConnector
from config import DB_PATH, set_online, set_offline
from query_handler import execute_query, QueryHandler
from knowledge_graph import KnowledgeGraph
from models import KGExtractor, LLM, EntityExtractor, Embedder, cleanup

async def _watch_for_exit():
    loop = asyncio.get_event_loop()
    while True:
        line = await loop.run_in_executor(None, input, "")
        if line.strip().lower() == "exit":
            print("Exit command received — shutting down...")
            return

async def terminal_loop(retrieval, db, query_handler, kg, llm):
    while True:
        try:
            query = input("Enter query: ").strip()
            if not query:
                continue
            await execute_query(query, query_handler, retrieval, db, kg, llm, cli=None)
        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print("Error:", e)
            continue

async def bot_loop(cli, retrieval, db, query_handler, kg, llm):
    exit_watcher = asyncio.create_task(_watch_for_exit())
    while True:
        try:
            if exit_watcher.done():
                break

            recipient_data = await cli.receive()
            if not recipient_data:
                continue
            text   = recipient_data.get("text")
            source = recipient_data.get("source")
            if not text or not source:
                continue
            if text.lower().startswith("hey bot"):
                query = re.sub(r'^hey bot[\s,!.?:]*', '', text, flags=re.IGNORECASE).strip()
                if not query:
                    continue
                await execute_query(query, query_handler, retrieval, db, kg, llm, cli, recipient_data)
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
        action = "store_true",
        help   = "Run as Signal bot."
    )
    return parser.parse_args()

async def main():
    args = parse_args()

    # --------------- Update Database ---------------
    db = DBHandler(DB_PATH)
    await db.update()

    # --------------- Download Models from Hugging Face ---------------
    set_online()

    kg_extractor = KGExtractor()
    kg = KnowledgeGraph.from_db(db, kg_extractor)

    embedder  = Embedder()
    retrieval = RetrievalSystem.from_db(db, embedder)
    embedder.to_cpu()

    cleanup()

    llm    = LLM()
    gliner = EntityExtractor()

    query_handler = QueryHandler(llm, gliner)

    set_offline()
    # --------------- Download Done ---------------

    # ── run ───────────────────────────────────────────────────────────────────
    if args.bot:
        cli = SignalConnector()
        print("Bot ready to receive queries...")
        await cli.connect()
        await bot_loop(cli, retrieval, db, query_handler, kg, llm)
    else:
        await terminal_loop(retrieval, db, query_handler, kg, llm)

    db.close()

if __name__ == "__main__":
    asyncio.run(main())