from db_handler import DBHandler
from retrieval_system import RetrievalSystem
from signal_connector import SignalConnector
from file_utilities import DB_PATH
import asyncio
import re
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bot",
        action="store_true",
        help="Run as Signal bot. Ensure signal-cli is running first:\n"
             "signal-cli -u <YOUR_NUMBER> daemon --tcp 127.0.0.1:7583"
    )
    return parser.parse_args()


async def handle_query(query, retrieval, cli, recipient_data=None):
    """Handle a query from either terminal or Signal bot."""
    if query.startswith("surprise me"):
        match = re.search(r'\d+', query)
        n = int(match.group(0)) if match else 1
        n = min(n, 10)
        articles = retrieval.get_random(n)
    else:
        articles = retrieval.retrieve_bm25(query)

    if cli:
        await cli.send_article(articles, recipient_data)
    else:
        for a in articles:
            print(f"{a['url']}\n{a['description']}\n\n")


async def bot_loop(cli, retrieval):
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

            if "🙄" in text:
                await cli.send_text("nerv nicht", recipient_data)
                continue

            if text.lower().startswith("hey bot"):
                query = re.sub(r'^hey bot[\s,!.?:]*', '', text, flags=re.IGNORECASE).strip()
                if not query:
                    continue
                await handle_query(query, retrieval, cli, recipient_data)

        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print("Error:", e)
            continue


async def terminal_loop(retrieval):
    """Plain terminal loop."""
    while True:
        try:
            query = input("Enter query: ").strip()
            if not query:
                continue
            await handle_query(query, retrieval, cli=None)
        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print("Error:", e)
            continue


async def main():
    args = parse_args()
    db = DBHandler(DB_PATH)
    await db.update()
    retrieval = RetrievalSystem(db)

    if args.bot:
        cli = SignalConnector()
        await cli.connect()
        await bot_loop(cli, retrieval)
    else:
        await terminal_loop(retrieval)

    db.close()


if __name__ == "__main__":
    asyncio.run(main())