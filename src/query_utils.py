import re
from datetime import date
import ollama
from config import PROMPT_PATH, LLM_MODEL
LIST_TRIGGERS = ["alle artikel", "zeig alle", "liste alle","list alle", "alle beiträge"]

def parse_query_date(raw: str) -> tuple[str, str]:
    lines = raw.strip().split('\n')
    expanded_query = lines[0].replace('ANFRAGE:', '').strip()
    date_str = lines[1].replace('DATUM:', '').strip()
    date_filter = None if date_str.lower() == "null" else date_str
    return expanded_query, date_filter

def call_llm(prompt: str) -> str:
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.message.content.strip()

def process_query(query: str, probe: str = False):
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    prompt = prompt_template.format(
        query=query,
        today=date.today().strftime("%d-%m-%Y"),
    )
    raw = call_llm(prompt)
    expanded_query, date_filter = parse_query_date(raw)
    if not expanded_query:
        expanded_query = query
    if probe:
        print(f"         ANFRAGE: {expanded_query!r}")
        print(f"         DATUM:   {date_filter!r}")
    return expanded_query, date_filter

def _is_list_query(query: str) -> bool:
    return any(query.strip().lower().startswith(t) for t in LIST_TRIGGERS)

async def handle_query(query, retrieval, db, cli, recipient_data=None):

    if query.startswith("surprise me"):
        match = re.search(r'\d+', query)
        n = int(match.group(0)) if match else 1
        n = min(n, 10)
        articles = retrieval.get_random(n)
    else:
        expanded_query, date_filter = process_query(query)
        if date_filter == "null":
            date_filter = None
        if _is_list_query(expanded_query):
            articles = db.get_articles_by_date(date_filter)
        else:
            import traceback
            try:
                articles = retrieval.retrieve_hybrid(expanded_query, date_filter, db)
            except Exception as e:
                traceback.print_exc()  # ← shows exact line

    if cli:
        if not articles:
            text = "Leider konnten wir keine passenden Artikel finden."
            await cli.send_text(text, recipient_data)
        else:
            await cli.send_article(articles, recipient_data)
    else:
        if not articles:
            print("Leider konnten wir keine passenden Artikel finden.")
        else:
            for rank, a in enumerate(articles, start=1):
                print(f"Rank {rank}:\n{a['date']}\n{a['url']}\n{a['title']}\n{a['description']}\n")
