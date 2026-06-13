import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from config import QUERY_PROMPT_PATH
from models import LLM, EntityExtractor

LIST_TRIGGERS = ["alle artikel", "zeig alle", "liste alle", "list alle", "alle beiträge"]


@dataclass
class QueryResult:
    query_type:    str
    cleaned_query: str
    date_from:     str | None
    date_to:       str | None
    entities:      list[str] = field(default_factory=list)

    def has_date_filter(self) -> bool:
        return self.date_from is not None or self.date_to is not None

    def __repr__(self) -> str:
        return (
            f"QueryResult(type={self.query_type}, query='{self.cleaned_query}', "
            f"date={self.date_from}→{self.date_to}, entities={self.entities})"
        )


class QueryHandler:
    def __init__(self, llm: LLM, gliner: EntityExtractor = None):
        self.prompt = QUERY_PROMPT_PATH.read_text(encoding="utf-8")
        self.llm    = llm
        self.gliner = gliner

    # ── public ────────────────────────────────────────────────────────────────

    def process(self, query: str, probe: bool = False) -> QueryResult:
        today  = date.today().strftime("%d. %B %Y")
        prompt = self.prompt.format(today=today, query=query)
        raw    = self.llm.generate(prompt, max_new_tokens=256)
        result = self._parse(raw)
    
        if not result.cleaned_query:
            result.cleaned_query = query
    
        # entity extraction via GLiNER on the cleaned query
        if self.gliner is not None:
            result.entities = self.gliner.extract(result.cleaned_query)
    
        if probe:
            print(f"  [Router] raw:\n{raw}")
            print(f"  TYP:      {result.query_type}")
            print(f"  ANFRAGE:  {result.cleaned_query!r}")
            print(f"  DATUM:    {result.date_from} → {result.date_to}")
            print(f"  ENTITÄTEN: {result.entities}")
    
        return result

    # ── private ───────────────────────────────────────────────────────────────

    def _parse(self, response: str) -> QueryResult:
        query_type    = "RETRIEVE"
        cleaned_query = ""
        date_from     = None
        date_to       = None
        entities      = []

        for line in response.strip().splitlines():
            line       = line.strip()
            line_upper = line.upper()

            if line_upper.startswith("TYP:"):
                query_type = "QA" if "QA" in line_upper[4:] else "RETRIEVE"

            elif line_upper.startswith("ANFRAGE:"):
                cleaned_query = line[8:].strip().strip('"')

            elif line_upper.startswith("DATUM:"):
                val = line[6:].strip()
                if val.lower() != "null":
                    m = re.match(
                        r'\[(\*|\d{4}-\d{2}-\d{2}):(\*|\d{4}-\d{2}-\d{2})\]', val
                    )
                    if m:
                        date_from = None if m.group(1) == "*" else m.group(1)
                        date_to   = None if m.group(2) == "*" else m.group(2)

        return QueryResult(
            query_type    = query_type,
            cleaned_query = cleaned_query,
            date_from     = date_from,
            date_to       = date_to,
            entities      = entities,
        )

# ── standalone helpers ────────────────────────────────────────────────────────

def _is_list_query(query: str) -> bool:
    return any(query.strip().lower().startswith(t) for t in LIST_TRIGGERS)

async def execute_query(
    query:          str,
    handler:        QueryHandler,
    retrieval,
    db,
    kg,
    llm:            LLM,
    cli             = None,
    recipient_data  = None,
):
    articles = []
    answer   = None

    if query.startswith("surprise me"):
        match    = re.search(r'\d+', query)
        n        = min(int(match.group(0)) if match else 1, 10)
        articles = retrieval.get_random(n)
    else:
        result = handler.process(query, probe=True)
        if _is_list_query(result.cleaned_query):
            articles = db.get_articles_by_date(result.date_from, result.date_to)
        elif result.query_type == "RETRIEVE":
            try:
                articles = retrieval.retrieve_articles(
                    result.cleaned_query,
                    result.date_from,
                    result.date_to,
                    db,
                )
            except Exception:
                import traceback
                traceback.print_exc()
                articles = []

        else:  # QA
            try:
                answer = retrieval.retrieve_answer(
                    question = result.cleaned_query,
                    entities = result.entities,
                    llm      = llm,
                    kg       = kg,
                )
            except Exception:
                import traceback
                traceback.print_exc()
                answer = None

    if cli:
        if answer is not None:
            await cli.send_text(answer, recipient_data)
        elif not articles:
            await cli.send_text("Leider konnten wir keine passenden Artikel finden.", recipient_data)
        else:
            await cli.send_article(articles, recipient_data)
    else:
        if answer is not None:
            print(f"\nAntwort:\n{answer}\n")
        elif not articles:
            print("Leider konnten wir keine passenden Artikel finden.")
        else:
            for rank, a in enumerate(articles, start=1):
                print(f"Rank {rank}:\n{a['date']}\n{a['url']}\n{a['title']}\n{a['description']}\n")