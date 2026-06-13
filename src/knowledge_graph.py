import re
import networkx as nx
from collections import defaultdict
from config import KG_PATH, load_pickle, save_pickle
from db_handler import DBHandler


class KnowledgeGraph:

    def __init__(self, kg_extractor, db):
        self.updated_ids = db.updated_ids or []
        self.model = kg_extractor
        self.G          = self.create(db)

    @classmethod
    def from_db(cls, db: DBHandler, kg_extractor) -> "KnowledgeGraph":
        return cls(kg_extractor, db)

    def create(self, db):
        print(f"Loading Knowledge Graph: {KG_PATH}")
        if not KG_PATH.is_file():
            print("No knowledge graph detected. Please create it before running this script.")
            return None
        
        self.G = load_pickle(KG_PATH)

        if not self.updated_ids:
            print("Knowledge graph loaded.")
            return self.G
        
        self.G = self.update(db)
        print("Knowledge graph updated.")
        return self.G

    def deduplicate(self) -> None:
        # preserve graph-level attributes across rebuild
        graph_attrs = dict(self.G.graph)

        edge_data: dict[tuple, list[dict]] = defaultdict(list)
        for subj, obj, data in self.G.edges(data=True):
            key = (subj, data["relation"], obj)
            if "sources" in data:                       # already-deduplicated edge
                edge_data[key].extend(data["sources"])
            else:                                        # freshly added raw edge
                edge_data[key].append({
                    "article_id":   data.get("article_id"),
                    "article_url":  data.get("article_url"),
                    "article_date": data.get("article_date"),
                })

        G_new = nx.DiGraph()
        G_new.graph.update(graph_attrs)  # restore graph-level attributes

        for (subj, rel, obj), sources in edge_data.items():
            if not G_new.has_node(subj):
                G_new.add_node(subj)
            if not G_new.has_node(obj):
                G_new.add_node(obj)
            G_new.add_edge(subj, obj,
                relation  = rel,
                sources   = sources,
                n_sources = len(sources),
            )

        return G_new

    def update(self, db) -> None:
        missing_ids = self.updated_ids
        if not missing_ids:
            print(f"  KG up to date — no new articles.")
            return

        articles_id = self.updated_ids
        print(f"  KG update: {len(articles_id)} new articles to process ...")

        for article_id in articles_id:
            article = db.get_article_by_id(article_id)
            text    = self.build_article_text(article)
            chunks  = self.chunk_text(text)
            triples = self.model.extract_triples(chunks)
            self._add_triples(triples, article)

        self.G = self.deduplicate()
        save_pickle(self.G, KG_PATH)
        return self.G

    def query(self, entities: list[str], max_facts: int = 10) -> list[str]:
        if not entities or self.G.number_of_nodes() == 0:
            return []
    
        facts, seen = [], set()
        for entity in entities:
            entity_lower = entity.lower()
            candidates = []
            for n in self.G.nodes():
                n_lower = n.lower()
                # exact match or entity is a full word within node (not substring)
                if (entity_lower == n_lower or
                    entity_lower in n_lower.split() or
                    n_lower in entity_lower.split()):
                    candidates.append(n)
            
            for node in candidates:
                for _, obj, data in self.G.out_edges(node, data=True):
                    t = f"{node} → {data['relation']} → {obj}"
                    if t not in seen:
                        seen.add(t)
                        facts.append(t)
                for subj, _, data in self.G.in_edges(node, data=True):
                    t = f"{subj} → {data['relation']} → {node}"
                    if t not in seen:
                        seen.add(t)
                        facts.append(t)
    
        return facts[:max_facts]

    # ── private ───────────────────────────────────────────────────────────────

    def _add_triples(self, triples: list[tuple], article: dict) -> None:
        article_id   = article["id"]
        article_url  = article.get("url", "")
        article_date = article.get("date", "")

        seen = set()
        for subj, rel, obj in triples:
            key = (subj.lower(), rel.lower(), obj.lower())
            if key in seen:
                continue
            seen.add(key)
            if not self.G.has_node(subj):
                self.G.add_node(subj)
            if not self.G.has_node(obj):
                self.G.add_node(obj)
            self.G.add_edge(subj, obj,
                relation     = rel,
                article_id   = article_id,
                article_url  = article_url,
                article_date = article_date,
            )

    # ── static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def build_article_text(article: dict) -> str:
        title = re.sub(r"\s*-\s*science\.orf\.at\b.*", "", article.get("title", ""),
                       flags=re.IGNORECASE)
        desc  = article.get("description", "")
        body  = article.get("markdown", "")
        return f"{title}. {desc} {body}".strip()

    def chunk_text(self, text: str, max_length: int = 200, stride: int = 32) -> list[str]:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text, language="german")
        chunks      = []
        current     = []
        current_len = 0

        for sent in sentences:
            sent_tokens = len(self.model.tokenize(sent))
            if current_len + sent_tokens > max_length and current:
                chunks.append(" ".join(current))
                current     = current[-1:] if stride > 0 else []
                current_len = (len(self.model.tokenize(current[0]))
                               if current else 0)
            current.append(sent)
            current_len += sent_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks