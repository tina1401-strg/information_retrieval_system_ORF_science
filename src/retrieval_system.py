import numpy as np
import re
from random import sample
from db_handler import DBHandler
from datetime import date
from collections import defaultdict
import bm25s
from config import (
    INDEX_CACHE, QA_INDEX_CACHE,
    EMB_CACHE,   QA_EMB_CACHE,
    QA_PROMPT_PATH,
    load_pickle,
    save_pickle
)
from models import LLM

_TITLE_SUFFIX_RE = re.compile(r"\s*-\s*science\.orf\.at\b.*", re.IGNORECASE)
_TOKEN_RE        = re.compile(r"\b\w+\b", re.UNICODE)
_SPLITTER        = None


def _get_splitter():
    global _SPLITTER
    if _SPLITTER is None:
        from charsplit import Splitter
        _SPLITTER = Splitter()
    return _SPLITTER


class RetrievalSystem:

    def __init__(self, articles: list[dict], embedder, updated_ids: list[int] = None):
        self.articles    = articles
        self.updated_ids = updated_ids or []
        self.embedder    = embedder

        # ── article retrieval indices ─────────────────────────────────────────
        self.chunk_texts, self.chunk_to_doc, _, self.chunk_emb = \
            self._build_emb("ARTICLE")
        self.bm_retriever = self._build_bm25("ARTICLE")

        # ── QA indices ────────────────────────────────────────────────────────
        self.qa_chunk_texts, self.qa_chunk_to_doc, self.qa_chunk_ids, self.qa_chunk_emb = \
            self._build_emb("QA")
        self.qa_bm_retriever = self._build_bm25("QA")

    @classmethod
    def from_db(cls, db: DBHandler, embedder) -> "RetrievalSystem":
        return cls(db.get_all_articles(), embedder, db.updated_ids)

    def get_random(self, n: int = 1) -> list[dict]:
        return sample(self.articles, n)

    # ── public retrieval ──────────────────────────────────────────────────────

    def retrieve_articles(
        self,
        query:     str,
        date_from: str | None,
        date_to:   str | None,
        db,
        top_n:     int = 3,
    ) -> list[dict]:
        fused   = self._hybrid_scores(query, date_from, date_to, db)
        top_idx = [i for i in np.argsort(fused)[::-1][:top_n] if fused[i] >= 0.8]
        return [self.articles[i] for i in top_idx]

    def retrieve_answer(
        self,
        question: str,
        entities: list[str],
        llm:      LLM,
        kg        = None,
        top_n: int = 3,
    ) -> str:
        fused = self._hybrid_scores_qa(question)
        top_cidx     = np.argsort(fused)[::-1][:top_n]
        parts = []
        for cidx in top_cidx:
            doc_idx = self.qa_chunk_to_doc[cidx]
            article = self.articles[doc_idx]
            text    = self.qa_chunk_texts[cidx]
            if text.startswith("passage: "):
                text = text[len("passage: "):]
            parts.append(
                f"[Quelle: {article.get('title', '')} | {article.get('url', '')}]\n{text}"
            )
        context = "\n\n---\n\n".join(parts)

        kg_str  = self._build_kg_str(entities, kg)
        print(f"Wissensgraph-Fakten:\n{kg_str}")
        prompt  = QA_PROMPT_PATH.read_text(encoding="utf-8").format(
            context  = context,
            kg_facts = kg_str,
            question = question,
        )
        return llm.generate(prompt, max_new_tokens=512)

    def probe_hybrid(self, query, date_from, date_to, db, articles2check=None):
        fused         = self._hybrid_scores(query, date_from, date_to, db)
        sparse_scores = self._compute_bm25_scores(query)
        dense_scores  = self._compute_emb_sim(query)
        top_idx       = [i for i in np.argsort(fused)[::-1][:10] if fused[i] >= 0.0]

        print("\n--- Hybrid Retrieval Debug ---")
        for rank, i in enumerate(top_idx):
            print(f"  Rank {rank+1} | idx: {i} | "
                  f"sparse: {sparse_scores[i]:.4f} | "
                  f"dense: {dense_scores[i]:.4f} | "
                  f"fused: {fused[i]:.4f} | "
                  f"date: {self.articles[i].get('date')} | "
                  f"title: {self.articles[i].get('title')}")
        print("------------------------------\n")

        if articles2check:
            print("\n--- Articles2Check Debug ---")
            for article_id in articles2check:
                matches = [(i, a) for i, a in enumerate(self.articles)
                           if str(a.get("id", "")) == str(article_id)]
                if not matches:
                    print(f"  ID {article_id} not found")
                    continue
                for i, article in matches:
                    rank = top_idx.index(i) + 1 if i in top_idx else "not in top 10"
                    print(f"  ID {article_id} | idx: {i} | "
                          f"sparse: {sparse_scores[i]:.4f} | "
                          f"dense: {dense_scores[i]:.4f} | "
                          f"fused: {fused[i]:.4f} | rank: {rank}")
            print("------------------------------\n")

    # ── shared scoring ────────────────────────────────────────────────────────

    def _hybrid_scores(
        self,
        query:     str,
        date_from: str | None,
        date_to:   str | None,
        db,
    ) -> np.ndarray:
        mask          = self._apply_date_mask(date_from, date_to, db)
        sparse_scores = self._compute_bm25_scores(query, "ARTICLE")
        dense_scores  = self._compute_emb_sim(query, "ARTICLE")
        fused         = self._fuse_scores(sparse_scores, dense_scores)
        fused[~mask]  = -1e9
        return fused

    def _hybrid_scores_qa(
        self,
        query:     str,
    ) -> np.ndarray:
        sparse_scores = self._compute_bm25_scores(query, "QA")
        dense_scores  = self._compute_emb_sim(query, "QA")
        fused         = self._fuse_scores(sparse_scores, dense_scores)
        return fused

    @staticmethod
    def _build_kg_str(entities: list[str], kg) -> str:
        if kg is not None and entities:
            facts = kg.query(entities)
            if facts:
                return "\n".join(f"• {f}" for f in facts)
        return "Keine KG-Fakten gefunden."

    # ── index builders ────────────────────────────────────────────────────────

    def _build_bm25(self, type: str) -> bm25s.BM25:
        cache = QA_INDEX_CACHE if type == "QA" else INDEX_CACHE

        if cache.is_file() and not self.updated_ids:
            index = load_pickle(cache)
            # sanity check for article-level BM25
            if type != "QA":
                if len(index.get_scores(["test"])) != len(self.articles):
                    print(f"  BM25 cache mismatch — rebuilding.")
                else:
                    return index
            else:
                return index

        print(f"  Building {'QA ' if type == 'QA' else ''}BM25 index ...")
        texts  = self._build_bm25_texts(type)
        tokens = [self._preprocess(t) for t in texts]
        index  = bm25s.BM25()
        index.index(tokens)
        save_pickle(index, cache)
        print(f"  Saved: {cache.name}")
        return index

    def _build_emb(self, type: str) -> tuple:
        """
        Returns:
            ARTICLE: (chunk_texts, chunk_to_doc, [], chunk_emb)
            QA:      (chunk_texts, chunk_to_doc, chunk_ids, chunk_emb)
        """
        cache = QA_EMB_CACHE if type == "QA" else EMB_CACHE

        # ── build from scratch ────────────────────────────────────────────────
        if not cache.is_file():
            print(f"  Building {'QA ' if type == 'QA' else ''}chunk embeddings ...")
            chunk_texts, chunk_to_doc, chunk_ids = self._build_chunks(self.articles, type)
            chunk_emb = self.embedder.encode(chunk_texts)
            if type == "QA":
                save_pickle((chunk_texts, chunk_to_doc, chunk_ids, chunk_emb), cache)
            else:
                save_pickle((chunk_texts, chunk_to_doc, chunk_emb), cache)
            print(f"  Saved: {cache.name}")
            return chunk_texts, chunk_to_doc, chunk_ids, chunk_emb

        # ── load from cache ───────────────────────────────────────────────────
        if not self.updated_ids:
            print(f"  Loading {'QA ' if type == 'QA' else ''}chunk embeddings: {cache.name}")
            if type == "QA":
                chunk_texts, chunk_to_doc, chunk_ids, chunk_emb = load_pickle(cache)
            else:
                chunk_texts, chunk_to_doc, chunk_emb = load_pickle(cache)
                chunk_ids = []
            return chunk_texts, chunk_to_doc, chunk_ids, chunk_emb

        # ── incremental update ────────────────────────────────────────────────
        print(f"  Updating {'QA ' if type == 'QA' else ''}chunk embeddings ...")
        if type == "QA":
            chunk_texts, chunk_to_doc, chunk_ids, chunk_emb = load_pickle(cache)
        else:
            chunk_texts, chunk_to_doc, chunk_emb = load_pickle(cache)
            chunk_ids = []

        new_articles             = [a for a in self.articles if a["id"] in set(self.updated_ids)]
        new_texts, new_to_doc, new_ids = self._build_chunks(new_articles, type)
        n_existing               = max(chunk_to_doc) + 1 if chunk_to_doc else 0
        new_to_doc               = [i + n_existing for i in new_to_doc]

        chunk_texts  += new_texts
        chunk_to_doc += new_to_doc
        chunk_emb     = np.vstack([chunk_emb, self.embedder.encode(new_texts)])

        if type == "QA":
            chunk_ids += new_ids
            save_pickle((chunk_texts, chunk_to_doc, chunk_ids, chunk_emb), cache)
        else:
            save_pickle((chunk_texts, chunk_to_doc, chunk_emb), cache)

        return chunk_texts, chunk_to_doc, chunk_ids, chunk_emb

    # ── text / chunk builders ─────────────────────────────────────────────────

    def _build_bm25_texts(self, type: str) -> list[str]:
        """Build flat text list for BM25 indexing."""
        texts = []
        for a in self.articles:
            title = _TITLE_SUFFIX_RE.sub("", a.get("title", ""))
            desc  = a.get("description", "")
            body  = a.get("markdown", "")
            if type == "QA":
                # chunk 0: title + desc
                texts.append(f"{title} {title} {title} {desc} {desc}")
                # chunks 1+: body sections only
                for chunk in [c.strip() for c in re.split(r"\n\n##", body) if c.strip()]:
                    texts.append(chunk)
            else:
                texts.append(f"{title} {title} {title} {desc} {desc} {body}")
        return texts

    @staticmethod
    def _build_chunks(articles: list[dict], type: str) -> tuple[list[str], list[int], list[str]]:
        """Build chunks for embedding. Always returns (texts, to_doc, ids)."""
        chunk_texts:  list[str] = []
        chunk_to_doc: list[int] = []
        chunk_ids:    list[str] = []

        for doc_idx, article in enumerate(articles):
            title      = _TITLE_SUFFIX_RE.sub("", article.get("title", ""))
            desc       = article.get("description", "")
            body       = article.get("markdown", "")
            article_id = article["id"]

            if type == "QA":
                # chunk 0: title + desc
                chunk_texts.append(f"passage: {title}. {desc}".strip())
                chunk_to_doc.append(doc_idx)
                chunk_ids.append(f"{article_id}_0")
                # chunks 1+: body only — no title prefix
                for ci, chunk in enumerate(
                    [c.strip() for c in re.split(r"\n\n##", body) if c.strip()], start=1
                ):
                    chunk_texts.append(f"passage: {chunk}")
                    chunk_to_doc.append(doc_idx)
                    chunk_ids.append(f"{article_id}_{ci}")
            else:
                # chunk 0: title + desc
                chunk_texts.append(f"passage: {title}. {desc}".strip())
                chunk_to_doc.append(doc_idx)
                chunk_ids.append(f"{article_id}_0")
                # chunks 1+: title + body section
                for ci, chunk in enumerate(
                    [c.strip() for c in re.split(r"\n\n##", body) if c.strip()], start=1
                ):
                    chunk_texts.append(f"passage: {title}. {chunk}".strip())
                    chunk_to_doc.append(doc_idx)
                    chunk_ids.append(f"{article_id}_{ci}")

        return chunk_texts, chunk_to_doc, chunk_ids

    # ── scoring ───────────────────────────────────────────────────────────────

    def _apply_date_mask(self, date_from, date_to, db) -> np.ndarray:
        if date_from is None and date_to is None:
            return np.ones(len(self.articles), dtype=bool)
        filtered_ids = {a["id"] for a in db.get_articles_by_date(date_from, date_to)}
        return np.array([a["id"] in filtered_ids for a in self.articles], dtype=bool)

    def _compute_bm25_scores(self, query: str, type: str, lambda_: float = 0.4, decay: float = 0.2) -> np.ndarray:
        tokens = self._preprocess(query)
        if type == "QA":
            scores = (
                self.qa_bm_retriever.get_scores(tokens).astype(float)
                if tokens else np.zeros(len(self.qa_chunk_ids))
            )
        else: 
            scores = (
                self.bm_retriever.get_scores(tokens).astype(float)
                if tokens else np.zeros(len(self.articles))
            )
            for i, article in enumerate(self.articles):
                scores[i] *= self._compute_temporal_boost(article["date"], lambda_, decay)
        return scores
    
    def _compute_emb_sim(self, query: str, type: str, lambda_: float = 0.03, decay: float = 0.2) -> np.ndarray:
        q_emb        = self.embedder.encode_query(query)
        if type == "QA":
            doc_scores  = (self.qa_chunk_emb @ q_emb).astype(float)
        else:
            chunk_scores = (self.chunk_emb @ q_emb).astype(float)
            doc_scores   = self._penalized_pool_to_docs(chunk_scores)
            for i, article in enumerate(self.articles):
                doc_scores[i] *= self._compute_temporal_boost(article["date"], lambda_, decay)
        return doc_scores

    def _penalized_pool_to_docs(self, chunk_scores: np.ndarray, penalty: float = 0.4) -> np.ndarray:
        doc_chunks = defaultdict(list)
        for chunk_idx, doc_idx in enumerate(self.chunk_to_doc):
            doc_chunks[doc_idx].append(chunk_scores[chunk_idx])
        doc_scores = np.full(len(self.articles), -np.inf)
        for doc_idx, scores in doc_chunks.items():
            spread = max(scores) - min(scores) if len(scores) > 1 else 0
            doc_scores[doc_idx] = max(scores) - penalty * spread
        return doc_scores

    @staticmethod
    def _compute_temporal_boost(article_date_str: str, lambda_: float = 0.1, decay: float = 0.3) -> float:
        age_years = (date.today() - date.fromisoformat(article_date_str)).days / 365.25
        return 1.0 + lambda_ * np.exp(-decay * age_years)

    @staticmethod
    def _fuse_scores(
        sparse_scores: np.ndarray,
        dense_scores:  np.ndarray,
        sparse_norm:   int   = 28,
        sparse_curve:  int = 3,
    ) -> np.ndarray:
        fused_scores = dense_scores + (sparse_scores / sparse_norm) ** sparse_curve
        return fused_scores
        
    # ── preprocessing ─────────────────────────────────────────────────────────

    @staticmethod
    def _preprocess(text: str) -> list[str]:
        tokens = _TOKEN_RE.findall(text.lower())
        return [part for token in tokens
                for part in RetrievalSystem._decompound_token(token)]

    @staticmethod
    def _decompound_token(token: str, conf: float = 0.8) -> list[str]:
        if len(token) < 7:
            return [token]
        result = _get_splitter().split_compound(token)
        best   = result[0]
        parts  = [p.lower() for p in best[1:] if len(p) > 2]
        if best[0] < conf or len(parts) <= 1:
            return [token]
        return [sub for part in parts for sub in RetrievalSystem._decompound_token(part, conf)]
