import sys
import numpy as np
import re
import pickle
from random import sample
from db_handler import DBHandler
from datetime import date
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from config import INDEX_CACHE, EMB_CACHE, MODEL_NAME

_TITLE_SUFFIX_RE = re.compile(r"\s*-\s*science\.orf\.at\b.*", re.IGNORECASE)
_TOKEN_RE        = re.compile(r"\b\w+\b", re.UNICODE)

_SPLITTER = None  # module level, lazy init


def _get_splitter():
    global _SPLITTER
    if _SPLITTER is None:
        from charsplit import Splitter
        _SPLITTER = Splitter()
    return _SPLITTER

class RetrievalSystem:

    def __init__(self, articles: list[dict], updated_ids: list[int] = None):
        self.articles = articles
        self.updated_ids = updated_ids or []
        self.model = SentenceTransformer(MODEL_NAME)
        self.chunk_texts, self.chunk_to_doc, self.chunk_emb = self._build_emb()
        self.retriever = self._build_index()

    @classmethod
    def from_db(cls, db: DBHandler) -> "RetrievalSystem":
        articles = db.get_all_articles()
        updated_ids = db.updated_ids
        return cls(articles, updated_ids)

    def get_random(self, n=1):
        return sample(self.articles, n)

    def probe_hybrid(self, query, date_filter, db, articles2check=None):
        fuse_threshold = 0.0
        mask          = self._apply_date_mask(date_filter, db)
        sparse_scores = self._compute_BM25s_scores(query)
        dense_scores  = self._compute_emb_sim(query)
        fused = self._fuse_scores(sparse_scores, dense_scores, len(self.articles))
        fused[~mask] = -1e9

        top_idx = np.argsort(fused)[::-1][:10]
        top_idx = [i for i in top_idx if fused[i] >= fuse_threshold]

        print("\n--- Hybrid Retrieval Debug ---")
        for rank, i in enumerate(top_idx):
            title = self.articles[i].get("title", "N/A")  # adjust key if needed
            date = self.articles[i].get("date", "N/A")  # adjust key if needed
            url = self.articles[i].get("url", "N/A")
            print(f"  Rank {rank+1} | idx: {i} | sparse: {sparse_scores[i]:.4f} | dense: {dense_scores[i]:.4f} | fused: {fused[i]:.4f} | date: {date} | url: {url} | title: {title}")
            print("------------------------------\n")
        
        if articles2check:
            print("\n--- Articles2Check Debug ---")
            for article_id in articles2check:
                # find the article with matching id
                matches = [(i, a) for i, a in enumerate(self.articles) if str(a.get("id", "")) == str(article_id)]
                if not matches:
                    print(f"  ID {article_id} not found in articles")
                    continue
                for i, article in matches:
                    title = article.get("title", "N/A")
                    date = article.get("date", "N/A")
                    url = article.get("url", "N/A")
                    rank = top_idx.index(i) + 1 if i in top_idx else "not in top 10"
                    print(f"  ID {article_id} | idx: {i} | sparse: {sparse_scores[i]:.4f} | dense: {dense_scores[i]:.4f} | fused: {fused[i]:.4f} | date: {date} | url: {url} | title: {title} | rank: {rank}")
                print("------------------------------\n")
    
    def retrieve_hybrid(self, query, date_filter, db, n = 3):
        fuse_threshold = 0.0
        mask          = self._apply_date_mask(date_filter, db)
        sparse_scores = self._compute_BM25s_scores(query)
        dense_scores  = self._compute_emb_sim(query)
        fused = self._fuse_scores(sparse_scores, dense_scores, len(self.articles))
        fused[~mask] = -1e9

        top_idx = np.argsort(fused)[::-1][:n]
        top_idx = [i for i in top_idx if fused[i] >= fuse_threshold]

        if not top_idx:
            return []
        return [self.articles[i] for i in top_idx]

    # -------- Build Sparse --------

    def _build_index(self):
        # cache exists and nothing new → load cache
        if INDEX_CACHE.is_file() and not self.updated_ids:
            return self._load_pickle(INDEX_CACHE)

        # no cache or new articles → rebuild fully
        try:
            import bm25s
        except ImportError:
            sys.exit("bm25s not installed. Run: pip install bm25s")

        docs = self._build_texts_weighted()
        docs_tokens = [self._preprocess(d) for d in docs]
        retriever = bm25s.BM25()
        retriever.index(docs_tokens)
        self._save_pickle(retriever, INDEX_CACHE)
        return retriever

    def _build_texts_weighted(self) -> list[str]:
        texts = []
        for a in self.articles:
            title = a.get("title", "")
            title = _TITLE_SUFFIX_RE.sub("", title)
            desc  = a.get("description", "")
            body  = a.get("markdown", "")
            text  = f"{title} {title} {title} {desc} {desc} {body}"
            texts.append(text)
        return texts

    # -------- Build Dense --------

    def _build_emb(self):
        if not EMB_CACHE.is_file():
            chunk_texts, chunk_to_doc = self._build_chunks(self.articles)
            chunk_emb = self._encode(chunk_texts)
            self._save_pickle((chunk_texts, chunk_to_doc, chunk_emb), EMB_CACHE)
            return chunk_texts, chunk_to_doc, chunk_emb

        if not self.updated_ids:
            return self._load_pickle(EMB_CACHE)

        chunk_texts, chunk_to_doc, chunk_emb = self._load_pickle(EMB_CACHE)
        new_articles = [a for a in self.articles if a["id"] in set(self.updated_ids)]
        new_chunk_texts, new_chunk_to_doc = self._build_chunks(new_articles)

        n_existing_docs = max(chunk_to_doc) + 1 if chunk_to_doc else 0
        new_chunk_to_doc = [i + n_existing_docs for i in new_chunk_to_doc]

        new_chunk_emb = self._encode(new_chunk_texts)

        chunk_texts += new_chunk_texts
        chunk_to_doc += new_chunk_to_doc
        chunk_emb = np.vstack([chunk_emb, new_chunk_emb])

        self._save_pickle((chunk_texts, chunk_to_doc, chunk_emb), EMB_CACHE)
        return chunk_texts, chunk_to_doc, chunk_emb
    
    @staticmethod
    def _build_chunks(articles) -> tuple[list[str], list[int]]:
        """Split each article body on \\n\\n## and prepend title to each chunk.
        Returns:
        chunk_texts  — list of strings to encode
        chunk_to_doc — parallel list mapping chunk index → article index
        """
        chunk_texts:  list[str] = []
        chunk_to_doc: list[int] = []
        for doc_idx, article in enumerate(articles):
            title = article.get("title", "")
            desc  = article.get("description", "")
            body  = article.get("markdown", "")
            title = _TITLE_SUFFIX_RE.sub("", title)
            prefix = "passage: "
            # title + description as its own chunk
            chunk_texts.append(f"{prefix}{title}. {desc}".strip())
            chunk_to_doc.append(doc_idx)
            # split body on ## headings
            for chunk in [c.strip() for c in re.split(r"\n\n##", body) if c.strip()]:
                chunk_texts.append(f"{prefix}{title}. {chunk}".strip())
                chunk_to_doc.append(doc_idx)
        return chunk_texts, chunk_to_doc

    # -------- Retrieve Scores --------

    def _apply_date_mask(self, date_filter: str | None, db) -> np.ndarray:
        if not date_filter or date_filter.strip().lower() == "null":
            return np.ones(len(self.articles), dtype=bool)

        # let SQL do the filtering
        filtered = db.get_articles_by_date(date_filter)
        filtered_ids = {a["id"] for a in filtered}

        return np.array([a["id"] in filtered_ids for a in self.articles], dtype=bool)

    def _compute_BM25s_scores(self, query: str, lambda_: float = 0.4, decay: float = 0.2) -> np.ndarray:
        tokens = self._preprocess(query)
        if not tokens:
            return np.zeros(len(self.articles))
        #query_tokens = [tokens]
        scores = self.retriever.get_scores(tokens)
        #scores = scores[0]
        for i, article in enumerate(self.articles):
            boost = self._compute_temporal_boost(article['date'], lambda_=lambda_, decay=decay)
            scores[i] *= boost
        return scores

    def _compute_emb_sim(self, query, lambda_: float = 0.03, decay: float = 0.2):
        q_emb = self._encode_query(query)
        chunk_scores = (self.chunk_emb @ q_emb).astype(float)
        doc_scores = self._penalized_pool_to_docs(chunk_scores)
        for i, article in enumerate(self.articles):
            boost = self._compute_temporal_boost(article['date'], lambda_=lambda_, decay=decay)
            doc_scores[i] *= boost
        return doc_scores

    def _encode_query(self, query: str) -> np.ndarray:
        q_embs = self._encode([f"query: {query}"])  # ← wrap in list
        return q_embs[0]

    def _encode(self, chunk_texts: list[str]):
        return self.model.encode(
            chunk_texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def _penalized_pool_to_docs(self, chunk_scores, penalty=0.4):
        doc_chunks = defaultdict(list)
        for chunk_idx, doc_idx in enumerate(self.chunk_to_doc):
            doc_chunks[doc_idx].append(chunk_scores[chunk_idx])

        doc_scores = np.full(len(self.articles), -np.inf)
        for doc_idx, scores in doc_chunks.items():
            max_score = max(scores)
            spread = max(scores) - min(scores) if len(scores) > 1 else 0
            doc_scores[doc_idx] = max_score - penalty * spread
        return doc_scores

    @staticmethod
    def _compute_temporal_boost(article_date_str: str, lambda_: float = 0.1, decay: float = 0.3) -> float:
        article_date = date.fromisoformat(article_date_str)
        today = date.today()
        age_years = (today - article_date).days / 365.25
        return 1.0 + lambda_ * np.exp(-decay * age_years)

    @staticmethod
    def _fuse_scores(sparse_scores: np.ndarray, dense_scores: np.ndarray, n_docs: int, k: int = 60, threshold: float = 4.0, sparse_weight: float = 0.4):
        dense_order = np.argsort(dense_scores)[::-1]
        dense_ranks = np.empty(n_docs, dtype=int)
        dense_ranks[dense_order] = np.arange(n_docs)

        sparse_order = np.argsort(sparse_scores)[::-1]
        sparse_ranks = np.empty(n_docs, dtype=int)
        sparse_ranks[sparse_order] = np.arange(n_docs)

        fused = np.zeros(n_docs)

        # STrans always contributes fully
        fused += 1.0 / (k + dense_ranks + 1)

        # BM25 only contributes when score exceeds threshold
        sparse_mask = sparse_scores > threshold
        fused[sparse_mask] += sparse_weight * (1.0 / (k + sparse_ranks[sparse_mask] + 1))

        return fused

    @staticmethod
    def _preprocess(text: str) -> list[str]:
        tokens = _TOKEN_RE.findall(text.lower())
        return [
            part
            for token in tokens
            for part in RetrievalSystem._decompound_token(token)
        ]

    @staticmethod
    def _decompound_token(token: str, conf: int = 0.8) -> list[str]:
        if len(token) < 7:
            return [token]
        splitter = _get_splitter()
        result = splitter.split_compound(token)
        best = result[0]
        score = best[0]
        parts = [p.lower() for p in best[1:] if len(p) > 2]
        if score < conf or len(parts) <= 1:
            return [token]
        final_parts = []
        for part in parts:
            sub_parts = RetrievalSystem._decompound_token(part, conf)
            final_parts.extend(sub_parts)
        return final_parts

# -------- Utils --------

    @staticmethod
    def _load_pickle(file):
        with open(file, "rb") as f:
            return pickle.load(f)
        
    @staticmethod
    def _save_pickle(obj, file) -> None:
        with open(file, "wb") as f:
            pickle.dump(obj, f)
