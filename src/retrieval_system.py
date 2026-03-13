# This is the real magic

from random import sample
from db_handler import DBHandler
from rank_bm25 import BM25Okapi

class RetrievalSystem:

    def __init__(self, db):
        #self.idx = self.get_indices()
        self.db = db

    def get_random(self, n=1):
        articles = self.db.get_all_articles()
        return sample(articles, n)
    
    def get_indices():
        # whatever they look like
        return None
    
    #def find_articles():
     #   emb = self.get_embeddings()
    
    def get_embeddings():
        if DBHandler.updated_ids:
            pass
    
    def retrieve_bm25(self, query, n=3):
        articles = self.db.get_all_articles()
        docs = [
            {
                "doc_id": str(article["id"]),
                "text": f"{article['title']}. {article['description']} {article['markdown']}"
            }
            for article in self.db.get_all_articles()
        ]

        tokenized_corpus = [self.simple_tokenize(doc["text"]) for doc in docs]
        bm25 = BM25Okapi(tokenized_corpus)
        top_docs =  bm25.get_top_n(self.simple_tokenize(query), docs, n=n)
        top_ids = {doc["doc_id"] for doc in top_docs}
        return [article for article in articles if str(article["id"]) in top_ids]

    @staticmethod
    def simple_tokenize(text: str):
        return text.lower().split()

