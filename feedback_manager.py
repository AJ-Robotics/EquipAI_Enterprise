import os
import json
import uuid
from typing import List, Dict, Any
from rag_engine import VectorStore, TextSplitter
from config import FEEDBACK_FILE, FEEDBACK_VECTOR_STORE

class FeedbackManager:
    def __init__(self, embedding_model=None):
        self.feedback_store = []
        self._seen_queries = set()
        self.vector_store = VectorStore(model_name='./embedding_model', model=embedding_model)
        
        # Load existing vectors if available
        if os.path.exists(FEEDBACK_VECTOR_STORE) or os.path.exists(FEEDBACK_VECTOR_STORE + ".faiss"):
            self.vector_store.load(FEEDBACK_VECTOR_STORE)
            
        # Load raw data
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
                self.feedback_store = json.load(f)
            # Rebuild dedupe keys from already learned feedback
            for entry in self.feedback_store:
                if entry.get("rating", 0) > 0:
                    self._seen_queries.add(entry.get("query", "").strip().lower())

    def add_feedback(self, query: str, response: str, rating: int, context: List[Dict]):
        """
        rating: 1 for Like, -1 for Dislike
        """
        # Create a feedback entry
        entry = {
            "id": str(uuid.uuid4()),
            "query": query,
            "response": response,
            "rating": rating,
            "context": context
        }
        self.feedback_store.append(entry)
        self._save_json()

        # If it's a POSITIVE feedback, we want to "learn" from it (index it),
        # but only once per query to avoid redundant embeddings + disk growth.
        if rating > 0:
            key = query.strip().lower()
            if key in self._seen_queries:
                print(f"Skipping duplicate feedback learning: {query}")
                return
            self._seen_queries.add(key)

            doc_item = {
                "source": "user_feedback",
                "content": query,
                "original_response": response,
                "type": "feedback_memory"
            }
            splitter = TextSplitter()
            self.vector_store.add_documents([doc_item], splitter)
            self.vector_store.save(FEEDBACK_VECTOR_STORE)
            print(f"Learned from feedback: {query}")

    def get_relevant_past_success(self, query: str, k: int = 2) -> List[Dict]:
        """
        Finds past successful Q&A pairs similar to the current query.
        """
        results = self.vector_store.search(query, k=k)
        
        relevant_examples = []
        for res in results:
            # Logic: If query similarity is high, we include the answer
            if res['score'] > 0.5: # Threshold to ensure relevance
                # Use 'chunk_text' as VectorStore deletes 'content' and stores chunk in 'chunk_text'
                meta = res['metadata']
                relevant_examples.append({
                    "query": meta.get('chunk_text', meta.get('content', 'Unknown Query')),
                    "answer": meta.get('original_response', 'No response recorded'),
                    "score": res['score']
                })
        return relevant_examples

    def _save_json(self):
        with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.feedback_store, f, indent=2)
