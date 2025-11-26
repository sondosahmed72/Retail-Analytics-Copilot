import os
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Resolve correct absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))            # agent/rag
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))        # your_project/


class DocRetriever:
    """
    TF-IDF markdown document retriever.
    Loads files from docs folder and splits them into paragraph chunks.
    """
    def __init__(self, docs_path: str):
        # Safe fixed absolute path
        self.docs_path = os.path.join(PROJECT_ROOT, docs_path)

        if not os.path.exists(self.docs_path):
            raise FileNotFoundError(f"Docs folder not found: {self.docs_path}")

        self.docs = []
        self.doc_ids = []

        self._load_docs()

        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(self.docs)

    def _load_docs(self):
        for fname in os.listdir(self.docs_path):
            fpath = os.path.join(self.docs_path, fname)
            if os.path.isfile(fpath) and fname.endswith(".md"):

                with open(fpath, "r", encoding="utf-8") as f:
                    text = f.read()

                paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

                self.docs.extend(paragraphs)
                self.doc_ids.extend([fname + f"::chunk{i}" for i in range(len(paragraphs))])

    def query(self, q: str, top_k: int = 3) -> List[Tuple[str, float]]:
        q_vec = self.vectorizer.transform([q])
        scores = (self.doc_vectors * q_vec.T).toarray().flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], scores[i]) for i in top_idx]

# Example usage
if __name__ == "__main__":
    retriever = DocRetriever(r"C:\Users\sondo\OneDrive - Faculty of Computer and Information Sciences (Ain Shams University)\Desktop\your_project\docs")
    results = retriever.query("Return days for unopened Beverages")
    print("Top results:", results)
