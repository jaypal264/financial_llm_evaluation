"""
rag_pipeline.py
===============
Retrieval-Augmented Generation (RAG) pipeline for financial QA.

Steps:
  1. Load financial documents / contexts from the datasets
  2. Split documents into chunks
  3. Embed chunks using sentence-transformers (all-MiniLM-L6-v2)
  4. Store in FAISS index
  5. Retrieve top-k chunks per question
  6. Return augmented context for LLM inference

Also includes BM25 as a retrieval baseline.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Document chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """
    Splits text into overlapping chunks (word-based).

    Args:
        text: Input text string.
        chunk_size: Target chunk size in words.
        overlap: Overlap in words between consecutive chunks.

    Returns:
        List of text chunks.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


def build_corpus(
    datasets: dict,
    chunk_size: int = 400,
    overlap: int = 50,
) -> tuple[list[str], list[dict]]:
    """
    Builds a retrieval corpus from the contexts of all three datasets.

    IMPORTANT: Reference answers and test labels are NEVER included in the corpus.
    Only the 'context' column (provided document text) is used.

    Returns:
        (chunks, metadata) where:
          - chunks is a list of text strings
          - metadata is a list of dicts describing each chunk's origin
    """
    all_chunks = []
    all_meta = []

    for dataset_name, df in datasets.items():
        if df.empty:
            continue
        for _, row in df.iterrows():
            context = str(row.get("context", ""))
            if not context or len(context.strip()) < 20:
                continue

            text_chunks = chunk_text(context, chunk_size=chunk_size, overlap=overlap)
            for ci, chunk in enumerate(text_chunks):
                all_chunks.append(chunk)
                all_meta.append({
                    "dataset": dataset_name,
                    "question_id": str(row.get("question_id", "")),
                    "company": str(row.get("company", "")),
                    "document_name": str(row.get("document_name", "")),
                    "chunk_index": ci,
                })

    logger.info("Built corpus: %d chunks from %d datasets", len(all_chunks), len(datasets))
    return all_chunks, all_meta


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

class FAISSRetriever:
    """
    Dense retriever using sentence-transformers embeddings and FAISS index.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: Optional[Path] = None,
    ):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunks: list[str] = []
        self.metadata: list[dict] = []
        self.index_path = index_path or DATA_PROCESSED / "faiss_index"

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", self.model_name)
            self.model = SentenceTransformer(self.model_name)

    def build_index(
        self,
        chunks: list[str],
        metadata: list[dict],
        batch_size: int = 64,
        save: bool = True,
    ) -> None:
        """
        Encodes all chunks and builds a FAISS flat L2 index.
        """
        import faiss

        self._load_model()
        self.chunks = chunks
        self.metadata = metadata

        logger.info("Encoding %d chunks...", len(chunks))
        embeddings = self.model.encode(
            chunks,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        embeddings = embeddings.astype(np.float32)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine if normalized)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        logger.info("FAISS index built: %d vectors, dim=%d", self.index.ntotal, dim)

        if save:
            self.save(self.index_path)

    def save(self, path: Path) -> None:
        import faiss
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump({"chunks": self.chunks, "metadata": self.metadata}, f)
        logger.info("FAISS index saved to %s", path)

    def load(self, path: Optional[Path] = None) -> bool:
        import faiss
        path = Path(path or self.index_path)
        if not (path / "index.faiss").exists():
            return False
        self.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "chunks.pkl", "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.metadata = data["metadata"]
        logger.info("FAISS index loaded: %d vectors", self.index.ntotal)
        return True

    def retrieve(
        self, query: str, top_k: int = 5
    ) -> list[tuple[str, float, dict]]:
        """
        Retrieves top-k chunks for a query.

        Returns list of (chunk_text, score, metadata) tuples.
        """
        self._load_model()
        query_emb = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        import faiss
        faiss.normalize_L2(query_emb)
        scores, indices = self.index.search(query_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append((self.chunks[idx], float(score), self.metadata[idx]))
        return results

    def retrieve_context(self, query: str, top_k: int = 5, separator: str = "\n\n") -> str:
        """
        Convenience method: retrieves and joins top-k chunks into a single context string.
        """
        results = self.retrieve(query, top_k=top_k)
        return separator.join([chunk for chunk, _, _ in results])


# ---------------------------------------------------------------------------
# BM25 baseline
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    Sparse BM25 retriever using rank_bm25.
    """

    def __init__(self):
        self.bm25 = None
        self.chunks: list[str] = []
        self.metadata: list[dict] = []

    def build_index(self, chunks: list[str], metadata: list[dict]) -> None:
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        self.metadata = metadata
        tokenized = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built: %d documents", len(chunks))

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, float, dict]]:
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build_index() first.")
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(scores[idx]), self.metadata[idx]))
        return results

    def retrieve_context(self, query: str, top_k: int = 5, separator: str = "\n\n") -> str:
        results = self.retrieve(query, top_k=top_k)
        return separator.join([chunk for chunk, _, _ in results])


# ---------------------------------------------------------------------------
# RAG-augmented DataFrame builder
# ---------------------------------------------------------------------------

def build_rag_contexts(
    df: pd.DataFrame,
    retriever,
    top_k: int = 5,
    question_col: str = "question",
    fallback_to_original: bool = True,
) -> pd.DataFrame:
    """
    Adds a 'rag_context' column to df by retrieving top-k chunks for each question.

    If retrieval returns nothing and fallback_to_original=True,
    uses the original 'context' column instead.

    Does NOT include reference_answer in retrieval corpus.
    """
    rag_contexts = []
    for _, row in df.iterrows():
        query = str(row.get(question_col, ""))
        try:
            retrieved = retriever.retrieve_context(query, top_k=top_k)
        except Exception as e:
            logger.warning("Retrieval failed for qid=%s: %s", row.get("question_id"), e)
            retrieved = ""

        if not retrieved.strip() and fallback_to_original:
            retrieved = str(row.get("context", ""))

        rag_contexts.append(retrieved)

    df = df.copy()
    df["rag_context"] = rag_contexts
    return df

