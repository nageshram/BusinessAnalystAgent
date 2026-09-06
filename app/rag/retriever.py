"""
Hybrid Retrieval: Vector Search (semantic) + BM25 (keyword) + Re-ranking.

WHY hybrid:
- Vector search: Understands synonyms and concepts ("revenue" matches "income")
  but misses exact terms (product IDs, KPI names, column headers)
- BM25: Matches exact keywords perfectly but misses semantic similarity
- Hybrid: Best of both worlds via Reciprocal Rank Fusion (RRF)

WHY re-ranking:
- Initial retrieval casts a wide net (top K from each method)
- RRF combines and re-ranks based on position in both lists
- Documents ranked high in BOTH lists get boosted to the top

Pipeline:
  Query → ┬→ ChromaDB (vector) ──→ Top K results ─┬→ RRF Re-ranking → Final top N
          └→ BM25 (keyword) ──────→ Top K results ─┘
"""

import logging
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from app.config import get_settings
from app.rag.embeddings import get_vectorstore

logger = logging.getLogger(__name__)
settings = get_settings()

# BM25 index cache per thread_id
_bm25_cache: dict[str, dict] = {}


def _build_bm25_index(thread_id: str) -> tuple[Optional[BM25Okapi], list[Document]]:
    """
    Build a BM25 index from all documents stored for a thread.
    Cached to avoid rebuilding on every query.
    """
    if thread_id in _bm25_cache:
        cached = _bm25_cache[thread_id]
        return cached["index"], cached["docs"]

    try:
        vectorstore = get_vectorstore()
        collection = vectorstore._collection

        results = collection.get(
            where={"thread_id": thread_id},
            include=["documents", "metadatas"],
        )

        if not results or not results.get("documents"):
            return None, []

        docs = []
        for i, content in enumerate(results["documents"]):
            meta = results["metadatas"][i] if results.get("metadatas") else {}
            docs.append(Document(page_content=content, metadata=meta or {}))

        tokenized = [doc.page_content.lower().split() for doc in docs]
        bm25_index = BM25Okapi(tokenized)

        _bm25_cache[thread_id] = {"index": bm25_index, "docs": docs}
        logger.debug(f"Built BM25 index: thread={thread_id}, docs={len(docs)}")
        return bm25_index, docs

    except Exception as e:
        logger.error(f"Failed to build BM25 index: {e}")
        return None, []


def invalidate_bm25_cache(thread_id: str) -> None:
    """Invalidate BM25 cache when new documents are added."""
    _bm25_cache.pop(thread_id, None)
    logger.debug(f"BM25 cache invalidated for thread: {thread_id}")


def _reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    """
    Reciprocal Rank Fusion — combines multiple ranked lists.

    score(d) = Σ 1 / (k + rank_i(d))
    k=60 is the standard constant from the RRF paper (Cormack et al. 2009).
    """
    doc_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            doc_key = doc.page_content[:200]

            if doc_key not in doc_map:
                doc_map[doc_key] = doc
                doc_scores[doc_key] = 0.0

            doc_scores[doc_key] += 1.0 / (k + rank)

    sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)

    results = []
    for key in sorted_keys:
        doc = doc_map[key]
        doc.metadata["rrf_score"] = round(doc_scores[key], 6)
        results.append(doc)

    return results


def rag_retriever(
    query: str,
    thread_id: str,
    top_k: Optional[int] = None,
) -> str:
    """
    Retrieve relevant documents using hybrid search:
    1. Vector similarity search (semantic)
    2. BM25 keyword search (exact matching)
    3. RRF re-ranking (combines both)
    """
    top_k = top_k or settings.RAG_TOP_K

    if not query.strip():
        return "no relevant document content found"

    try:
        # Vector search
        vectorstore = get_vectorstore()
        vector_results = vectorstore.similarity_search(
            query, k=top_k, filter={"thread_id": thread_id},
        )

        # BM25 search
        bm25_index, bm25_docs = _build_bm25_index(thread_id)
        bm25_results = []

        if bm25_index and bm25_docs:
            tokenized_query = query.lower().split()
            scores = bm25_index.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:top_k]
            bm25_results = [bm25_docs[i] for i in top_indices if scores[i] > 0]

        # RRF re-ranking
        if vector_results and bm25_results:
            fused = _reciprocal_rank_fusion([vector_results, bm25_results], k=60)
            final_docs = fused[:top_k]
        elif vector_results:
            final_docs = vector_results
        elif bm25_results:
            final_docs = bm25_results[:top_k]
        else:
            return "no relevant document content found"

        # Format results
        results = []
        for doc in final_docs:
            source = doc.metadata.get("source", "unknown")
            chunk_idx = doc.metadata.get("chunk_index", "?")
            total = doc.metadata.get("total_chunks", "?")
            rrf = doc.metadata.get("rrf_score", "")
            score_str = f", RRF={rrf}" if rrf else ""
            results.append(
                f"[Source: {source}, Chunk {chunk_idx}/{total}{score_str}]\n{doc.page_content}"
            )

        return "\n\n---\n\n".join(results)

    except Exception as e:
        logger.error(f"RAG retrieval failed: {e}", exc_info=True)
        return f"Error searching documents: {str(e)[:200]}"


def get_document_stats(thread_id: str) -> dict:
    """Get statistics about documents stored for a thread."""
    try:
        vectorstore = get_vectorstore()
        collection = vectorstore._collection

        results = collection.get(where={"thread_id": thread_id}, include=[])
        doc_count = len(results.get("ids", []))

        if doc_count > 0:
            meta_results = collection.get(
                where={"thread_id": thread_id}, include=["metadatas"],
            )
            sources = set()
            for meta in meta_results.get("metadatas", []):
                if meta and "source" in meta:
                    sources.add(meta["source"])
        else:
            sources = set()

        return {
            "thread_id": thread_id,
            "total_chunks": doc_count,
            "unique_documents": len(sources),
            "document_names": sorted(sources),
        }

    except Exception as e:
        logger.error(f"Failed to get document stats: {e}")
        return {
            "thread_id": thread_id,
            "total_chunks": 0,
            "unique_documents": 0,
            "document_names": [],
            "error": str(e),
        }
