"""
Document ingestion pipeline with production safeguards.

Flow:
1. Validate file (size, type, content)
2. Extract text (PDF, DOCX, TXT, MD, etc.)
3. Chunk with RecursiveCharacterTextSplitter
4. Enrich metadata (source, thread_id, page numbers)
5. Store in ChromaDB vector store
6. Invalidate BM25 cache for hybrid search
"""

import logging
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.rag.embeddings import get_vectorstore

logger = logging.getLogger(__name__)
settings = get_settings()

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def validate_file(file_path: str) -> tuple[bool, str]:
    """Validate file before processing. Returns (is_valid, error_message)."""
    path = Path(file_path)

    if not path.exists():
        return False, f"File not found: {file_path}"
    if not path.is_file():
        return False, f"Not a file: {file_path}"

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return False, (
            f"Unsupported file format '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        return False, f"File too large: {size_mb:.1f}MB. Maximum: {settings.MAX_UPLOAD_SIZE_MB}MB"

    return True, ""


def read_file_text(file_path: str) -> str:
    """Extract text from various file formats."""
    path = Path(file_path)
    ext = path.suffix.lower()

    try:
        if ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text_parts = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"[Page {i+1}]\n{page_text}")
            return "\n\n".join(text_parts)

        elif ext == ".docx":
            import docx2txt
            return docx2txt.process(file_path)

        elif ext in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="ignore")

        else:
            raise ValueError(f"Unsupported file format: {ext}")

    except Exception as e:
        logger.error(f"Failed to extract text from {file_path}: {e}")
        raise ValueError(f"Could not extract text from {path.name}: {e}")


def add_document_to_rag(
    file_path: str,
    thread_id: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> dict:
    """
    Ingest a document into the RAG vector store.

    Steps:
    1. Validate file
    2. Extract text
    3. Chunk into manageable pieces
    4. Create Document objects with metadata
    5. Store embeddings in ChromaDB
    6. Invalidate BM25 cache for hybrid search
    """
    # Step 1: Validate
    is_valid, error_msg = validate_file(file_path)
    if not is_valid:
        raise ValueError(error_msg)

    # Step 2: Extract text
    text = read_file_text(file_path)

    if not text.strip():
        raise ValueError(
            f"No text could be extracted from {Path(file_path).name}. "
            "The file may be empty, image-based, or corrupted."
        )

    # Step 3: Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.RAG_CHUNK_SIZE,
        chunk_overlap=chunk_overlap or settings.RAG_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len,
    )
    chunks = splitter.split_text(text)

    if not chunks:
        raise ValueError("Text was extracted but no chunks were generated.")

    # Step 4: Create Document objects with rich metadata
    filename = Path(file_path).name
    docs = [
        Document(
            page_content=chunk,
            metadata={
                "thread_id": thread_id,
                "source": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
            },
        )
        for i, chunk in enumerate(chunks)
    ]

    # Step 5: Store in vector store
    vectorstore = get_vectorstore()
    vectorstore.add_documents(docs)

    # Step 6: Invalidate BM25 cache so hybrid search picks up new docs
    from app.rag.retriever import invalidate_bm25_cache
    invalidate_bm25_cache(thread_id)

    logger.info(
        f"Ingested document: file={filename}, chunks={len(chunks)}, "
        f"thread={thread_id}"
    )

    return {
        "filename": filename,
        "chunks_count": len(chunks),
        "thread_id": thread_id,
        "status": "success",
    }
