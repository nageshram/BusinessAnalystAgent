"""
CRUD operations with retry, error handling, and pagination.

Every function follows the same pattern:
1. Open session via context manager
2. Execute query
3. Auto-commit on success, auto-rollback on error
4. Return result or raise with clear error message

For write operations, retry_operation wraps the call to handle
transient DB failures (connection drops, deadlocks).
"""

import json
import logging
from typing import Optional
from datetime import datetime, timezone

from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.db.database import get_db, retry_operation
from app.db.models import Conversation, ChatMessage, Dataset, DatasetColumn, AnalysisResult

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Conversation CRUD
# ──────────────────────────────────────────────

def create_conversation(thread_id: str, title: str = "New Chat", model_name: str = "openai/gpt-4o-mini") -> Conversation:
    """
    Create a new conversation. Retries on transient DB errors.
    Title is auto-truncated from the first user message.
    """
    def _create():
        with get_db() as db:
            conversation = Conversation(
                thread_id=thread_id,
                title=title[:255] if title else "New Chat",
                model_name=model_name,
            )
            db.add(conversation)
            db.flush()
            logger.info(f"Created conversation: thread={thread_id}, title={title!r}")
            return conversation

    return retry_operation(_create)


def get_or_create_conversation(
    thread_id: str,
    first_message: Optional[str] = None,
    model_name: str = "openai/gpt-4o-mini",
) -> Conversation:
    """
    Get existing conversation or create a new one.
    If creating, uses the first message (truncated to 60 chars) as the title.
    """
    def _get_or_create():
        with get_db() as db:
            conversation = (
                db.query(Conversation)
                .filter(Conversation.thread_id == thread_id)
                .first()
            )

            if conversation:
                conversation.updated_at = datetime.now(timezone.utc)
                return conversation

            # Create new conversation
            title = "New Chat"
            if first_message:
                title = first_message[:60]
                if len(first_message) > 60:
                    title += "..."

            conversation = Conversation(
                thread_id=thread_id,
                title=title,
                model_name=model_name,
            )
            db.add(conversation)
            db.flush()
            logger.info(f"Created conversation: thread={thread_id}, title={title!r}")
            return conversation

    return retry_operation(_get_or_create)


def list_conversations(
    page: int = 1,
    page_size: int = 20,
    include_archived: bool = False,
) -> dict:
    """
    List conversations with pagination, ordered by most recently updated.

    Returns:
        {
            "conversations": [...],
            "total": 42,
            "page": 1,
            "page_size": 20,
            "has_next": True
        }
    """
    with get_db() as db:
        query = db.query(Conversation)

        if not include_archived:
            query = query.filter(Conversation.is_archived == 0)

        total = query.count()
        conversations = (
            query
            .order_by(desc(Conversation.updated_at))
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )

        return {
            "conversations": conversations,
            "total": total,
            "page": page,
            "page_size": page_size,
            "has_next": (page * page_size) < total,
        }


def get_conversation(thread_id: str) -> Optional[Conversation]:
    """Get a single conversation by thread_id."""
    with get_db() as db:
        return (
            db.query(Conversation)
            .filter(Conversation.thread_id == thread_id)
            .first()
        )


def archive_conversation(thread_id: str) -> bool:
    """Soft-delete a conversation (set is_archived=1)."""
    def _archive():
        with get_db() as db:
            conv = (
                db.query(Conversation)
                .filter(Conversation.thread_id == thread_id)
                .first()
            )
            if conv:
                conv.is_archived = 1
                logger.info(f"Archived conversation: thread={thread_id}")
                return True
            return False

    return retry_operation(_archive)


# ──────────────────────────────────────────────
# Chat Message CRUD
# ──────────────────────────────────────────────

def save_chat_message(
    thread_id: str,
    role: str,
    content: str,
    tool_calls: Optional[str] = None,
    token_count: Optional[int] = None,
) -> ChatMessage:
    """
    Save a chat message and update the parent conversation's timestamp.
    Retries on transient DB errors.
    """
    def _save():
        with get_db() as db:
            msg = ChatMessage(
                thread_id=thread_id,
                role=role,
                content=content,
                tool_calls=tool_calls,
                token_count=token_count,
            )
            db.add(msg)

            # Update conversation timestamp and message count
            conversation = (
                db.query(Conversation)
                .filter(Conversation.thread_id == thread_id)
                .first()
            )
            if conversation:
                conversation.updated_at = datetime.now(timezone.utc)
                conversation.message_count = (conversation.message_count or 0) + 1

            db.flush()
            logger.debug(f"Saved message: thread={thread_id}, role={role}")
            return msg

    return retry_operation(_save)


def get_chat_history(
    thread_id: str,
    limit: int = 50,
    offset: int = 0,
) -> list[ChatMessage]:
    """
    Get chat history for a thread, ordered chronologically.
    Returns the most recent `limit` messages.
    """
    with get_db() as db:
        return (
            db.query(ChatMessage)
            .filter(ChatMessage.thread_id == thread_id)
            .order_by(ChatMessage.created_at.asc())
            .offset(offset)
            .limit(limit)
            .all()
        )


def get_recent_messages(thread_id: str, count: int = 20) -> list[ChatMessage]:
    """
    Get the N most recent messages for a thread.
    Used by the agent to build context window without loading entire history.
    """
    with get_db() as db:
        messages = (
            db.query(ChatMessage)
            .filter(ChatMessage.thread_id == thread_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(count)
            .all()
        )
        return list(reversed(messages))


# ──────────────────────────────────────────────
# Dataset CRUD
# ──────────────────────────────────────────────

def save_dataset(
    dataset_id: str,
    name: str,
    file_path: str,
    file_type: str,
    row_count: Optional[int] = None,
    column_count: Optional[int] = None,
    file_size_bytes: Optional[int] = None,
    thread_id: Optional[str] = None,
) -> Dataset:
    """Save a new dataset entry with retry."""
    def _save():
        with get_db() as db:
            dataset = Dataset(
                dataset_id=dataset_id,
                name=name,
                file_path=file_path,
                file_type=file_type,
                row_count=row_count,
                column_count=column_count,
                file_size_bytes=file_size_bytes,
                thread_id=thread_id,
                status="uploaded",
            )
            db.add(dataset)
            db.flush()
            logger.info(f"Saved dataset: id={dataset_id}, name={name!r}")
            return dataset

    return retry_operation(_save)


def get_dataset(dataset_id: str) -> Optional[Dataset]:
    """Get a dataset by its ID."""
    with get_db() as db:
        return (
            db.query(Dataset)
            .filter(Dataset.dataset_id == dataset_id)
            .first()
        )


def list_datasets(
    thread_id: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> dict:
    """List datasets with optional thread_id filter and pagination."""
    with get_db() as db:
        query = db.query(Dataset)

        if thread_id:
            query = query.filter(Dataset.thread_id == thread_id)

        total = query.count()
        datasets = (
            query
            .order_by(desc(Dataset.created_at))
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )

        return {
            "datasets": datasets,
            "total": total,
            "page": page,
            "page_size": page_size,
            "has_next": (page * page_size) < total,
        }


def update_dataset_profile(
    dataset_id: str,
    row_count: int,
    column_count: int,
    profile_json: str,
    status: str = "profiled",
) -> bool:
    """Update dataset with profiling results."""
    def _update():
        with get_db() as db:
            dataset = (
                db.query(Dataset)
                .filter(Dataset.dataset_id == dataset_id)
                .first()
            )
            if dataset:
                dataset.row_count = row_count
                dataset.column_count = column_count
                dataset.profile_json = profile_json
                dataset.status = status
                dataset.updated_at = datetime.now(timezone.utc)
                return True
            return False

    return retry_operation(_update)


def save_dataset_columns(dataset_id: str, columns: list[dict]) -> None:
    """Save column metadata for a dataset. Replaces existing columns."""
    def _save():
        with get_db() as db:
            # Remove existing columns
            db.query(DatasetColumn).filter(
                DatasetColumn.dataset_id == dataset_id
            ).delete()

            # Insert new columns
            for col_data in columns:
                col = DatasetColumn(
                    dataset_id=dataset_id,
                    column_name=col_data["column_name"],
                    column_index=col_data["column_index"],
                    dtype=col_data["dtype"],
                    null_count=col_data.get("null_count", 0),
                    null_percentage=col_data.get("null_percentage", 0.0),
                    unique_count=col_data.get("unique_count"),
                    semantic_type=col_data.get("semantic_type"),
                    stats_json=json.dumps(col_data.get("stats", {})),
                )
                db.add(col)

            db.flush()
            logger.info(f"Saved {len(columns)} columns for dataset: {dataset_id}")

    retry_operation(_save)


# ──────────────────────────────────────────────
# Analysis Result CRUD
# ──────────────────────────────────────────────

def save_analysis_result(
    dataset_id: str,
    analysis_type: str,
    result_json: str,
    parameters_json: Optional[str] = None,
    status: str = "completed",
    error_message: Optional[str] = None,
) -> AnalysisResult:
    """Save an analysis result with retry."""
    def _save():
        with get_db() as db:
            result = AnalysisResult(
                dataset_id=dataset_id,
                analysis_type=analysis_type,
                result_json=result_json,
                parameters_json=parameters_json,
                status=status,
                error_message=error_message,
            )
            db.add(result)
            db.flush()
            logger.info(f"Saved analysis: dataset={dataset_id}, type={analysis_type}")
            return result

    return retry_operation(_save)


def get_analysis_result(
    dataset_id: str,
    analysis_type: str,
) -> Optional[AnalysisResult]:
    """Get the most recent analysis result for a dataset and type."""
    with get_db() as db:
        return (
            db.query(AnalysisResult)
            .filter(
                AnalysisResult.dataset_id == dataset_id,
                AnalysisResult.analysis_type == analysis_type,
                AnalysisResult.status == "completed",
            )
            .order_by(desc(AnalysisResult.created_at))
            .first()
        )


def get_all_analysis_results(dataset_id: str) -> list[AnalysisResult]:
    """Get all analysis results for a dataset."""
    with get_db() as db:
        return (
            db.query(AnalysisResult)
            .filter(
                AnalysisResult.dataset_id == dataset_id,
                AnalysisResult.status == "completed",
            )
            .order_by(AnalysisResult.created_at.desc())
            .all()
        )
