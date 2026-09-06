"""
SQLAlchemy ORM Models — Production Schema Design.

Design Decisions:
1. ForeignKey constraints enforce referential integrity (no orphan messages)
2. CASCADE deletes clean up child records automatically
3. Indexes on frequently queried columns (thread_id, created_at)
4. UUID thread_ids for distributed safety
5. Explicit column types with NOT NULL constraints
6. Timestamps use timezone-aware UTC
7. Dataset and DatasetColumn models for uploaded business data
8. AnalysisResult stores cached analysis outputs per dataset
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    DateTime,
    ForeignKey,
    Index,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from app.db.database import Base


def utcnow():
    """Timezone-aware UTC timestamp factory."""
    return datetime.now(timezone.utc)


class Conversation(Base):
    """
    A conversation thread. Each thread_id groups related messages.

    One conversation has many ChatMessages.
    Deleting a conversation cascades to all its children.
    """
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    thread_id = Column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
        comment="Unique conversation identifier (UUID)"
    )
    title = Column(String(255), default="New Chat", nullable=False)
    model_name = Column(
        String(64),
        default="openai/gpt-4o-mini",
        nullable=False,
        comment="LLM model used for this conversation"
    )
    message_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Denormalized count for fast listing queries"
    )
    is_archived = Column(Integer, default=0, nullable=False, comment="Soft delete flag")
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
        nullable=False,
    )

    # Relationships with cascade delete
    messages = relationship(
        "ChatMessage",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
        lazy="dynamic",
    )

    __table_args__ = (
        Index("ix_conversations_active_updated", "is_archived", "updated_at"),
    )

    def __repr__(self):
        return f"<Conversation thread={self.thread_id} title={self.title!r}>"


class ChatMessage(Base):
    """
    A single message in a conversation.
    Stores both human and AI messages with their roles.
    """
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    thread_id = Column(
        String(64),
        ForeignKey("conversations.thread_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(
        String(20),
        nullable=False,
        comment="Message role: 'human', 'ai', 'system', 'tool'"
    )
    content = Column(Text, nullable=False, comment="Message content")
    tool_calls = Column(Text, nullable=True, comment="JSON-serialized tool calls if any")
    token_count = Column(
        Integer,
        nullable=True,
        comment="Approximate token count for monitoring"
    )
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    # Relationship back to parent
    conversation = relationship("Conversation", back_populates="messages")

    __table_args__ = (
        Index("ix_chat_messages_thread_time", "thread_id", "created_at"),
    )

    def __repr__(self):
        preview = self.content[:50] if self.content else ""
        return f"<ChatMessage role={self.role} thread={self.thread_id} content={preview!r}>"


class Dataset(Base):
    """
    An uploaded business dataset (CSV, XLSX, JSON).

    Stores metadata about the file — actual data lives on disk as files.
    Analysis results reference this via dataset_id.
    """
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
        comment="Unique dataset identifier (UUID)"
    )
    name = Column(String(255), nullable=False, comment="Original filename or user-provided name")
    file_path = Column(String(512), nullable=False, comment="Path to stored file on disk")
    file_type = Column(String(20), nullable=False, comment="csv, xlsx, json")
    row_count = Column(Integer, nullable=True, comment="Number of rows in the dataset")
    column_count = Column(Integer, nullable=True, comment="Number of columns")
    file_size_bytes = Column(Integer, nullable=True, comment="File size in bytes")
    thread_id = Column(
        String(64),
        nullable=True,
        index=True,
        comment="Associated conversation thread"
    )
    status = Column(
        String(20),
        default="uploaded",
        nullable=False,
        comment="Status: 'uploaded', 'profiled', 'analyzed', 'error'"
    )
    profile_json = Column(Text, nullable=True, comment="JSON-serialized dataset profile")
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
        nullable=False,
    )

    # Relationships
    columns = relationship(
        "DatasetColumn",
        back_populates="dataset",
        cascade="all, delete-orphan",
        order_by="DatasetColumn.column_index",
    )
    analysis_results = relationship(
        "AnalysisResult",
        back_populates="dataset",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<Dataset name={self.name!r} rows={self.row_count} cols={self.column_count}>"


class DatasetColumn(Base):
    """
    Per-column metadata for an uploaded dataset.

    Stores dtype, null counts, semantic classification,
    and basic statistics for fast tool access.
    """
    __tablename__ = "dataset_columns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(
        String(64),
        ForeignKey("datasets.dataset_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    column_name = Column(String(255), nullable=False, comment="Column header name")
    column_index = Column(Integer, nullable=False, comment="Column position (0-indexed)")
    dtype = Column(String(50), nullable=False, comment="Pandas dtype: int64, float64, object, datetime64, etc.")
    null_count = Column(Integer, default=0, comment="Number of null/NaN values")
    null_percentage = Column(Float, default=0.0, comment="Null percentage")
    unique_count = Column(Integer, nullable=True, comment="Number of unique values")
    semantic_type = Column(
        String(50),
        nullable=True,
        comment="Semantic classification: 'measure', 'dimension', 'temporal', 'identifier', 'text'"
    )
    stats_json = Column(
        Text,
        nullable=True,
        comment="JSON stats: mean, median, std, min, max for numeric; top values for categorical"
    )

    # Relationship back to parent
    dataset = relationship("Dataset", back_populates="columns")

    __table_args__ = (
        UniqueConstraint("dataset_id", "column_name", name="uq_dataset_column"),
        Index("ix_dataset_columns_dataset_id", "dataset_id"),
    )

    def __repr__(self):
        return f"<DatasetColumn name={self.column_name!r} dtype={self.dtype}>"


class AnalysisResult(Base):
    """
    Cached analysis output for a dataset.

    Each analysis type (profiling, metrics, trends, anomalies, correlations)
    is stored as a separate row with analysis_type key.
    Results are JSON-serialized for flexibility.
    """
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(
        String(64),
        ForeignKey("datasets.dataset_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    analysis_type = Column(
        String(50),
        nullable=False,
        comment="Type: 'profile', 'metrics', 'trends', 'anomalies', 'correlations', 'charts'"
    )
    result_json = Column(Text, nullable=False, comment="JSON-serialized analysis output")
    parameters_json = Column(Text, nullable=True, comment="JSON parameters used for this analysis")
    status = Column(
        String(20),
        default="completed",
        nullable=False,
        comment="Status: 'running', 'completed', 'error'"
    )
    error_message = Column(Text, nullable=True, comment="Error details if status='error'")
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    # Relationship back to parent
    dataset = relationship("Dataset", back_populates="analysis_results")

    __table_args__ = (
        Index("ix_analysis_dataset_type", "dataset_id", "analysis_type"),
    )

    def __repr__(self):
        return f"<AnalysisResult dataset={self.dataset_id} type={self.analysis_type}>"
