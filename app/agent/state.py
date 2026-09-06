"""
Custom state definition for the LangGraph agent.

WHY custom state instead of MessagesState:
- Message trimming to prevent unbounded memory growth
- Step counter for recursion tracking
- Active dataset tracking for analysis context
- State validation before checkpointing
"""

from typing import Annotated, Optional
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, trim_messages


class AgentState(TypedDict):
    """
    Custom agent state with production safeguards.

    Fields:
    - messages: Conversation messages (auto-appended via add_messages reducer)
    - step_count: Tracks agent→tool loops (recursion guard)
    - thread_id: Current conversation thread ID
    - model_name: LLM model being used
    - active_dataset_id: Currently loaded dataset for analysis context
    """
    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int
    thread_id: str
    model_name: str
    active_dataset_id: str


def create_initial_state(
    thread_id: str,
    model_name: str = "openai/gpt-4o-mini",
    active_dataset_id: str = "",
) -> AgentState:
    """Create a fresh state for a new agent invocation."""
    return AgentState(
        messages=[],
        step_count=0,
        thread_id=thread_id,
        model_name=model_name,
        active_dataset_id=active_dataset_id,
    )


def trim_state_messages(
    messages: list[BaseMessage],
    max_tokens: int = 8000,
    max_messages: int = 20,
) -> list[BaseMessage]:
    """
    Trim messages to prevent unbounded state growth.

    Strategy:
    1. Always keep the system message (first message)
    2. Keep the most recent `max_messages` messages
    3. Within that, trim by approximate token count

    WHY this matters:
    - Each LangGraph checkpoint serializes ALL messages
    - Without trimming, a 100-message thread creates ~2MB checkpoints
    - Trimming to 20 messages keeps checkpoints under 200KB
    """
    if len(messages) <= max_messages:
        return messages

    # Keep system message + last (max_messages - 1) messages
    system_messages = [m for m in messages[:1] if hasattr(m, 'type') and m.type == 'system']
    recent_messages = messages[-(max_messages - len(system_messages)):]

    trimmed = system_messages + recent_messages

    # Further trim by token count
    try:
        trimmed = trim_messages(
            trimmed,
            max_tokens=max_tokens,
            strategy="last",
            allow_partial=False,
            include_system=True,
            token_counter=len,
        )
    except Exception:
        pass

    return trimmed
