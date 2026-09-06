"""
Production-safe LangGraph agent with:
- Recursion limit (max 15 steps)
- Message trimming (max 20 messages in state)
- Circuit breaker on LLM calls
- PostgreSQL checkpointer (via langgraph-checkpoint-postgres)
- Graceful error recovery
- Model caching

HOW THE GRAPH WORKS:

    ┌─────────┐
    │  START   │
    └────┬─────┘
         │
         ▼
    ┌──────────────┐
    │ trim_messages │ ◄── Prevents unbounded state growth
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   chatbot     │ ◄── LLM generates response (may include tool calls)
    └──────┬───────┘
           │
     ┌─────┴─────┐
     │ has tool   │
     │  calls?    │
     └─────┬─────┘
       yes │    │ no
           │    │
           ▼    ▼
    ┌──────────┐  ┌─────────┐
    │  tools   │  │  END    │
    └──────┬───┘  └─────────┘
           │
           ▼
    ┌──────────────┐
    │ step_guard   │ ◄── Checks if step_count >= 15, forces END if so
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   chatbot     │ ◄── Process tool results
    └──────────────┘
         ... (loop continues until no tool calls or step_guard stops it)
"""

import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from app.config import get_settings
from app.agent.state import AgentState, trim_state_messages
from app.agent.tools import get_tools
from app.agent.prompts import BUSINESS_ANALYST_SYSTEM_PROMPT
from app.circuit_breaker import llm_circuit_breaker, CircuitBreakerError

logger = logging.getLogger(__name__)
settings = get_settings()


def _get_checkpointer():
    """
    Get the PostgreSQL checkpointer for LangGraph state persistence.

    WHY PostgreSQL instead of SqliteSaver:
    - Concurrent access without file locks
    - Connection pooling for high throughput
    - Proper ACID transactions
    - Works in containerized/distributed environments
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        checkpointer = PostgresSaver.from_conn_string(settings.DATABASE_URL)
        checkpointer.setup()
        logger.info("PostgreSQL checkpointer initialized")
        return checkpointer
    except ImportError:
        logger.warning(
            "langgraph-checkpoint-postgres not installed. "
            "Falling back to in-memory checkpointer."
        )
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL checkpointer: {e}. Falling back to memory.")
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()


def build_agent(model_name: str = None) -> any:
    """
    Build a production-safe LangGraph business analyst agent.

    Safety features:
    1. recursion_limit=15 — hard stop on infinite loops
    2. Message trimming — keeps state under 200KB
    3. Circuit breaker — stops calling failing LLM API
    4. Step guard — custom node that counts iterations
    5. Error recovery — tool errors become messages, not crashes

    Uses OpenRouter (OpenAI-compatible API) for LLM access.
    """
    model_name = model_name or settings.OPENROUTER_MODEL

    if model_name not in settings.ALLOWED_MODELS:
        logger.warning(f"Model '{model_name}' not in allowed list, falling back to {settings.OPENROUTER_MODEL}")
        model_name = settings.OPENROUTER_MODEL

    # ── Initialize LLM with OpenRouter ──
    llm = ChatOpenAI(
        model=model_name,
        temperature=settings.LLM_TEMPERATURE,
        streaming=True,
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
        max_retries=2,
        timeout=settings.LLM_TIMEOUT,
        default_headers={
            "HTTP-Referer": "https://github.com/nageshram/BusinessAnalystAgent",
            "X-Title": "Business Analyst Agent",
        },
    )

    tools = get_tools()
    llm_with_tools = llm.bind_tools(tools)

    # ── Node: Trim Messages ──
    def trim_messages_node(state: AgentState) -> dict:
        """Trim messages BEFORE the LLM call to prevent checkpoint bloat."""
        trimmed = trim_state_messages(
            state["messages"],
            max_tokens=8000,
            max_messages=settings.AGENT_MAX_MESSAGES,
        )
        return {"messages": trimmed}

    # ── Node: Chatbot (LLM call) ──
    def chatbot_node(state: AgentState) -> dict:
        """
        Send messages to the LLM and return the response.
        Wrapped in circuit breaker to prevent cascading failures.
        """
        messages = [SystemMessage(content=BUSINESS_ANALYST_SYSTEM_PROMPT)] + state["messages"]

        try:
            def _call_llm():
                return llm_with_tools.invoke(messages)

            response = llm_circuit_breaker.call(_call_llm)
        except CircuitBreakerError as e:
            logger.error(f"LLM circuit breaker open: {e}")
            response = AIMessage(
                content=f"I'm temporarily unable to process your request. "
                f"The AI service is experiencing issues. Please try again in {e.time_remaining:.0f} seconds."
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            response = AIMessage(
                content="I encountered an error processing your request. Please try again."
            )

        new_step = state.get("step_count", 0) + 1

        return {
            "messages": [response],
            "step_count": new_step,
        }

    # ── Node: Step Guard ──
    def step_guard(state: AgentState) -> str:
        """
        Check if we've exceeded the maximum number of agent→tool loops.
        Returns "chatbot" to continue or "__end__" to force stop.
        """
        step_count = state.get("step_count", 0)
        max_steps = settings.AGENT_RECURSION_LIMIT

        if step_count >= max_steps:
            logger.warning(
                f"Step guard triggered: {step_count} >= {max_steps}. "
                f"Forcing agent to stop."
            )
            return END

        return "chatbot"

    # ── Build the Graph ──
    tools_list = get_tools()
    tool_node = ToolNode(tools_list)

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("trim_messages", trim_messages_node)
    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("tools", tool_node)

    # Add edges
    workflow.add_edge(START, "trim_messages")
    workflow.add_edge("trim_messages", "chatbot")

    # Conditional: chatbot → tools (if tool calls) or END
    workflow.add_conditional_edges(
        "chatbot",
        tools_condition,
    )

    # After tools execute, check step guard before looping back
    workflow.add_conditional_edges(
        "tools",
        step_guard,
        {END: END, "chatbot": "chatbot"},
    )

    # ── Compile with Checkpointer ──
    checkpointer = _get_checkpointer()

    compiled = workflow.compile(
        checkpointer=checkpointer,
    )

    logger.info(f"Agent built: model={model_name}, recursion_limit={settings.AGENT_RECURSION_LIMIT}")
    return compiled


# ──────────────────────────────────────────────
# Agent Cache (Singleton per model)
# ──────────────────────────────────────────────

_agent_cache: dict = {}


def get_agent(model_name: str = None) -> any:
    """
    Get or create a cached agent instance.
    Each model gets its own compiled graph.
    """
    model_name = model_name or settings.OPENROUTER_MODEL

    if model_name not in settings.ALLOWED_MODELS:
        model_name = settings.OPENROUTER_MODEL

    if model_name not in _agent_cache:
        _agent_cache[model_name] = build_agent(model_name)
        logger.info(f"Agent cached for model: {model_name}")

    return _agent_cache[model_name]


def clear_agent_cache():
    """Clear the agent cache (used in tests)."""
    _agent_cache.clear()
