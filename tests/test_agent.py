"""Tests for the agent tools and graph."""

import pytest
from app.agent.state import AgentState, trim_state_messages, create_initial_state
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class TestAgentState:
    def test_create_initial_state(self):
        state = create_initial_state(thread_id="test-123")
        assert state["thread_id"] == "test-123"
        assert state["step_count"] == 0
        assert state["messages"] == []

    def test_trim_messages_under_limit(self):
        messages = [HumanMessage(content=f"msg {i}") for i in range(5)]
        result = trim_state_messages(messages, max_messages=20)
        assert len(result) == 5

    def test_trim_messages_over_limit(self):
        messages = [HumanMessage(content=f"msg {i}") for i in range(30)]
        result = trim_state_messages(messages, max_messages=20)
        assert len(result) <= 20

    def test_trim_preserves_system_message(self):
        messages = [SystemMessage(content="system")] + [
            HumanMessage(content=f"msg {i}") for i in range(25)
        ]
        result = trim_state_messages(messages, max_messages=10)
        # Should keep the system message
        assert any(hasattr(m, 'type') and m.type == 'system' for m in result)


class TestToolsList:
    def test_tools_exist(self):
        from app.agent.tools import get_tools
        tools = get_tools()
        assert len(tools) == 8
        tool_names = [t.name for t in tools]
        assert "profile_dataset" in tool_names
        assert "query_data" in tool_names
        assert "compute_metrics" in tool_names
        assert "detect_trends" in tool_names
        assert "detect_anomalies" in tool_names
        assert "compute_correlations" in tool_names
        assert "suggest_charts" in tool_names
        assert "search_knowledge_base" in tool_names
