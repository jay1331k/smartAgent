# tests/test_agent.py
import pytest
from agent.agent import Agent
from agent.node import Node
from agent.attention_mechanism import AttentionMechanism
from agent.memory import LocalMemory, GlobalMemory
from unittest.mock import MagicMock, patch  # Import mocking utilities


# --- Test Agent Initialization ---
def test_agent_init():
    """
    Test the initialization of the Agent class.
    Checks if AttentionMechanism, GlobalMemory, and LLM are initialized correctly.
    """
    mock_llm = MagicMock()  # Mock the LLM
    agent = Agent(llm=mock_llm)

    assert isinstance(agent.attention_mechanism, AttentionMechanism)
    assert isinstance(agent.global_memory, GlobalMemory)
    assert agent.llm == mock_llm
    assert agent.execution_count == 0

# --- Test create_root_node ---
def test_create_root_node():
    """
    Test the creation of the root node.
    Checks:
        - Node is created with correct attributes.
        - Node is added to node_lookup.
        - Initial constraints are added to the AttentionMechanism.
        - Dependencies are tracked correctly.
    """
    mock_llm = MagicMock()
    agent = Agent(llm=mock_llm)
    with patch('streamlit.session_state', MagicMock()):
        # Mock st.session_state to avoid errors
        task_description = "Test Task"
        initial_constraints = ["format: json", "max_length: 100"]

        root_node = agent.create_root_node(task_description, initial_constraints)

        assert isinstance(root_node, Node)
        assert root_node.parent_id is None
        assert root_node.task_description == task_description
        assert root_node.depth == 0
        assert root_node.node_id in agent.attention_mechanism.dependency_graph # Check if dependency is tracked
        assert root_node.node_id in st.session_state.node_lookup # Check if present in node_lookup
        assert st.session_state.attention_mechanism.get_constraints(root_node.node_id) == initial_constraints

# --- Test create_child_node ---
def test_create_child_node():
    """
    Test the creation of child nodes.
    Checks:
        - Child node is created with correct attributes.
        - Child node is added to node_lookup.
        - Parent node's child_ids are updated.
        - Dependencies are tracked correctly.
    """
    mock_llm = MagicMock()
    agent = Agent(llm=mock_llm)
    with patch('streamlit.session_state', MagicMock()):
        # Mock st.session_state
        st.session_state.node_lookup