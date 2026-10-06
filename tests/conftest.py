from __future__ import annotations

import pytest
import pytest_asyncio
from langgraph.checkpoint.memory import InMemorySaver

from mas.access.policy import PolicyEngine
from mas.agents.registry import load_agent_definitions
from mas.mcp.client_manager import MCPClientManager
from mas.paths import SEED_DIR
from mas.store.document_store import DocumentStore


@pytest.fixture
def store() -> DocumentStore:
    s = DocumentStore()
    s.seed(SEED_DIR)
    return s


@pytest.fixture
def policy() -> PolicyEngine:
    return PolicyEngine()


@pytest.fixture
def agent_defs():
    return load_agent_definitions()


@pytest.fixture
def checkpointer() -> InMemorySaver:
    return InMemorySaver()


@pytest_asyncio.fixture
async def mcp(store: DocumentStore, agent_defs):
    manager = MCPClientManager()
    await manager.connect_all(agent_defs, store)
    # Deliberately not closing: the in-memory MCP transport runs its server in an anyio
    # task group tied to the task that opened it, and pytest-asyncio's per-test event loop
    # teardown doesn't guarantee fixture teardown runs in that same task, which trips
    # anyio's "exit cancel scope in a different task" check. These are in-memory mock
    # servers with no real resources to release, so skipping close() is safe for tests.
    return manager
