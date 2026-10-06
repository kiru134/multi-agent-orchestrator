"""A tiny in-memory document store shared by all the mock MCP servers.

Collections are plain dicts of id -> document. Everything is async (even though the
in-memory ops are instant) so the store's call shape matches a real database and a later
swap doesn't change any caller. Reads return deep copies so a caller mutating the dict it
got back can never corrupt the store.
"""

from __future__ import annotations

import asyncio
import copy
import itertools
import json
from pathlib import Path
from typing import Any, Callable

Document = dict[str, Any]


class DocumentNotFound(KeyError):
    pass


class DocumentStore:
    def __init__(self) -> None:
        self._collections: dict[str, dict[str, Document]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._counters: dict[str, itertools.count] = {}
        self._global_lock = asyncio.Lock()

    async def _lock_for(self, collection: str) -> asyncio.Lock:
        async with self._global_lock:
            if collection not in self._locks:
                self._locks[collection] = asyncio.Lock()
            return self._locks[collection]

    def _coll(self, collection: str) -> dict[str, Document]:
        return self._collections.setdefault(collection, {})

    # ---- seeding / reset -------------------------------------------------

    def seed(self, seed_dir: Path) -> None:
        """Synchronous on purpose: called once at startup / test setup, before any
        concurrent access is possible."""
        self._collections.clear()
        self._locks.clear()
        self._counters.clear()
        for f in sorted(seed_dir.glob("*.json")):
            collection = f.stem
            docs = json.loads(f.read_text(encoding="utf-8"))
            self._collections[collection] = {d["id"]: copy.deepcopy(d) for d in docs}

    def reset(self, seed_dir: Path) -> None:
        self.seed(seed_dir)

    def snapshot(self) -> dict[str, dict[str, Document]]:
        return copy.deepcopy(self._collections)

    # ---- id generation -----------------------------------------------

    def next_id(self, collection: str, prefix: str, start: int = 1000) -> str:
        if collection not in self._counters:
            # Lazily seed the counter above any pre-existing id with this prefix (seed data
            # uses the same "PREFIX-N" scheme), so generated ids never collide with it.
            highest = start - 1
            for doc_id in self._collections.get(collection, {}):
                suffix = doc_id.removeprefix(f"{prefix}-")
                if suffix != doc_id and suffix.isdigit():
                    highest = max(highest, int(suffix))
            self._counters[collection] = itertools.count(highest + 1)
        return f"{prefix}-{next(self._counters[collection])}"

    # ---- CRUD --------------------------------------------------------

    async def get(self, collection: str, doc_id: str) -> Document | None:
        lock = await self._lock_for(collection)
        async with lock:
            doc = self._coll(collection).get(doc_id)
            return copy.deepcopy(doc) if doc is not None else None

    async def find(
        self, collection: str, predicate: Callable[[Document], bool] | None = None
    ) -> list[Document]:
        lock = await self._lock_for(collection)
        async with lock:
            docs = list(self._coll(collection).values())
        docs = [copy.deepcopy(d) for d in docs if predicate is None or predicate(d)]
        return docs

    async def insert(self, collection: str, doc: Document) -> Document:
        if "id" not in doc:
            raise ValueError("document must have an 'id' field")
        lock = await self._lock_for(collection)
        async with lock:
            coll = self._coll(collection)
            if doc["id"] in coll:
                raise ValueError(f"duplicate id {doc['id']!r} in collection {collection!r}")
            coll[doc["id"]] = copy.deepcopy(doc)
            return copy.deepcopy(coll[doc["id"]])

    async def update(self, collection: str, doc_id: str, patch: dict[str, Any]) -> Document:
        lock = await self._lock_for(collection)
        async with lock:
            coll = self._coll(collection)
            if doc_id not in coll:
                raise DocumentNotFound(f"{collection}/{doc_id}")
            coll[doc_id] = {**coll[doc_id], **patch}
            return copy.deepcopy(coll[doc_id])

    async def delete(self, collection: str, doc_id: str) -> bool:
        lock = await self._lock_for(collection)
        async with lock:
            return self._coll(collection).pop(doc_id, None) is not None
