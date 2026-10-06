import pytest

from mas.store.document_store import DocumentNotFound


async def test_seed_loads_every_collection(store):
    snap = store.snapshot()
    assert set(snap.keys()) >= {
        "employees",
        "services",
        "incidents",
        "tickets",
        "assets",
        "quotas",
        "vm_requests",
        "inventory",
    }
    assert snap["services"]["payments"]["status"] == "degraded"


async def test_get_returns_deep_copy(store):
    doc = await store.get("services", "payments")
    doc["status"] = "mutated"
    doc2 = await store.get("services", "payments")
    assert doc2["status"] == "degraded"  # mutating the returned dict never touches the store


async def test_find_with_predicate(store):
    carols_tickets = await store.find("tickets", lambda t: t["requester"] == "carol")
    assert {t["id"] for t in carols_tickets} == {"TKT-1001"}


async def test_insert_and_duplicate_id_rejected(store):
    await store.insert("tickets", {"id": "TKT-9999", "requester": "alice", "title": "x", "status": "open"})
    assert await store.get("tickets", "TKT-9999") is not None
    with pytest.raises(ValueError):
        await store.insert("tickets", {"id": "TKT-9999", "requester": "alice", "title": "y", "status": "open"})


async def test_update_missing_doc_raises(store):
    with pytest.raises(DocumentNotFound):
        await store.update("tickets", "TKT-DOES-NOT-EXIST", {"status": "closed"})


async def test_next_id_increments_and_is_unique(store):
    ids = {store.next_id("tickets", "TKT") for _ in range(5)}
    assert len(ids) == 5


async def test_delete(store):
    await store.insert("tickets", {"id": "TKT-DEL", "requester": "alice", "title": "x", "status": "open"})
    assert await store.delete("tickets", "TKT-DEL") is True
    assert await store.get("tickets", "TKT-DEL") is None
    assert await store.delete("tickets", "TKT-DEL") is False
