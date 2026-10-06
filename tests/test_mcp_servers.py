from mas.access.models import Caller

ALICE = Caller(user_id="alice", team="platform", roles=("engineer",))
CAROL = Caller(user_id="carol", team="sales", roles=("employee",))
DAVE = Caller(user_id="dave", team="platform", roles=("manager",))


async def test_tool_count_and_names(mcp):
    infra_tools = {t.tool_name for t in mcp.tools_for_server("infra")}
    assert infra_tools == {"get_service_status", "list_incidents", "restart_service", "scale_deployment"}
    assert mcp.tool_count() == 11


async def test_restart_service_updates_store_and_logs_incident(mcp, store):
    before = await store.get("services", "payments")
    assert before["status"] == "degraded"
    result = await mcp.call_tool("infra.restart_service", {"service": "payments"}, ALICE)
    assert not result.is_error
    assert result.structured_content["status_after"] == "healthy"
    after = await store.get("services", "payments")
    assert after["status"] == "healthy"
    incidents = await store.find("incidents", lambda i: i["service_id"] == "payments" and "restart" in i["message"].lower())
    assert len(incidents) == 1


async def test_list_my_tickets_is_scoped_to_caller(mcp):
    carol_result = await mcp.call_tool("it.list_my_tickets", {}, CAROL)
    ids = {t["id"] for t in carol_result.structured_content["tickets"]}
    assert ids == {"TKT-1001"}
    assert "TKT-1000" not in ids  # alice's ticket must never leak to carol


async def test_create_ticket_sets_requester_from_caller_not_args(mcp):
    result = await mcp.call_tool("it.create_ticket", {"title": "t", "description": "d"}, CAROL)
    assert result.structured_content["requester"] == "carol"


async def test_check_quota_denies_cross_team_for_non_manager(mcp):
    result = await mcp.call_tool("resources.check_quota", {"team": "platform"}, CAROL)
    assert "error" in result.structured_content


async def test_check_quota_allows_manager_cross_team(mcp):
    result = await mcp.call_tool("resources.check_quota", {"team": "sales"}, DAVE)
    assert result.structured_content["team"] == "sales"


async def test_request_vm_decrements_quota_and_records_request(mcp, store):
    before = await store.get("quotas", "platform")
    result = await mcp.call_tool("resources.request_vm", {"count": 1, "purpose": "testing"}, ALICE)
    assert result.structured_content["vm_used"] == before["vm_used"] + 1
    after = await store.get("quotas", "platform")
    assert after["vm_used"] == before["vm_used"] + 1
    requests = await store.find("vm_requests", lambda r: r["requester"] == "alice")
    assert len(requests) == 1


async def test_request_laptop_out_of_stock(mcp, store):
    await store.update("inventory", "ThinkPad X1", {"stock": 0})
    result = await mcp.call_tool("resources.request_laptop", {"model": "ThinkPad X1"}, CAROL)
    assert "error" in result.structured_content
