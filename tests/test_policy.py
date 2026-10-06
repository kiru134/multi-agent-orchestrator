def test_engineer_can_use_infra_not_it(policy):
    caller = policy.get_caller("alice")
    assert policy.can_use_agent(caller, "infra").allow
    assert not policy.can_use_agent(caller, "it_support").allow


def test_engineer_can_read_but_not_restart_infra(policy):
    caller = policy.get_caller("alice")
    assert policy.can_call_tool(caller, "infra.get_service_status").allow
    assert policy.can_call_tool(caller, "infra.scale_deployment").allow
    assert not policy.can_call_tool(caller, "infra.restart_service").allow


def test_it_admin_wildcard_grants_all_it_tools(policy):
    caller = policy.get_caller("bob")
    assert policy.can_call_tool(caller, "it.create_ticket").allow
    assert policy.can_call_tool(caller, "it.reset_password").allow
    assert policy.can_call_tool(caller, "infra.restart_service").allow
    assert not policy.can_call_tool(caller, "infra.scale_deployment").allow


def test_employee_scoped_to_self_service_tools(policy):
    caller = policy.get_caller("carol")
    assert policy.can_use_agent(caller, "it_support").allow
    assert policy.can_use_agent(caller, "resource_alloc").allow
    assert not policy.can_use_agent(caller, "infra").allow
    assert policy.can_call_tool(caller, "resources.request_vm").allow
    assert not policy.can_call_tool(caller, "infra.get_service_status").allow


def test_manager_wildcard_grants_everything(policy):
    caller = policy.get_caller("dave")
    for agent in ("infra", "it_support", "resource_alloc"):
        assert policy.can_use_agent(caller, agent).allow
    for tool in ("infra.restart_service", "it.reset_password", "resources.request_vm"):
        assert policy.can_call_tool(caller, tool).allow


def test_audit_log_records_every_decision(policy):
    caller = policy.get_caller("carol")
    policy.can_use_agent(caller, "infra")
    policy.can_call_tool(caller, "it.create_ticket")
    log = policy.audit_log()
    assert len(log) == 2
    assert log[0].allowed is False
    assert log[1].allowed is True
