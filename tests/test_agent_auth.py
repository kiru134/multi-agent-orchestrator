from mas.auth.agent_auth import AgentSessionRegistry, agent_secret, compute_response, ensure_authenticated, run_handshake


def test_correct_secret_authenticates():
    registry = AgentSessionRegistry()
    session = run_handshake(registry, "infra", agent_secret("infra"))
    assert session is not None
    assert registry.is_authenticated("infra")
    assert registry.valid_token("infra") == session.token


def test_wrong_secret_excludes_agent():
    registry = AgentSessionRegistry()
    session = run_handshake(registry, "infra", "totally-wrong-secret")
    assert session is None
    assert not registry.is_authenticated("infra")
    assert registry.valid_token("infra") is None


def test_nonce_is_single_use():
    registry = AgentSessionRegistry()
    nonce = registry.issue_challenge("infra")
    good_response = compute_response("infra", nonce, agent_secret("infra"))
    assert registry.authenticate("infra", good_response) is not None
    # Replaying the same (now-consumed) nonce's response must fail - there's no challenge
    # pending anymore.
    assert registry.authenticate("infra", good_response) is None


def test_ensure_authenticated_reuses_valid_session():
    registry = AgentSessionRegistry()
    run_handshake(registry, "infra", agent_secret("infra"))
    token1 = ensure_authenticated(registry, "infra")
    token2 = ensure_authenticated(registry, "infra")
    assert token1 == token2  # no re-handshake needed while still valid


def test_unauthenticated_agent_not_in_authenticated_list():
    registry = AgentSessionRegistry()
    run_handshake(registry, "infra", agent_secret("infra"))
    run_handshake(registry, "it_support", "bad-secret")
    assert registry.authenticated_agent_ids() == ["infra"]
