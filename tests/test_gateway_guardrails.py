"""Gateway guardrail tests with downstream services mocked via httpx.MockTransport."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from common.privacy import PRIVACY_REFUSAL
from mcp_server import server


def _downstream(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/cohort/risk":
        filters = json.loads(request.content)["filters"]
        n = 3 if filters.get("job_role") == "Research Director" else 120
        if n < 5:
            return httpx.Response(400, json={"detail": PRIVACY_REFUSAL})
        return httpx.Response(200, json={"group_size": n, "mean_predicted_risk": 0.2})
    if path == "/department/summary":
        return httpx.Response(200, json={"group_size": 446, "by_job_role": {
            "Sales Executive": {"group_size": 326}, "Manager": {"group_size": 37}}, "suppressed_cells": 0})
    if path == "/query":
        q = json.loads(request.content)["question"]
        return httpx.Response(200, json={"answer": f"Echo: {q}", "contexts": []})
    return httpx.Response(200, json={"ok": True})


@pytest.fixture()
def client():
    server.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(_downstream))
    with TestClient(server.app) as c:
        yield c


@pytest.mark.parametrize("g", [0, 1, 4, -3])
def test_small_group_size_param_refused(client, g):
    r = client.post("/tools/get_attrition_risk", json={"filters": {"department": "Sales"}, "group_size": g})
    assert r.status_code == 400
    assert r.json()["detail"] == PRIVACY_REFUSAL


def test_small_resolved_cohort_refused(client):
    r = client.post("/tools/get_attrition_risk", json={"filters": {"job_role": "Research Director"}})
    assert r.status_code == 400 and r.json()["detail"] == PRIVACY_REFUSAL


def test_valid_cohort_ok(client):
    r = client.post("/tools/get_attrition_risk", json={"filters": {"department": "Sales"}})
    assert r.status_code == 200 and r.json()["data"]["group_size"] == 120


def test_stricter_group_floor_suppresses_cells(client):
    r = client.post("/tools/get_department_summary", json={"department": "Sales", "group_size": 50})
    data = r.json()["data"]
    assert "Manager" not in data["by_job_role"] and data["suppressed_cells"] == 1


def test_pii_redacted_before_processing(client):
    r = client.post("/tools/search_hr_policy",
                    json={"question": "Can Priya Sharma (priya.sharma@acme.com) carry forward leave?"})
    body = r.text
    assert r.status_code == 200
    assert "priya.sharma@acme.com" not in body and "Priya Sharma" not in body
    assert "EMAIL_ADDRESS" in r.json()["privacy"]["pii_entities_redacted"]


def test_extra_fields_rejected_without_echo(client):
    r = client.post("/tools/get_attrition_risk", json={"filters": {"department": "Sales"}, "employee_id": "EMP-12345"})
    assert r.status_code == 422 and "EMP-12345" not in r.text


def test_pii_in_enum_rejected_without_echo(client):
    r = client.post("/tools/get_department_summary", json={"department": "neha@acme.com"})
    assert r.status_code == 422 and "neha@acme.com" not in r.text


def test_manifest_lists_five_tools(client):
    names = [t["name"] for t in client.get("/tools").json()["tools"]]
    assert len(names) == 5
