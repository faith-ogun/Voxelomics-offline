import os
import sys
import tempfile
from pathlib import Path

import pytest


SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

# Keep tests hermetic regardless of local shell/.env values.
os.environ["MDT_EXECUTION_MODE"] = "local"
os.environ["MDT_CASE_STORE_BACKEND"] = "sqlite"
os.environ["MDT_RETRIEVAL_MODE"] = "local"
os.environ["MDT_MEDASR_MODE"] = "local"
os.environ["MDT_DIAGNOSTICORE_FETCH_MODE"] = "off"
os.environ["MDT_MODEL_ROUTER_MODE"] = "medgemma_only"
os.environ["MDT_ADK_STAGE2_LLM_ENABLED"] = "false"
os.environ["MDT_AGENT_CALL_TIMEOUT_SECONDS"] = "5"
os.environ["MDT_AGENT_CALL_TIMEOUT_CAP_SECONDS"] = "5"
os.environ["MDT_DISABLE_MEDGEMMA_ON_TIMEOUT"] = "true"

_TEST_DATA_DIR = Path(tempfile.gettempdir()) / "mdt-command-service-tests"
_TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MDT_LOCAL_DATA_DIR"] = str(_TEST_DATA_DIR)
os.environ["MDT_SQLITE_DB_PATH"] = str(_TEST_DATA_DIR / "mdt_cases.sqlite3")
os.environ["MDT_MEDGEMMA_LOCAL_MODEL_ID"] = str(_TEST_DATA_DIR / "missing-medgemma-model")


@pytest.fixture(autouse=True)
def _stub_medgemma_runtime(monkeypatch):
    from agents import MDTCommandOrchestrator

    async def _fast_generate_json(self, **_kwargs):
        return {}

    monkeypatch.setattr(MDTCommandOrchestrator, "_generate_medgemma_json", _fast_generate_json)
    monkeypatch.setattr(
        MDTCommandOrchestrator,
        "_agent_uses_medgemma_endpoint",
        lambda self, _agent_name: False,
    )
