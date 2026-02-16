import sys
from pathlib import Path
import os


SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

# Keep tests hermetic regardless of local shell/.env values.
os.environ["MDT_EXECUTION_MODE"] = "mock"
os.environ["MDT_CASE_STORE_BACKEND"] = "memory"
os.environ["MDT_RETRIEVAL_MODE"] = "mock"
os.environ["MDT_MEDASR_MODE"] = "mock"
os.environ["MDT_DIAGNOSTICORE_FETCH_MODE"] = "off"
os.environ["MDT_MODEL_ROUTER_MODE"] = "primary"
os.environ["MDT_ADK_STAGE2_LLM_ENABLED"] = "false"
