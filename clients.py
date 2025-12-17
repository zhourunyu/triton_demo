from tritonclient.grpc import InferenceServerClient
from openai import OpenAI
import os
import requests

SERVER_ADDR = os.getenv("SERVER_ADDR", "127.0.0.1")
TRITON_PORT = os.getenv("TRITON_PORT", "8001")
MINDIE_PORT = os.getenv("MINDIE_PORT", "8080")
METRICS_PORT = os.getenv("METRICS_PORT", "9000")

triton_client = InferenceServerClient(f"{SERVER_ADDR}:{TRITON_PORT}")

openai_client = OpenAI(base_url=f"http://{SERVER_ADDR}:{MINDIE_PORT}/v1", api_key="")

metrics_url = f"http://{SERVER_ADDR}:{METRICS_PORT}"

def get_status() -> tuple[bool, bool]:
    try:
        response = requests.get(f"{metrics_url}/health", timeout=1)
        if response.status_code == 200:
            return True, True
        elif response.status_code == 503:
            status = response.json()
            return status.get("triton", False), status.get("mindie", False)
    except Exception:
        pass
    return False, False

def get_npu_metrics() -> dict:
    try:
        response = requests.get(f"{metrics_url}/metrics/npu", timeout=1)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {}
