from tritonclient.grpc.aio import InferenceServerClient
import aiohttp
import os

SERVER_ADDR = os.getenv("SERVER_ADDR", "127.0.0.1")
TRITON_PORT = os.getenv("TRITON_PORT", "8001")
METRICS_PORT = os.getenv("METRICS_PORT", "9000")

_triton_client: InferenceServerClient | None = None

metrics_url = f"http://{SERVER_ADDR}:{METRICS_PORT}"

# Lazy initialize TritonClient in Gradio's event loop
def triton_client() -> InferenceServerClient:
    global _triton_client
    if _triton_client is None:
        _triton_client = InferenceServerClient(
            url=f"{SERVER_ADDR}:{TRITON_PORT}",
            verbose=False,
        )
    return _triton_client

async def get_status() -> bool:
    try:
        return await triton_client().is_server_live() or False
    except Exception:
        return False

async def get_npu_metrics() -> dict:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{metrics_url}/metrics/npu") as response:
                if response.status == 200:
                    return await response.json()
    except Exception:
        pass
    return {}
