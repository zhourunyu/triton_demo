from tritonclient.grpc.aio import InferenceServerClient
import aiohttp
import os

SERVER_ADDR = os.getenv("SERVER_ADDR", "127.0.0.1")
TRITON_PORT = os.getenv("TRITON_PORT", "8001")
METRICS_PORT = os.getenv("METRICS_PORT", "9000")
METRICS_DEVICE = os.getenv("METRICS_DEVICE", "NPU:0")

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

async def get_metrics() -> dict:
    device, id = METRICS_DEVICE.split(":")
    device = device.lower()
    if not device in ["npu", "dlgpu", "corex"]:
        raise ValueError(f"Unknown device type: {device}. Supported types are 'npu', 'dlgpu' and 'corex'.")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{metrics_url}/metrics/{device}?device={id}") as response:
                if response.status == 200:
                    metrics = await response.json()
                    if device == "npu":
                        return {
                            "name": metrics.get("name"),
                            "state": 0 if metrics.get("health") == 0 else 1,
                            "temperature": metrics.get("temperature"),
                            "memory_utilization": metrics.get("memory_utilization"),
                            "utilization": metrics.get("aicore_utilization"),
                        }
                    else:
                        return {
                            "name": metrics.get("name"),
                            "state": 0 if metrics.get("state") >= 0 else 1,
                            "temperature": metrics.get("temperature"),
                            "memory_utilization": metrics.get("memory_utilization"),
                            "utilization": metrics.get("gpu_utilization"),
                        }
    except Exception:
        pass
    return {}
