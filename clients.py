from tritonclient.grpc import InferenceServerClient
from openai import OpenAI
import os

SERVER_ADDR = os.getenv("SERVER_ADDR", "127.0.0.1")
TRITON_PORT = os.getenv("TRITON_PORT", "8001")
MINDIE_PORT = os.getenv("MINDIE_PORT", "8080")

triton_client = InferenceServerClient(f"{SERVER_ADDR}:{TRITON_PORT}")

openai_client = OpenAI(base_url=f"http://{SERVER_ADDR}:{MINDIE_PORT}/v1", api_key="")
