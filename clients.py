from tritonclient.grpc import InferenceServerClient
import os

SERVER_ADDR = os.getenv("SERVER_ADDR", "127.0.0.1")
TRITON_PORT = os.getenv("TRITON_PORT", "8001")

triton_client = InferenceServerClient(f"{SERVER_ADDR}:{TRITON_PORT}")
