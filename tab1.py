# Image Classification Tab
import gradio as gr
import numpy as np
from tritonclient.grpc import InferInput, InferRequestedOutput
from clients import triton_client as client
import cv2
import json

with open("examples/imagenet-labels.json") as f:
    LABELS: list[str] = json.load(f)

# Preprocess the input image
def preprocess(image: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (232, 232))
    image = image[4:228, 4:228]                 # Center crop to 224x224
    image = image.astype(np.float32) / 255.0    # Scale to [0, 1]
    image = (image - mean) / std                # Normalize
    image = np.transpose(image, (2, 0, 1))      # HWC to CHW
    return image

# Get top-5 predictions from logits
def postprocess(output: np.ndarray) -> dict[str, float]:
    # Apply softmax to get probabilities
    output = output - np.max(output)
    probabilities = np.exp(output) / np.sum(np.exp(output))

    top5_idx = np.argsort(probabilities)[-5:][::-1]
    results_dict = {}
    for idx in top5_idx:
        class_name = LABELS[idx]
        confidence = float(probabilities[idx])
        results_dict[class_name] = confidence

    return results_dict

def predict(image: np.ndarray | None, model_name: str) -> dict[str, float] | None:
    if image is None:
        return None

    input = preprocess(image)
    input = np.expand_dims(input, axis=0)

    inputs = [InferInput("input", input.shape, "FP32")]
    inputs[0].set_data_from_numpy(input)
    outputs = [InferRequestedOutput("output")]

    result = client.infer(model_name, inputs=inputs, outputs=outputs)
    if result is None:
        raise gr.Error("Inference request failed.")
    output: np.ndarray | None = result.as_numpy("output")
    if output is None:
        raise gr.Error("Failed to get output from inference result.")

    return postprocess(output[0])

def create():
    with gr.TabItem("Image Classification"):
        gr.Markdown("# Image Classification")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="numpy", height=300)
                model = gr.Dropdown(["resnet50", "mobilenet_v2"], value="resnet50", label="Model")
                examples = gr.Examples("examples/classification", image)

            with gr.Column():
                output = gr.Label(num_top_classes=5)

    image.change(fn=predict, inputs=[image, model], outputs=output)
    model.change(fn=predict, inputs=[image, model], outputs=output)
