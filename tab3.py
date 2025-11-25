# Masked Language Modeling Tab
import gradio as gr
import numpy as np
from tritonclient.grpc import InferInput, InferRequestedOutput
from clients import triton_client as client
from transformers import BertTokenizer

# Get prediction results
def postprocess(logits: np.ndarray, input_ids: np.ndarray, tokenizer) -> dict:
    masked_index = np.where(input_ids == tokenizer.mask_token_id)[0][0]
    probabilities = logits[masked_index]
    probabilities = probabilities - np.max(probabilities)
    probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))

    top5_idx = np.argsort(probabilities)[-5:][::-1]
    results_dict = {}
    for idx in top5_idx:
        token = tokenizer.decode(idx)
        results_dict[token] = float(probabilities[idx])
    return results_dict

def predict(text: str, model_name: str) -> dict | None:
    if not text:
        return None

    tokenizer = BertTokenizer.from_pretrained(model_name)
    input: dict[str, np.ndarray] = tokenizer(text, return_tensors="np",
        padding="max_length", max_length=32    # for ascendcl backend
    )
    input_ids = input["input_ids"].astype(np.int64)
    attention_mask = input["attention_mask"].astype(np.int64)
    token_type_ids = input["token_type_ids"].astype(np.int64)

    inputs = [
        InferInput("input_ids", input_ids.shape, "INT64"),
        InferInput("attention_mask", attention_mask.shape, "INT64"),
        InferInput("token_type_ids", token_type_ids.shape, "INT64"),
    ]
    inputs[0].set_data_from_numpy(input_ids)
    inputs[1].set_data_from_numpy(attention_mask)
    inputs[2].set_data_from_numpy(token_type_ids)
    outputs = [InferRequestedOutput("output")]

    result = client.infer(model_name, inputs=inputs, outputs=outputs)
    if result is None:
        raise gr.Error("Inference request failed.")
    output: np.ndarray | None = result.as_numpy("output")
    if output is None:
        raise gr.Error("Failed to get output from inference result.")

    return postprocess(output[0], input_ids[0], tokenizer)

def create():
    with gr.TabItem("Masked Language Modeling"):
        gr.Markdown("# Masked Language Modeling")
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Input", lines=3)
                model = gr.Dropdown(["bert-base-cased"], value="bert-base-cased", label="Model")
                submit_btn = gr.Button("Submit", variant="primary")
                examples = gr.Dataset(
                    components=[text],
                    samples=[
                        ["Paris is the [MASK] of France."],
                        ["The goal of life is [MASK]."],
                        ["[MASK] is the largest planet in our solar system."],
                    ],
                    label="Examples"
                )

            with gr.Column():
                output = gr.Label(num_top_classes=5)

    def validate_input(text: str, model: str) -> tuple[str, str]:
        if "[MASK]" not in text:
            raise gr.Error("Input text must contain [MASK] for prediction.")
        return text, model

    submit_btn.click(fn=predict, inputs=[text, model], outputs=output, validator=validate_input)
    model.change(fn=predict, inputs=[text, model], outputs=output)
    examples.click(lambda x: x[0], examples, text, queue=False).then(predict, [text, model], output)
