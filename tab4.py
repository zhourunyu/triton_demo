# Large Language Model Tab
import gradio as gr
import time
from tritonclient.grpc import InferInput, InferRequestedOutput
from clients import triton_client as client
import json
import numpy as np
from random import randint
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import Any, AsyncIterator

TOKENIZER_NAMES = {
    "Qwen3-4B": "Qwen/Qwen3-4B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
    "DeepSeek-R1-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}

stopped = False

def stop_chat():
    global stopped
    stopped = True

def process_message(message: dict[str, Any]) -> dict[str, Any]:
    role = message["role"]
    content = message["content"]
    if isinstance(content, list):
        if len(content) > 1:
            raise gr.Error("Multiple message contents are not supported.")
        content = content[0]
    if isinstance(content, str):
        return {"role": role, "content": content}
    elif isinstance(content, dict):
        content_type = content.get("type")
        if content_type == "text":
            return {"role": role, "content": content.get("text", "")}
        else:
            raise gr.Error(f"Unsupported content type: {content_type}")
    else:
        raise gr.Error(f"Unsupported content: {content}")

def build_request(
    messages: list[dict[str, Any]],
    model_name: str,
    stream: bool = True,
    max_tokens: int = 2048,
    seed: int | None = None,
    thinking: bool = True,
) -> AsyncIterator[dict[str, Any]]:
    # filter out 'thinking' messages
    def is_thinking(message: dict[str, Any]) -> bool:
        metadata = message["metadata"] or {}
        return "title" in metadata
    messages = list(filter(lambda msg: not is_thinking(msg), messages))
    # preprocess messages
    messages = [process_message(msg) for msg in messages]
    # build prompt
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAMES[model_name])
    prompt: str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
    )

    inputs = []
    prompt_data = np.array([prompt.encode("utf-8")], dtype=np.object_)
    inputs.append(InferInput("text_input", [1], "BYTES"))
    inputs[-1].set_data_from_numpy(prompt_data)

    stream_data = np.array([stream], dtype=bool)
    inputs.append(InferInput("stream", [1], "BOOL"))
    inputs[-1].set_data_from_numpy(stream_data)

    sampling_parameters = {
        "max_tokens": max_tokens,
        "seed": randint(0, 2**31 - 1) if seed is None else seed,
    }
    sampling_parameters_data = np.array(
        [json.dumps(sampling_parameters).encode("utf-8")], dtype=np.object_
    )
    inputs.append(InferInput("sampling_parameters", [1], "BYTES"))
    inputs[-1].set_data_from_numpy(sampling_parameters_data)

    # Add requested outputs
    outputs = []
    outputs.append(InferRequestedOutput("text_output"))

    async def request_iterator():
        yield {
            "model_name": model_name,
            "inputs": inputs,
            "outputs": outputs,
            "parameters": sampling_parameters,
        }

    return request_iterator()

async def chat(messages: list[dict[str, Any]], model: str) -> AsyncIterator[list[dict[str, Any]]]:
    if not messages:
        return

    global stopped
    stopped = False

    request = build_request(messages, model)
    response = client().stream_infer(request)
    if response is None:
        raise gr.Error("Inference request failed.")

    messages += [{"role": "assistant", "content": ""}]
    thinking = False
    thinking_start = 0.0
    async for (result, error) in response:
        if stopped:
            response.cancel()
            break
        if error:
            raise gr.Error(f"Inference error: {error}")
        if result is None:
            continue

        output = result.as_numpy("text_output")
        if output is None:
            raise gr.Error("Failed to get output from inference result.")
        content: str = output[0].decode("utf-8")

        if model.startswith("DeepSeek-R1") and thinking_start == 0.0:
            # DeepSeek-R1 has <think> at the end of prompt
            content = "<think>" + content

        if not thinking and content.startswith("<think>"):
            thinking = True
            thinking_start = time.time()
            content = content[len("<think>") :]
            messages[-1]["metadata"] = {"title": "思考中...", "duration": 0.0}

        if thinking:
            messages[-1]["metadata"]["duration"] = time.time() - thinking_start
            if "</think>" in content:
                thinking_content, content = content.split("</think>", 1)
                thinking = False
                messages[-1]["content"] += thinking_content
                messages[-1]["metadata"]["title"] = "思考内容"
                messages += [{"role": "assistant", "content": ""}]

        messages[-1]["content"] += content
        yield messages

def create():
    with gr.TabItem("大语言模型"):
        gr.Markdown("# 大语言模型")

        chatbot = gr.Chatbot(height=500, label="对话记录")
        input = gr.Textbox(label="输入", lines=3)

        model = gr.Dropdown(["Qwen3-4B", "Qwen3-8B", "Qwen2.5-7B", "DeepSeek-R1-7B"], value="Qwen3-4B", label="模型")
        with gr.Row():
            submit_btn = gr.Button("提交", variant="primary")
            stop_btn = gr.Button("停止", variant="stop")
            clear_btn = gr.Button("清除记录")
        examples = gr.Dataset(
            components=[input],
            samples=[
                ["9.9和9.11哪个更大？"],
                ["strawberry中有多少个r？"],
                ["解释相对论。"],
                ["写一首关于大海的诗。"],
            ],
            label="示例"
        )

        # append user message
        append_message = lambda input, messages: messages + [{"role": "user", "content": input}]

        input.submit(append_message, [input, chatbot], [chatbot], queue=False).then(lambda: "", None, input, queue=False).then(
            chat, [chatbot, model], [chatbot]
        )
        submit_btn.click(append_message, [input, chatbot], [chatbot], queue=False).then(lambda: "", None, input, queue=False).then(
            chat, [chatbot, model], [chatbot]
        )
        examples.click(lambda example: [{"role": "user", "content": example}], [examples], [chatbot], queue=False).then(
            chat, [chatbot, model], [chatbot]
        )
        clear_btn.click(lambda: None, None, chatbot, queue=False)
        stop_btn.click(stop_chat, None, None, queue=False)
