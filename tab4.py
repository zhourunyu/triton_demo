# Large Language Model Tab
import gradio as gr
import time
from tritonclient.grpc import InferInput, InferRequestedOutput
from clients import triton_client as client
import json
import numpy as np
from random import randint
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import AsyncIterator

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

def process_message(message: dict) -> dict:
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
    messages: list[dict],
    tokenizer: PreTrainedTokenizer,
    model_name: str,
    stream: bool = True,
    max_tokens: int = 2048,
    seed: int | None = None,
    thinking: bool = True,
) -> AsyncIterator[dict]:
    # filter out 'thinking' messages
    def is_thinking(message: dict) -> bool:
        metadata = message["metadata"] or {}
        return "title" in metadata
    messages = list(filter(lambda msg: not is_thinking(msg), messages))
    # preprocess messages
    messages = [process_message(msg) for msg in messages]
    # build prompt
    prompt: str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
    ) # type: ignore

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

def build_stats(ttft: float, token_count: int, first_token_time: float) -> str:
    if ttft == 0.0:
        return ""
    throughput = token_count / (time.time() - first_token_time) if token_count > 1 else 0.0
    return f"首token用时: {ttft:.2f}s | {throughput:.2f} token/s ({token_count} tokens)"

async def chat(messages: list[dict], model: str) -> AsyncIterator[tuple[list[dict], str]]:
    if not messages:
        return

    global stopped
    stopped = False

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAMES[model])
    request = build_request(messages, tokenizer, model)

    start_time = time.time()
    first_token_time = 0.0
    ttft = 0.0
    output_token_count = 0
    response = client().stream_infer(request)
    if response is None:
        raise gr.Error("Inference request failed.")

    messages += [{"role": "assistant", "content": ""}]
    thinking = False
    thinking_start_time = 0.0
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

        if content:
            # record first token time
            if first_token_time == 0.0:
                first_token_time = time.time()
                ttft = first_token_time - start_time

            # accumulate token count
            output_token_count += len(tokenizer.encode(content, add_special_tokens=False))

        if model.startswith("DeepSeek-R1") and thinking_start_time == 0.0:
            # DeepSeek-R1 has <think> at the end of prompt
            content = "<think>" + content

        if not thinking and content.startswith("<think>"):
            thinking = True
            thinking_start_time = time.time()
            content = content[len("<think>") :]
            messages[-1]["metadata"] = {"title": "思考中...", "duration": 0.0}

        if thinking:
            messages[-1]["metadata"]["duration"] = time.time() - thinking_start_time
            if "</think>" in content:
                thinking_content, content = content.split("</think>", 1)
                thinking = False
                messages[-1]["content"] += thinking_content
                messages[-1]["metadata"]["title"] = "思考内容"
                messages += [{"role": "assistant", "content": ""}]

        messages[-1]["content"] += content
        yield messages, build_stats(ttft, output_token_count, first_token_time)

def create():
    with gr.TabItem("大语言模型"):
        gr.Markdown("# 大语言模型")

        chatbot = gr.Chatbot(height=500, label="对话记录")
        stats = gr.HTML(html_template='<div style="text-align:right; color:gray; font-size:var(--size-3); padding:2px 4px;">${value}</div>')
        input = gr.Textbox(label="输入", lines=3)

        model = gr.Dropdown(["Qwen3-4B", "Qwen3-8B", "Qwen2.5-7B", "DeepSeek-R1-7B"], value="Qwen3-4B", label="模型")
        with gr.Row():
            submit_btn = gr.Button("提交", variant="primary")
            stop_btn = gr.Button("停止", variant="stop")
            clear_btn = gr.Button("清除记录")
        examples = gr.Dataset(
            components=[input],
            samples=[
                ["自主着陆的注意事项有哪些？"],
                ["液压系统b系统失效如何处理？"],
                ["波音飞机多重？"],
                ["无人机如何实现自主避障？"],
            ],
            label="示例"
        )

        # append user message
        append_message = lambda input, messages: messages + [{"role": "user", "content": input}]

        (
            # update chat history
            input.submit(append_message, inputs=[input, chatbot], outputs=chatbot)
            # clear input box
            .then(lambda: "", outputs=input)
            # call chat function
            .then(chat, inputs=[chatbot, model], outputs=[chatbot, stats], show_progress_on=chatbot)
        )
        (
            submit_btn.click(append_message, inputs=[input, chatbot], outputs=chatbot)
            .then(lambda: "", outputs=input)
            .then(chat, inputs=[chatbot, model], outputs=[chatbot, stats], show_progress_on=chatbot)
        )
        (
            examples.click(append_message, inputs=[examples, chatbot], outputs=chatbot)
            .then(chat, inputs=[chatbot, model], outputs=[chatbot, stats], show_progress_on=chatbot)
        )
        clear_btn.click(lambda: (None, ""), outputs=[chatbot, stats])
        stop_btn.click(stop_chat)
