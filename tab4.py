# Large Language Model Tab
import gradio as gr
import time
from clients import openai_client as client
from typing import Iterator, Any
from openai.types.chat import ChatCompletionMessageParam

stopped = False

def stop_chat():
    global stopped
    stopped = True

def convert_message(message: dict[str, Any]) -> ChatCompletionMessageParam:
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
        raise gr.Error(f"Unsupported content format: {type(content)}")

def chat(messages: list[dict[str, Any]], model: str) -> Iterator[list[dict[str, Any]]]:
    if not messages:
        return

    global stopped
    stopped = False

    # filter out 'thinking' messages
    def is_thinking(message: dict[str, Any]) -> bool:
        metadata = message.get("metadata", {}) or {}
        return "title" in metadata

    messages_converted = [convert_message(m) for m in messages if not is_thinking(m)]
    response = client.chat.completions.create(
        model=model,
        messages=messages_converted,
        stream=True,
    )

    messages += [{"role": "assistant", "content": ""}]
    thinking = False
    thinking_start = 0.0
    for chunk in response:
        if stopped:
            response.close()
            break
        if len(chunk.choices) == 0:
            continue

        content = chunk.choices[0].delta.content or ""
        reasoning_content: str = getattr(chunk.choices[0].delta, "reasoning_content", "")
        if not thinking and reasoning_content:
            thinking = True
            thinking_start = time.time()
            messages[-1]["metadata"] = {"title": "Thinking...", "duration": 0.0}

        if thinking:
            messages[-1]["metadata"]["duration"] = time.time() - thinking_start
            if reasoning_content:
                messages[-1]["content"] += reasoning_content
            if content:
                thinking = False
                messages[-1]["metadata"]["title"] = "Thoughts"
                messages += [{"role": "assistant", "content": ""}]

        messages[-1]["content"] += content
        yield messages

def create():
    with gr.TabItem("Large Language Model"):
        gr.Markdown("# Large Language Model")

        chatbot = gr.Chatbot(height=500)
        input = gr.Textbox(label="Input", lines=3)

        model = gr.Dropdown(["Qwen3-4B", "Qwen3-8B", "Qwen2.5-7B", "DeepSeek-R1-7B"], value="Qwen3-4B", label="Model")
        with gr.Row():
            submit_btn = gr.Button("Submit", variant="primary")
            stop_btn = gr.Button("Stop", variant="stop")
            clear_btn = gr.Button("Clear")
        examples = gr.Dataset(
            components=[input],
            samples=[
                ["Which is bigger, 9.9 or 9.11?"],
                ['How many "r"s are there in "strawberry"?'],
                ["Explain the theory of relativity."],
                ["Write a poem about the sea."],
            ],
            label="Examples"
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
