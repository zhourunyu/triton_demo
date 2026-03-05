import gradio as gr
import tab1
import tab2
import tab3
import tab4
from clients import get_status, get_npu_metrics

async def status_text():
    triton_status = await get_status()
    status_text = {
        True: "🟢 在线",
        False: "🔴 离线"
    }
    status = f"服务状态： {status_text[triton_status]}"
    return status

async def npu_metrics_text():
    npu_metrics = await get_npu_metrics()
    health_text = {
        0: "🟢 正常",
        1: "🟡 警告",
        2: "🔴 严重",
        3: "🔴 错误",
    }
    health = npu_metrics.get("health", -1)
    status = health_text.get(health, "❔ 未知")
    temp = npu_metrics.get("temperature", "N/A")
    mem_util = npu_metrics.get("memory_utilization", "N/A")
    aicore_util = npu_metrics.get("aicore_utilization", "N/A")
    metrics = f"NPU: {status} | 温度: {temp}°C | 内存: {mem_util}% | AI Core: {aicore_util}%"
    return metrics

async def update_status_bar():
    status = await status_text()
    npu_metrics = await npu_metrics_text()
    return f"{status}<br>{npu_metrics}"

with gr.Blocks(title="大小模型混合推理服务演示", analytics_enabled=False) as demo:
    with gr.Tabs():
        tab1.create()
        tab2.create()
        tab3.create()
        tab4.create()

    status_bar = gr.HTML(
        html_template='<div style="text-align:center; color:gray; font-size:var(--size-4); border-top:1px solid lightgray; margin-top:var(--size-4); padding-top:var(--size-2);">${value}</div>'
    )

    timer = gr.Timer(1)
    timer.tick(fn=update_status_bar, outputs=status_bar, show_progress_on=[])
demo.queue(default_concurrency_limit=None)

css = """
.progress-text {
    display: none !important;
}
footer {
    display: none !important;
}
"""

if __name__ == "__main__":
    demo.launch(css=css)
