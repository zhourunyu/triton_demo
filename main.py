import gradio as gr
import tab1
import tab2
import tab3
import tab4
from clients import get_status, get_npu_metrics

def status_text():
    triton_status, mindie_status = get_status()
    status_text = {
        True: "🟢 Up",
        False: "🔴 Down"
    }
    status = f"Triton Server: {status_text[triton_status]} | MindIE Server: {status_text[mindie_status]}"
    return status

def npu_metrics_text():
    npu_metrics = get_npu_metrics()
    health_text = {
        0: "🟢 Normal",
        1: "🟡 Warning",
        2: "🔴 Critical",
        3: "🔴 Error",
    }
    health = npu_metrics.get("health", -1)
    status = health_text.get(health, "❔ Unknown")
    temp = npu_metrics.get("temperature", "N/A")
    mem_util = npu_metrics.get("memory_utilization", "N/A")
    aicore_util = npu_metrics.get("aicore_utilization", "N/A")
    metrics = f"NPU: {status} | Temp: {temp}°C | Mem Util: {mem_util}% | AI Core Util: {aicore_util}%"
    return metrics

def update_status_bar():
    status = status_text()
    npu_metrics = npu_metrics_text()
    return f"{status}<br>{npu_metrics}"

with gr.Blocks(analytics_enabled=False) as demo:
    with gr.Tabs():
        tab1.create()
        tab2.create()
        tab3.create()
        tab4.create()

    status_bar = gr.HTML(update_status_bar(), elem_classes="status-bar")

    timer = gr.Timer(1)
    timer.tick(fn=update_status_bar, outputs=status_bar, show_progress_on=[])

css = """
.status-bar {
    text-align: center;
    font-size: var(--size-4);
    border-top: 1px solid #ccc;
    margin-top: var(--size-4);
    padding-top: var(--size-2);
    color: #666666;
}
footer {
    display: none !important;
}
"""

if __name__ == "__main__":
    demo.launch(css=css)
