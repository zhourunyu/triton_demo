import gradio as gr
import tab1
import tab2
import tab3

with gr.Blocks(analytics_enabled=False) as demo:
    with gr.Tabs():
        tab1.create()
        tab2.create()
        tab3.create()

if __name__ == "__main__":
    demo.launch(css="footer {display: none !important;}")
