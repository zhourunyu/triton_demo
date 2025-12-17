# Triton Demo

Gradio demo for models served with Triton Inference Server and OpenAI API.

## Usage

1. Clone and install dependencies
```bash
git clone https://github.com/zhourunyu/triton-demo.git
cd triton-demo
pip install -r requirements.txt
```

2. Set the server URL and ports if needed
```bash
export SERVER_ADDR=localhost
export TRITON_PORT=8001
export MINDIE_PORT=8080
export METRICS_PORT=9000
```

3. Run the Gradio app
```bash
python main.py
```

4. Open your browser and go to `http://localhost:7860` to access the demo.
