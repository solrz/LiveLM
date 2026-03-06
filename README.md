# 📷 LiveLM

Live camera & screen share → local VLM analysis on Apple Silicon.

Built with [Gradio](https://gradio.app) + [MLX-VLM](https://github.com/Blaizzy/mlx-vlm). Runs entirely on-device using Apple's MLX framework — no cloud API needed.

## Features

- **📷 Camera** — stream your webcam, capture frames for VLM analysis
- **🖥️ Screen Share** — share any screen/window/tab via browser `getDisplayMedia`, then analyze
- **🔄 Auto Mode** — continuous analysis at configurable intervals
- **⚡ Streaming** — token-by-token response display
- **📋 Upload / Clipboard** — drag & drop or paste images directly
- **🔒 Local** — everything runs on your Mac, nothing leaves your machine

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Quick Start

```bash
# Clone
git clone https://github.com/ncmare/LiveLM.git
cd LiveLM

# Install & run (uv)
uv run python app.py

# Or with pip
pip install -e .
python app.py
```

First launch downloads the default model (~2GB). The server starts at **http://localhost:8765**.

## Usage

### Camera Tab
1. Click the webcam area to grant camera access
2. Click **📸 Capture & Ask** to analyze the current frame
3. Toggle **🔄 Auto** for continuous analysis every N seconds
4. Change the **Prompt** to ask different questions

### Screen Tab
1. Click **▶️ Start Screen Share**
2. Browser prompts you to select a screen, window, or tab
3. Live preview streams in the left panel
4. Click **📸 Capture & Ask** to analyze
5. Toggle **🔄 Auto** for continuous screen analysis

## Options

```
python app.py [OPTIONS]

  --model MODEL    HuggingFace model path (default: mlx-community/Qwen2.5-VL-3B-Instruct-4bit)
  --host HOST      Listen host (default: 0.0.0.0)
  --port PORT      Listen port (default: 8765)
  --share          Create a public Gradio share link (HTTPS, good for mobile)
```

### Recommended Models

| Model | Size | RAM | Quality |
|-------|------|-----|---------|
| `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` | ~2GB | 8GB+ | Good |
| `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` | ~4GB | 16GB+ | Better |
| `mlx-community/Qwen2.5-VL-72B-Instruct-4bit` | ~40GB | 64GB+ | Best |

## Mobile / LAN Access

From other devices on the same network, use `http://<your-mac-ip>:8765`.

> ⚠️ Mobile browsers require HTTPS to access the camera. Use `--share` to get a public HTTPS link, or set up a reverse proxy with SSL.

## License

MIT
