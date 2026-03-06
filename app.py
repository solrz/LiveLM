"""
LiveLM — Camera + Screen Share (streaming) → VLM analysis
Both camera and screen use continuous streaming with on-demand or auto capture.
"""

import argparse
import base64
import io
import os
import tempfile
import threading
import time

import gradio as gr
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# VLM backend
# ---------------------------------------------------------------------------

_model = None
_processor = None
_model_lock = threading.Lock()
_tmp_dir = tempfile.mkdtemp(prefix="livelm_")

DEFAULT_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"


def _ensure_model(model_path: str):
    global _model, _processor
    with _model_lock:
        if _model is not None:
            return
        print(f"⏳ Loading model: {model_path} ...")
        from mlx_vlm import load
        _model, _processor = load(model_path)
        print("✅ Model loaded!")


def _resize(img: Image.Image, max_dim: int = 1024) -> Image.Image:
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize(
            (int(img.width * ratio), int(img.height * ratio)),
            Image.LANCZOS,
        )
    return img


def vlm_generate(image: Image.Image, prompt: str, max_tokens: int = 512):
    from mlx_vlm import stream_generate

    img_path = os.path.join(_tmp_dir, f"cap_{time.time():.0f}.jpg")
    image.save(img_path, format="JPEG", quality=85)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text_prompt = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    output = ""
    for token in stream_generate(
        _model, _processor, text_prompt,
        image=[img_path],
        max_tokens=max_tokens,
        temperature=0.6,
        repetition_penalty=1.5,
        repetition_context_size=128,
        top_p=0.9,
    ):
        output += token.text
        if len(output) > 20 and len(set(output[-20:])) <= 2:
            output += "\n\n[stopped: repetition detected]"
            yield output
            return
        yield output

    try:
        os.remove(img_path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def analyze_numpy(image_data, prompt, max_tokens):
    """Analyze a numpy image (from camera or upload)."""
    if image_data is None:
        yield "⏳ No image yet..."
        return
    img = Image.fromarray(image_data.astype(np.uint8))
    img = _resize(img)
    print(f"[LiveLM] Camera: {img.size}")
    for partial in vlm_generate(img, prompt, int(max_tokens)):
        yield partial


def analyze_b64(b64_data, prompt, max_tokens):
    """Analyze a base64 image (from screen capture JS)."""
    if not b64_data:
        yield "⏳ Click 'Start Screen Share' first, then 'Capture & Ask'."
        return
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(b64_data)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        yield f"❌ Failed to decode image: {e}"
        return
    img = _resize(img)
    print(f"[LiveLM] Screen: {img.size}")
    for partial in vlm_generate(img, prompt, int(max_tokens)):
        yield partial


# ---------------------------------------------------------------------------
# JS for screen share streaming
# ---------------------------------------------------------------------------

# This JS:
# 1. Calls getDisplayMedia to start screen sharing
# 2. Renders the stream to a <video> in the preview area
# 3. Sets up a hidden canvas for frame grabbing
START_SCREEN_JS = """
async () => {
    try {
        // Stop existing stream if any
        if (window._livelm_stream) {
            window._livelm_stream.getTracks().forEach(t => t.stop());
        }

        const stream = await navigator.mediaDevices.getDisplayMedia({
            video: { cursor: "always", frameRate: 30 },
            audio: false
        });
        window._livelm_stream = stream;

        // Find or create video element in preview
        let container = document.querySelector('#screen-video-container');
        if (!container) return "❌ Container not found";

        container.innerHTML = '';
        const video = document.createElement('video');
        video.srcObject = stream;
        video.autoplay = true;
        video.playsInline = true;
        video.muted = true;
        video.style.cssText = 'width:100%;max-height:400px;border-radius:8px;object-fit:contain;background:#000;';
        video.id = 'screen-video';
        container.appendChild(video);

        // Handle stream end (user clicks "Stop sharing")
        stream.getVideoTracks()[0].onended = () => {
            container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:300px;border:2px dashed #ccc;border-radius:8px;color:#999;">Screen share ended. Click Start to share again.</div>';
            window._livelm_stream = null;
        };

        return "✅ Screen sharing started! Click 'Capture & Ask' to analyze.";
    } catch (err) {
        return "❌ " + err.message;
    }
}
"""

# Grab a single frame from the video stream → base64 JPEG
# JS receives (screen_b64, prompt, max_tok), must return all three with b64 replaced
GRAB_FRAME_JS = """
(current_b64, prompt, max_tok) => {
    const video = document.getElementById('screen-video');
    if (!video || !video.srcObject || video.videoWidth === 0) {
        return ["", prompt, max_tok];
    }
    const canvas = document.createElement('canvas');
    const scale = Math.min(1, 1280 / Math.max(video.videoWidth, video.videoHeight));
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const b64 = canvas.toDataURL('image/jpeg', 0.75);
    return [b64, prompt, max_tok];
}
"""

STOP_SCREEN_JS = """
() => {
    if (window._livelm_stream) {
        window._livelm_stream.getTracks().forEach(t => t.stop());
        window._livelm_stream = null;
    }
    const container = document.querySelector('#screen-video-container');
    if (container) {
        container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:300px;border:2px dashed #ccc;border-radius:8px;color:#999;">Click Start Screen Share to begin.</div>';
    }
    return "⏹️ Screen share stopped.";
}
"""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="LiveLM") as demo:
        gr.Markdown("# 📷 LiveLM\nCamera / Screen → local VLM analysis")

        with gr.Tabs():
            # ---- Camera Tab ----
            with gr.Tab("📷 Camera"):
                with gr.Row():
                    with gr.Column(scale=1):
                        webcam = gr.Image(
                            sources=["webcam"],
                            type="numpy",
                            label="Camera",
                            streaming=True,
                        )
                        upload_img = gr.Image(
                            sources=["upload", "clipboard"],
                            type="numpy",
                            label="Or upload / paste",
                        )
                    with gr.Column(scale=1):
                        cam_output = gr.Textbox(
                            label="💬 Response",
                            lines=18, max_lines=30,
                            interactive=False, buttons=["copy"],
                        )
                cam_prompt = gr.Textbox(
                    value="Describe what you see in this image.",
                    label="Prompt", lines=2,
                )
                cam_max_tok = gr.Slider(64, 2048, value=512, step=64, label="Max tokens")
                cam_btn = gr.Button("📸 Capture & Ask", variant="primary", size="lg")
                with gr.Row():
                    cam_auto = gr.Checkbox(label="🔄 Auto", value=False)
                    cam_interval = gr.Slider(3, 30, value=5, step=1, label="Interval (s)")

                def cam_analyze(cam, upload, prompt, tok):
                    data = cam if cam is not None else upload
                    yield from analyze_numpy(data, prompt, tok)

                cam_inputs = [webcam, upload_img, cam_prompt, cam_max_tok]
                cam_btn.click(fn=cam_analyze, inputs=cam_inputs, outputs=cam_output)

                cam_timer = gr.Timer(value=5, active=False)
                cam_auto.change(
                    fn=lambda on, s: gr.Timer(active=on, value=s),
                    inputs=[cam_auto, cam_interval], outputs=[cam_timer])
                cam_interval.change(
                    fn=lambda on, s: gr.Timer(active=on, value=s),
                    inputs=[cam_auto, cam_interval], outputs=[cam_timer])
                cam_timer.tick(fn=cam_analyze, inputs=cam_inputs, outputs=cam_output)

            # ---- Screen Tab ----
            with gr.Tab("🖥️ Screen"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML(
                            '<div id="screen-video-container" style="min-height:300px;">'
                            '<div style="display:flex;align-items:center;justify-content:center;'
                            'height:300px;border:2px dashed #ccc;border-radius:8px;color:#999;">'
                            'Click "Start Screen Share" to begin</div></div>'
                        )
                        screen_status = gr.Textbox(
                            label="Status", interactive=False, lines=1,
                        )
                        with gr.Row():
                            start_btn = gr.Button("▶️ Start Screen Share", variant="primary")
                            stop_btn = gr.Button("⏹️ Stop", variant="stop")

                    with gr.Column(scale=1):
                        screen_output = gr.Textbox(
                            label="💬 Response",
                            lines=18, max_lines=30,
                            interactive=False, buttons=["copy"],
                        )

                screen_prompt = gr.Textbox(
                    value="Describe what's on this screen.",
                    label="Prompt", lines=2,
                )
                screen_max_tok = gr.Slider(64, 2048, value=512, step=64, label="Max tokens")

                # Hidden textbox for frame data (base64)
                screen_b64 = gr.Textbox(visible=False, elem_id="screen-b64")

                screen_btn = gr.Button("📸 Capture & Ask", variant="primary", size="lg")
                with gr.Row():
                    screen_auto = gr.Checkbox(label="🔄 Auto", value=False)
                    screen_interval = gr.Slider(3, 30, value=8, step=1, label="Interval (s)")

                # Start/stop screen share (JS only)
                start_btn.click(fn=None, outputs=screen_status, js=START_SCREEN_JS)
                stop_btn.click(fn=None, outputs=screen_status, js=STOP_SCREEN_JS)

                # Capture: single click does JS grab + Python analyze
                # The JS runs first (grabs frame, returns b64 to screen_b64),
                # then Python fn runs with the updated value
                screen_btn.click(
                    fn=analyze_b64,
                    inputs=[screen_b64, screen_prompt, screen_max_tok],
                    outputs=screen_output,
                    js=GRAB_FRAME_JS,
                )

                # Auto mode for screen
                screen_timer = gr.Timer(value=8, active=False)
                screen_auto.change(
                    fn=lambda on, s: gr.Timer(active=on, value=s),
                    inputs=[screen_auto, screen_interval], outputs=[screen_timer])
                screen_interval.change(
                    fn=lambda on, s: gr.Timer(active=on, value=s),
                    inputs=[screen_auto, screen_interval], outputs=[screen_timer])

                # On timer tick: grab frame via JS, then analyze
                screen_timer.tick(
                    fn=analyze_b64,
                    inputs=[screen_b64, screen_prompt, screen_max_tok],
                    outputs=screen_output,
                    js=GRAB_FRAME_JS,
                )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LiveLM")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    _ensure_model(args.model)
    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
