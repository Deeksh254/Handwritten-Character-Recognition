"""Handwritten Character Recognition - Web App

Usage:
    python app.py              # http://127.0.0.1:7860
    python app.py --share      # + temporary public link

Models are loaded from runs/<name>/best.pt automatically.
"""
import argparse
import os
import traceback
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

from hcr.crnn import CRNN, greedy_decode, line_image_to_tensor
from hcr.htr import detect_text_lines, image_quality_warnings
from hcr.preprocess import single_box, to_binary
from hcr.utils import get_device

# ──────────────────────────────────────────────────────────────────────────────
# Global state
# ──────────────────────────────────────────────────────────────────────────────
DEVICE = get_device()
_checkpoint_setting = Path(os.environ.get("HCR_CRNN_CHECKPOINT", "runs/crnn/best.pt"))
CRNN_CHECKPOINT = _checkpoint_setting if _checkpoint_setting.is_absolute() else ROOT / _checkpoint_setting
_CRNN_CACHE = None


def get_crnn():
    global _CRNN_CACHE
    if _CRNN_CACHE is None:
        checkpoint = torch.load(CRNN_CHECKPOINT, map_location=DEVICE, weights_only=False)
        classes = checkpoint["classes"]
        model = CRNN(len(classes), rnn_type=checkpoint.get("rnn_type", "lstm")).to(DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        _CRNN_CACHE = (model, classes)
    return _CRNN_CACHE


# ──────────────────────────────────────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────────────────────────────────────

def coerce_image(image) -> np.ndarray:
    """Accept a numpy array or Gradio Sketchpad/ImageEditor dict."""
    if image is None:
        return None
    if isinstance(image, dict):
        for key in ("composite", "image", "background"):
            candidate = image.get(key)
            if candidate is not None:
                image = candidate
                break
    return image


def numpy_to_gray(image: np.ndarray, max_side: int = 1600) -> np.ndarray:
    """Convert uploaded numpy (RGB/RGBA) → grayscale, optionally resize."""
    image = coerce_image(image)
    if image is None:
        return None
    image = np.asarray(image)
    if image.ndim == 3:
        if image.shape[2] == 4:          # RGBA → composite on white
            alpha = image[:, :, 3:4] / 255.0
            image = (image[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    h, w = gray.shape
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        gray = cv2.resize(gray, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return gray


# ──────────────────────────────────────────────────────────────────────────────
# Core recognition function
# ──────────────────────────────────────────────────────────────────────────────

def recognise(
    image: np.ndarray,
    sketch,
    mode: str,
    invert: bool,
    dilate_val: int,
):
    """Main recognition pipeline — called by the Gradio button."""
    image = coerce_image(image)
    if image is None:
        image = coerce_image(sketch)
    # ── Validation ─────────────────────────────────────────────────────────
    if image is None:
        return None, "", "⚠️ Please upload or draw an image first."
    if not CRNN_CHECKPOINT.exists():
        return None, "", (
            f"CRNN checkpoint not found: {CRNN_CHECKPOINT}\n"
            "Train the unified recognizer with:\n"
            "python -m hcr.crnn --dataset balanced --rnn gru --epochs 10 --out-dir runs/crnn"
        )
    try:
        gray   = numpy_to_gray(image)
        mode_key = mode.lower()
        model, classes = get_crnn()

        if mode_key == "single character":
            binary = to_binary(gray, invert=invert, dilate=dilate_val)
            boxes = single_box(binary)
            if not boxes:
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), "", "No character detected."
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            x, y, width, height = boxes[0]
            crop = gray[y:y + height, x:x + width]
            input_tensor = line_image_to_tensor(
                crop, dilate=dilate_val, invert=invert
            ).to(DEVICE)
            with torch.inference_mode():
                text = greedy_decode(model(input_tensor), classes)[0]
            cv2.rectangle(vis, (x, y), (x + width, y + height), (0, 210, 0), 2)
            cv2.putText(vis, text or "?", (x, max(16, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 220), 2, cv2.LINE_AA)
            note = "Single character decoded by the CRNN (CNN + BiGRU/LSTM + CTC)."
            if not text:
                note += " The model emitted only CTC blanks; retraining or a clearer sample may help."
            return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), text, note

        page, line_boxes = detect_text_lines(gray, invert=invert)
        if not line_boxes:
            return cv2.cvtColor(page, cv2.COLOR_GRAY2RGB), "", "No text lines detected."

        vis = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
        line_texts, note_lines = [], [
            f"CRNN line recognizer: {len(line_boxes)} line(s), "
            f"{model.rnn_type.upper()} + CTC."
        ]
        note_lines.extend(image_quality_warnings(gray))
        with torch.inference_mode():
            for index, (x, y, width, height) in enumerate(line_boxes, 1):
                crop = page[y:y + height, x:x + width]
                input_tensor = line_image_to_tensor(crop, dilate=dilate_val).to(DEVICE)
                line_text = greedy_decode(model(input_tensor), classes)[0]
                line_texts.append(line_text)
                cv2.rectangle(vis, (x, y), (x + width, y + height), (0, 210, 0), 2)
                cv2.putText(vis, f"Line {index}", (x, max(16, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 220), 2, cv2.LINE_AA)
                if not line_text:
                    note_lines.append(f"Line {index}: CRNN emitted only blanks.")
        text = "\n".join(line_texts)
        return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), text, "\n".join(note_lines)

    except Exception as exc:
        tb = traceback.format_exc()
        err_msg = f"❌ Error: {exc}\n\nDetails:\n{tb}"
        return None, "", err_msg


# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS  — premium dark glassmorphism theme
# ──────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

body, .gradio-container {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    min-height: 100vh;
}

/* ── Header / title ── */
.app-header {
    text-align: center;
    padding: 2rem 1rem 1rem;
}
.app-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    margin-bottom: 0.4rem;
}
.app-subtitle {
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 400;
}

/* ── Cards / glass panels ── */
.glass-card {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    padding: 1.25rem !important;
    transition: border-color 0.2s;
}
.glass-card:hover { border-color: rgba(167,139,250,0.35) !important; }

/* ── Labels & text ── */
label, .label-wrap span, .block .label-wrap {
    color: #cbd5e1 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

/* ── Textbox / output ── */
textarea, input[type=text] {
    background: rgba(15,12,41,0.7) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
}

/* ── Image upload area ── */
.upload-container, .image-frame {
    background: rgba(255,255,255,0.04) !important;
    border: 2px dashed rgba(167,139,250,0.4) !important;
    border-radius: 12px !important;
}

/* ── Primary button — gradient glow ── */
.gr-button-primary, button.primary {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.45) !important;
    width: 100% !important;
}
.gr-button-primary:hover, button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124,58,237,0.65) !important;
}
.gr-button-primary:active, button.primary:active {
    transform: translateY(0) !important;
}

/* ── Secondary button ── */
button.secondary {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: #cbd5e1 !important;
    border-radius: 10px !important;
    transition: all 0.2s !important;
}
button.secondary:hover {
    background: rgba(255,255,255,0.14) !important;
    color: #fff !important;
}

/* ── Radio buttons ── */
.gr-radio-group label {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    padding: 0.4rem 0.9rem !important;
    color: #94a3b8 !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    text-transform: none !important;
    font-size: 0.88rem !important;
}
.gr-radio-group label:has(input:checked) {
    background: rgba(124,58,237,0.3) !important;
    border-color: #7c3aed !important;
    color: #c4b5fd !important;
}

/* ── Dropdown ── */
.gr-dropdown > div {
    background: rgba(15,12,41,0.8) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── Slider ── */
.gr-slider input[type=range] { accent-color: #7c3aed !important; }

/* ── Accordion ── */
.gr-accordion {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 10px !important;
}
.gr-accordion summary {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
}

/* ── Stats badge ── */
.stat-badge {
    display: inline-block;
    background: rgba(124,58,237,0.2);
    border: 1px solid rgba(167,139,250,0.4);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    color: #c4b5fd;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.2rem;
}

/* ── Tabs ── */
.tab-nav button {
    color: #64748b !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s !important;
    background: transparent !important;
}
.tab-nav button.selected {
    color: #a78bfa !important;
    border-bottom-color: #7c3aed !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.03); }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.4); border-radius: 3px; }

/* ── Notification / note output ── */
.note-output textarea {
    font-size: 0.85rem !important;
    color: #94a3b8 !important;
    line-height: 1.6 !important;
}
.prediction-output textarea {
    font-size: 1.15rem !important;
    color: #e2e8f0 !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
    line-height: 1.8 !important;
}
"""

# ──────────────────────────────────────────────────────────────────────────────
# UI builder
# ──────────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="HCR — Handwritten Character Recognition",
    ) as demo:

        # ── Header ─────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="app-header">
            <div class="app-title">✍️ Handwritten Character Recognition</div>
            <div class="app-subtitle">
                Upload a photo · Choose a model · Instantly recognise letters, digits &amp; sentences
            </div>
        </div>
        """)

        # ── CRNN checkpoint status ─────────────────────────────────────────
        if CRNN_CHECKPOINT.exists():
            gr.HTML(f'<div style="text-align:center;margin-bottom:1rem">'
                    f'Unified recognizer: CNN + bidirectional recurrent layer + CTC</div>')
        else:
            gr.HTML("""
            <div style="
                text-align:center; margin-bottom:1rem;
                background:rgba(239,68,68,0.15);
                border:1px solid rgba(239,68,68,0.4);
                border-radius:12px; padding:1rem;
                color:#fca5a5;
            ">
                <b>⚠️ CRNN checkpoint not found.</b><br>
                <span style="font-size:0.88rem;color:#94a3b8">
                    Train the unified model with <code style="color:#a78bfa">python -m hcr.crnn --dataset balanced --rnn gru --epochs 10 --out-dir runs/crnn</code><br>
                    then restart the app.
                </span>
            </div>
            """)

        # ── Main layout ─────────────────────────────────────────────────────
        with gr.Row(equal_height=True):

            # ── Left panel — inputs ────────────────────────────────────────
            with gr.Column(scale=5):
                with gr.Group(elem_classes="glass-card"):
                    with gr.Tabs():
                        with gr.Tab("📷 Upload"):
                            img_input = gr.Image(
                                label="Upload Handwriting",
                                type="numpy",
                                sources=["upload", "webcam", "clipboard"],
                                height=320,
                            )
                        with gr.Tab("✏️ Draw"):
                            try:
                                sketch_input = gr.Sketchpad(
                                    label="Draw a character or word",
                                    type="numpy",
                                    height=320,
                                    layers=False,
                                    canvas_size=(320, 320),
                                )
                            except (TypeError, AttributeError):
                                sketch_input = gr.Image(
                                    label="Draw a character or word",
                                    type="numpy",
                                    sources=["upload", "clipboard"],
                                    height=320,
                                )

                with gr.Group(elem_classes="glass-card"):
                    mode_radio = gr.Radio(
                        choices=["Single Character", "Word / Line", "Paragraph / Page"],
                        value="Single Character",
                        label="📝 Recognition Mode",
                        info="All modes use the same CRNN checkpoint.",
                    )

                with gr.Accordion("⚙️ Advanced Options", open=False):
                    invert_cb = gr.Checkbox(
                        label="Invert (light ink on dark background)",
                        value=False,
                    )
                    dilate_sl = gr.Slider(
                        minimum=0, maximum=3, value=1, step=1,
                        label="Stroke thickening (higher = thicker strokes)",
                        info="Increase if thin strokes are missed",
                    )

                run_btn = gr.Button(
                    "🔍  Recognise",
                    variant="primary",
                    size="lg",
                )
                clear_btn = gr.Button("🗑️  Clear", variant="secondary", size="sm")

            # ── Right panel — outputs ──────────────────────────────────────
            with gr.Column(scale=6):
                with gr.Tabs():
                    with gr.Tab("📸 Annotated Image"):
                        vis_output = gr.Image(
                            label="Detected Characters",
                            type="numpy",
                            height=320,
                            interactive=False,
                        )
                    with gr.Tab("📄 Text Output"):
                        txt_output = gr.Textbox(
                            label="Recognised Text",
                            lines=10,
                            placeholder="Recognised text will appear here...",
                            elem_classes="prediction-output",
                        )
                    with gr.Tab("ℹ️ Details"):
                        note_output = gr.Textbox(
                            label="Recognition Notes",
                            lines=8,
                            interactive=False,
                            placeholder="Status and confidence notes...",
                            elem_classes="note-output",
                        )

        # ── Tips row ────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="
            display:flex; gap:1rem; flex-wrap:wrap; justify-content:center;
            margin-top:1.5rem; padding:1rem;
            background:rgba(255,255,255,0.04);
            border:1px solid rgba(255,255,255,0.1);
            border-radius:14px;
        ">
          <div style="color:#94a3b8;font-size:0.82rem;text-align:center;flex:1;min-width:160px">
            <div style="font-size:1.5rem;margin-bottom:4px">💡</div>
            <b style="color:#cbd5e1">Best results</b><br>Dark ink on white paper
          </div>
          <div style="color:#94a3b8;font-size:0.82rem;text-align:center;flex:1;min-width:160px">
                        <div style="font-size:1.5rem;margin-bottom:4px">✍️</div>
                        <b style="color:#cbd5e1">Text modes</b><br>Write full lines naturally
          </div>
          <div style="color:#94a3b8;font-size:0.82rem;text-align:center;flex:1;min-width:160px">
            <div style="font-size:1.5rem;margin-bottom:4px">🔢</div>
            <b style="color:#cbd5e1">Numbers + Letters</b><br>Use <em>byclass</em> or <em>balanced</em> model
          </div>
          <div style="color:#94a3b8;font-size:0.82rem;text-align:center;flex:1;min-width:160px">
            <div style="font-size:1.5rem;margin-bottom:4px">🎥</div>
            <b style="color:#cbd5e1">Capture</b><br>Keep page in frame; hold steady
          </div>
        </div>
        """)

        # ── Wiring ──────────────────────────────────────────────────────────
        inputs  = [img_input, sketch_input, mode_radio, invert_cb, dilate_sl]
        outputs = [vis_output, txt_output, note_output]

        run_btn.click(fn=recognise, inputs=inputs, outputs=outputs)

        # Also run on image upload for convenience
        img_input.upload(fn=recognise, inputs=inputs, outputs=outputs)

        def _clear():
            return None, None, None, "", ""

        clear_btn.click(
            fn=_clear,
            outputs=[img_input, sketch_input, vis_output, txt_output, note_output],
        )
        img_input.clear(
            fn=lambda: (None, "", ""),
            outputs=[vis_output, txt_output, note_output],
        )

    return demo


# ──────────────────────────────────────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--share",  action="store_true", help="Create a public link")
    p.add_argument("--port",   type=int, default=7860)
    p.add_argument("--host",   default="127.0.0.1")
    args = p.parse_args()

    if not CRNN_CHECKPOINT.exists():
        print(
            f"CRNN checkpoint not found: {CRNN_CHECKPOINT}\n"
            "Train the app recognizer with:\n"
            "  python -m hcr.crnn --dataset balanced --rnn gru --epochs 10 --out-dir runs/crnn\n"
            "The app will launch, but predictions will be unavailable until training completes."
        )

    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print(f"[OK] CRNN checkpoint: {CRNN_CHECKPOINT}")
    print(f"[>>] Launching on http://{args.host}:{args.port}")

    build_ui().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Base(
            primary_hue="violet",
            secondary_hue="blue",
            neutral_hue="slate",
        ),
        css=CUSTOM_CSS,
    )
