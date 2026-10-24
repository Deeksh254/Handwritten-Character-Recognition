"""Handwritten Character Recognition - Web App

Usage:
    python app.py              # http://127.0.0.1:7860
    python app.py --share      # + temporary public link

Models are loaded from runs/<name>/best.pt automatically.
"""
import argparse
import os
import string
import traceback
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

from hcr.predict import (
    classify_topk, load_model,
    draw_details,
)
from hcr.preprocess import box_to_mnist, single_box, to_binary
from hcr.htr import DEFAULT_MODEL_ID, image_quality_warnings, recognize_handwriting
from hcr.utils import get_device

# ──────────────────────────────────────────────────────────────────────────────
# Global state
# ──────────────────────────────────────────────────────────────────────────────
DEVICE = get_device()
HTR_MODEL_ID = DEFAULT_MODEL_ID
_MODEL_CACHE: dict = {}

FRIENDLY_NAMES = {
    "byclass":  "All Characters (0-9, A-Z, a-z) · 62 classes",
    "balanced": "Balanced (0-9, A-Z, common) · 47 classes",
    "letters":  "Letters Only (A-Z) · 26 classes",
    "digits":   "Digits Only (0-9) · 10 classes",
    "mnist":    "MNIST Digits (0-9) · 10 classes",
}


def find_models() -> dict:
    """Scan bundled and local model directories for trained checkpoints."""
    found = {}
    checkpoint_paths = sorted((ROOT / "samples" / "runs").glob("*/best.pt"))
    checkpoint_paths += sorted((ROOT / "runs").glob("*/best.pt"))
    for p in checkpoint_paths:
        name = p.parent.name
        if name in ("crnn", "smoke_test"):
            continue
        label = FRIENDLY_NAMES.get(name, name)
        found[label] = str(p)
    models_dir = ROOT / "models"
    if models_dir.exists():
        for p in sorted(models_dir.glob("*.pt")):
            found[p.stem] = str(p)
    return found


MODELS = find_models()
MODEL_LABELS = list(MODELS.keys())
DEFAULT_MODEL = next(
    (label for label, path in MODELS.items() if Path(path).parent.name == "byclass"),
    next(
        (label for label, path in MODELS.items() if Path(path).parent.name == "balanced"),
        MODEL_LABELS[0] if MODEL_LABELS else None,
    ),
)


def get_model(label: str):
    if label not in _MODEL_CACHE:
        _MODEL_CACHE[label] = load_model(MODELS[label], DEVICE)
    return _MODEL_CACHE[label]


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
    model_label: str,
    mode: str,
    invert: bool,
    dilate_val: int,
    allowed_charset: str = "Any class",
):
    """Main recognition pipeline — called by the Gradio button."""
    image = coerce_image(image)
    if image is None:
        image = coerce_image(sketch)
    # ── Validation ─────────────────────────────────────────────────────────
    if image is None:
        return None, "", "⚠️ Please upload or draw an image first."
    try:
        gray   = numpy_to_gray(image)
        mode_key = mode.lower()

        if mode_key == "single character":
            if not MODELS:
                return None, "", "No single-character checkpoints found."
            if not model_label or model_label not in MODELS:
                return None, "", "Select a model for single-character recognition."
            model, classes = get_model(model_label)
            binary = to_binary(gray, invert=invert, dilate=dilate_val)
            boxes = single_box(binary)
            if not boxes:
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), "", "No character detected."
            allowed_labels = {
                "Digits only": set(string.digits),
                "Uppercase only": set(string.ascii_uppercase),
                "Letters and digits": set(string.ascii_letters + string.digits),
            }.get(allowed_charset)
            top = classify_topk(
                model, box_to_mnist(binary, boxes[0]), DEVICE, classes,
                allowed_labels=allowed_labels,
            )
            text = classes[top[0][0]]
            details = [(text, top[0][1], boxes[0])]
            if not details:
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), "", "No character detected."

            char_box = details[0][2]
            alternatives = ", ".join(f"{classes[index]} {score:.0%}" for index, score in top)
            scores_are_calibrated = model.is_temperature_calibrated and allowed_labels is None
            score_label = (
                "Top guesses (temperature-scaled probabilities)"
                if scores_are_calibrated
                else "Top guesses (raw, uncalibrated scores)"
            )
            note_lines = [
                f"Character model: {model_label.split('·')[0].strip()}",
                f"{score_label}: {alternatives}",
            ]
            if not scores_are_calibrated:
                note_lines.append("Scores are not calibrated confidence estimates.")
            elif top[0][1] < model.review_threshold:
                note_lines.append("Review needed: top calibrated probability is below the held-out threshold.")
            if allowed_labels is not None:
                note_lines.append(f"Alphabet constraint: {allowed_charset}.")
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            draw_details(vis, details, font_scale=max(0.5, gray.shape[0] / 900))
            return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), text, "\n".join(note_lines)

        page, line_results = recognize_handwriting(
            gray, invert=invert, model_id=HTR_MODEL_ID
        )
        if not line_results:
            return cv2.cvtColor(page, cv2.COLOR_GRAY2RGB), "", "No text lines detected."

        text = "\n".join(primary for primary, _, _ in line_results)
        vis = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
        note_lines = [f"TrOCR line recognizer: {len(line_results)} line(s)."]
        note_lines.extend(image_quality_warnings(gray))
        for index, (primary, hypotheses, (x, y, width, height)) in enumerate(line_results, 1):
            cv2.rectangle(vis, (x, y), (x + width, y + height), (0, 210, 0), 2)
            cv2.putText(vis, f"Line {index}", (x, max(16, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 220), 2, cv2.LINE_AA)
            alternatives = " | ".join(hypotheses[:3])
            note_lines.append(f"Line {index} alternatives (beam order, not confidence): {alternatives}")
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

        # ── Model info banner ───────────────────────────────────────────────
        if MODEL_LABELS:
            badges = "".join(
                f'<span class="stat-badge">{lbl.split("·")[0].strip()}</span>'
                for lbl in MODEL_LABELS
            )
            gr.HTML(f'<div style="text-align:center;margin-bottom:1rem">'
                    f'Available models: {badges}</div>')
        else:
            gr.HTML("""
            <div style="
                text-align:center; margin-bottom:1rem;
                background:rgba(239,68,68,0.15);
                border:1px solid rgba(239,68,68,0.4);
                border-radius:12px; padding:1rem;
                color:#fca5a5;
            ">
                <b>⚠️ No trained models found.</b><br>
                <span style="font-size:0.88rem;color:#94a3b8">
                    Run in a terminal: <code style="color:#a78bfa">python -m hcr.train --dataset byclass --epochs 25</code><br>
                    then restart the app once training completes.
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
                    model_dd = gr.Dropdown(
                        choices=MODEL_LABELS,
                        value=DEFAULT_MODEL,
                        label="🤖 Single-Character Model",
                        info="Used only for isolated-character recognition",
                    )
                    mode_radio = gr.Radio(
                        choices=["Single Character", "Word / Line", "Paragraph / Page"],
                        value="Single Character",
                        label="📝 Recognition Mode",
                        info="Text modes use TrOCR and download about 1.33 GB on first use.",
                    )
                    charset_dd = gr.Dropdown(
                        choices=["Any class", "Digits only", "Uppercase only", "Letters and digits"],
                        value="Any class",
                        label="Single-Character Alphabet",
                        info="Applied only in Single Character mode",
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
        inputs  = [img_input, sketch_input, model_dd, mode_radio, invert_cb, dilate_sl, charset_dd]
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

    if not MODELS:
        print(
            "⚠️  No trained models found yet.\n"
            "Train one with:\n"
            "  python -m hcr.train --dataset byclass --epochs 25\n"
            "  python -m hcr.train --dataset letters --epochs 15\n"
            "The app will launch anyway — restart it after training to load a model."
        )

    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print(f"[OK] Found {len(MODELS)} model(s): {', '.join(MODEL_LABELS)}")
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
