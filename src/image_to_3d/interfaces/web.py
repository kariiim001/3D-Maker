"""Gradio web UI (imported lazily so the command line works without gradio)."""
from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from ..config import DEFAULT_REMBG_MODEL, INSTALL_HINT, QUALITY_LEVELS
from ..errors import ImageTo3DError, JobCancelled
from ..pipeline import convert_image_to_3d
from ..utils import short


def build_ui(output_dir: Path):
    import gradio as gr

    skip = getattr(gr, "skip", None) or gr.update  # "leave this output unchanged"
    provider_choices = [
        ("Auto - try several Spaces, fall back if one fails", "auto"),
        ("TRELLIS - fast, good textures", "trellis"),
        ("TRELLIS.2 - highest quality, heavier on GPU quota", "trellis2"),
        ("Hunyuan3D - Tencent", "hunyuan3d"),
    ]

    def generate(image_path, provider, remove_bg, quality, out_format, textured, randomize, seed, rembg_model, token):
        """
        Generator: streams progress lines into the log box.  The finished file paths go into `result_state`
        and are displayed by show_result() in a separate chained event: Gradio's Model3D viewer does not
        draw a model that is delivered straight from a streaming generator (verified with Gradio 6.31).
        """
        if not image_path:
            yield "Please upload a photo first.", None
            return
        yield "Starting ...", None  # also clears any previous result from result_state

        events: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        cancel = threading.Event()
        lines: list[str] = []
        t0 = time.monotonic()

        def stamp(msg: str) -> str:
            return f"[{time.monotonic() - t0:4.0f}s] {msg}"

        def worker() -> None:
            try:
                result = convert_image_to_3d(
                    image_path, provider=provider, remove_bg=bool(remove_bg), rembg_model=rembg_model or DEFAULT_REMBG_MODEL,
                    quality=quality, seed=None if randomize else int(seed or 0), textured=bool(textured),
                    out_format=out_format, token=(token or "").strip() or None, output_dir=output_dir,
                    on_status=lambda m: events.put(("log", m)), cancel_event=cancel)
                events.put(("done", result))
            except BaseException as exc:  # reported to the user below, never lost
                events.put(("error", exc))

        threading.Thread(target=worker, daemon=True, name="image-to-3d-worker").start()
        try:
            while True:
                try:
                    kind, payload = events.get(timeout=1.0)
                except queue.Empty:
                    yield skip(), skip()  # heartbeat: lets Gradio notice the Stop button
                    continue
                if kind == "log":
                    lines.append(stamp(payload))
                    yield "\n".join(lines), skip()
                elif kind == "done":
                    lines.append(stamp(f"Done in {payload.seconds:.0f}s with {payload.space_id} "
                                       f"(seed {payload.seed}). Saved to {payload.files[0]}"))
                    yield "\n".join(lines), {
                        "preview": str(payload.preview),
                        "files": [str(f) for f in payload.files],
                        "cutout": str(payload.cutout) if payload.cutout else None,
                    }
                    return
                else:
                    if isinstance(payload, JobCancelled):
                        lines.append(stamp("Cancelled."))
                    elif isinstance(payload, ImageTo3DError):
                        lines.append(stamp(f"ERROR: {payload}"))
                    else:
                        lines.append(stamp(f"UNEXPECTED ERROR: {type(payload).__name__}: {short(payload, 300)}"))
                    yield "\n".join(lines), None
                    return
        finally:
            cancel.set()  # no-op when finished; aborts the remote job if the user pressed Stop / closed the tab

    def show_result(res):
        """Plain (non-generator) function: fills the 3D viewer, the download box and the cut-out preview."""
        if not res:
            return None, None, None
        return res["preview"], res["files"], res["cutout"]

    with gr.Blocks(title="Image to 3D") as demo:
        gr.Markdown(
            "# Image to 3D model\n"
            "Upload a photo, press **Generate**. The heavy lifting runs on free Hugging Face Spaces, "
            "so this works without a GPU. Typical time: 1-3 minutes (longer if the queue is busy)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(label="Photo", type="filepath", height=320)
                provider_in = gr.Dropdown(provider_choices, value="auto", label="Backend")
                with gr.Row():
                    quality_in = gr.Radio(list(QUALITY_LEVELS), value="balanced", label="Quality")
                    format_in = gr.Radio(["glb", "obj"], value="glb", label="Download format")
                remove_bg_in = gr.Checkbox(True, label="Remove background first (recommended)")
                with gr.Accordion("Advanced", open=False):
                    textured_in = gr.Checkbox(True, label="Texture the model (Hunyuan3D only; slower)")
                    randomize_in = gr.Checkbox(True, label="Random seed")
                    seed_in = gr.Number(0, precision=0, label="Seed (used when 'Random seed' is off)")
                    rembg_in = gr.Dropdown(["silueta", "u2netp", "u2net", "isnet-general-use"],
                                           value=DEFAULT_REMBG_MODEL, label="Background-removal model",
                                           allow_custom_value=True)
                    token_in = gr.Textbox(label="Hugging Face token (optional)", type="password",
                                          placeholder="hf_...   (or set the HF_TOKEN environment variable)")
                with gr.Row():
                    run_btn = gr.Button("Generate 3D model", variant="primary")
                    stop_btn = gr.Button("Stop")
            with gr.Column(scale=1):
                model_out = gr.Model3D(label="Preview (drag to rotate)", height=420)
                files_out = gr.File(label="Download", file_count="multiple", interactive=False)
                cutout_out = gr.Image(label="Background removed", interactive=False, height=180)
                status_out = gr.Textbox(label="Progress", lines=9, max_lines=14, interactive=False, autoscroll=True)

        result_state = gr.State()
        clear_event = run_btn.click(  # empty the viewer / download boxes so every run starts clean
            lambda: (None, None, None), inputs=None, outputs=[model_out, files_out, cutout_out], queue=False)
        run_event = clear_event.then(
            generate,
            inputs=[image_in, provider_in, remove_bg_in, quality_in, format_in, textured_in, randomize_in,
                    seed_in, rembg_in, token_in],
            outputs=[status_out, result_state],
        )
        show_event = run_event.then(show_result, inputs=[result_state], outputs=[model_out, files_out, cutout_out])
        stop_btn.click(  # cancelling the click handler also aborts the remote job (see the `finally` in generate())
            lambda log: (log or "") + "\n[stopped by user - the remote job is being cancelled]",
            inputs=[status_out], outputs=[status_out], cancels=[run_event, show_event])
    return demo


def launch_ui(output_dir: Path, host: str, port: Optional[int], open_browser: bool, share: bool) -> None:
    try:
        import gradio  # noqa: F401
    except ImportError:
        sys.exit(f"The web UI needs gradio.\n  {INSTALL_HINT}")
    output_dir.mkdir(parents=True, exist_ok=True)
    demo = build_ui(output_dir)
    demo.queue()  # one generation at a time per click handler: be polite to the free Spaces
    demo.launch(server_name=host, server_port=port, inbrowser=open_browser, share=share,
                allowed_paths=[str(output_dir)], show_error=True)


