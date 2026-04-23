"""
Minimal Streamlit UI: upload one image, pick model, see who-is-who against ``known_faces/`` folders.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from backends import list_backends
from core.pipeline import (
    MODEL_IDS,
    FaceResult,
    build_backend,
    identify,
    known_faces_tree_mtime,
    load_known_faces,
)

KNOWN_FACES_DIR = Path("known_faces")


def _model_choices() -> list[str]:
    if "insightface" in list_backends():
        return list(MODEL_IDS)
    return ["dlib"]


def _default_tolerance(model_id: str) -> float:
    return 0.6 if model_id == "dlib" else 0.5


@st.cache_resource
def _cached_backend(model_id: str, num_jitters: int) -> object:
    """Backend init tolerance is unused when callers pass tolerance into ``identify``."""
    return build_backend(
        model_id,
        tolerance=_default_tolerance(model_id),
        num_jitters=int(num_jitters),
    )


@st.cache_resource
def _cached_gallery(
    model_id: str,
    num_jitters: int,
    gallery_mtime: float,
) -> tuple[object, list, list[str], dict[str, int], list[str]]:
    warnings: list[str] = []

    def on_warn(msg: str) -> None:
        warnings.append(msg)

    backend = _cached_backend(model_id, num_jitters)
    emb, names, counts = load_known_faces(
        backend, KNOWN_FACES_DIR, on_warning=on_warn
    )
    return backend, emb, names, counts, warnings


def _annotate_image(
    rgb: np.ndarray, results: list[FaceResult], highlight_index: int
) -> Image.Image:
    img = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(img)
    for fr in results:
        b = fr.box
        color = (34, 197, 94) if fr.best.is_match else (239, 68, 68)
        w = 4 if fr.index == highlight_index else 2
        draw.rectangle(
            [b.left, b.top, b.right, b.bottom],
            outline=color,
            width=w,
        )
    return img


def main() -> None:
    st.set_page_config(page_title="Face ID", layout="wide")
    st.title("Face identification")
    st.caption(
        "Gallery is read-only from the ``known_faces/`` folder on disk (configure people there)."
    )

    with st.sidebar:
        st.header("Controls")
        choices = _model_choices()
        model_id = st.selectbox(
            "Model",
            choices,
            index=0,
            help="dlib = face_recognition; InsightFace packs download on first run.",
        )
        if model_id == "dlib":
            num_jitters = st.slider("Encoding jitters (dlib)", 1, 10, 1)
        else:
            num_jitters = 1
            st.caption("Jitters apply to dlib only.")

        tol_help = (
            "dlib: max L2 distance (lower = stricter). "
            "InsightFace: min cosine similarity 0–1 (higher = stricter)."
        )
        tolerance = st.slider(
            "Tolerance",
            min_value=0.0,
            max_value=1.0,
            value=float(_default_tolerance(model_id)),
            step=0.01,
            help=tol_help,
        )

        top_k = st.slider("Top-K candidates (persons)", 1, 5, 3)

        gallery_mtime = known_faces_tree_mtime(KNOWN_FACES_DIR)
        try:
            _b, emb, names, counts, warm = _cached_gallery(
                model_id, int(num_jitters), gallery_mtime
            )
        except Exception as e:
            st.error(f"Could not load gallery or model: {e}")
            st.stop()

        st.session_state["gallery_n"] = len(emb)
        st.session_state["gallery_people"] = len(counts)

        if warm:
            with st.expander("Gallery load warnings", expanded=False):
                for w in warm:
                    st.text(w)

        st.metric("Known embeddings", len(emb))
        st.metric("People (folders)", len(counts))
        if counts:
            st.caption(
                "Counts per folder: "
                + ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
            )

    uploaded = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
    )

    rgb: np.ndarray | None = None
    if uploaded is not None:
        data = uploaded.read()
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        rgb = np.asarray(pil)
        st.session_state["upload_rgb"] = rgb
    elif st.session_state.get("upload_rgb") is not None:
        rgb = st.session_state["upload_rgb"]

    run = st.button("Identify", type="primary")

    if run:
        if rgb is None:
            st.warning("Please upload an image first.")
        else:
            backend, emb, names, counts, _warm = _cached_gallery(
                model_id, int(num_jitters), gallery_mtime
            )
            if not emb:
                st.error(
                    f"No embeddings loaded from {KNOWN_FACES_DIR.resolve()}. "
                    "Add subfolders with reference photos."
                )
            else:
                t0 = time.perf_counter()
                progress = st.progress(0, text="Preparing...")
                try:

                    def _on_progress(p: float, msg: str) -> None:
                        progress.progress(min(1.0, max(0.0, p)), text=msg)

                    results = identify(
                        rgb,
                        backend,
                        emb,
                        names,
                        float(tolerance),
                        top_k=int(top_k),
                        on_progress=_on_progress,
                    )
                except Exception as e:
                    progress.empty()
                    st.exception(e)
                    results = None
                else:
                    progress.empty()
                if results is not None:
                    wall = time.perf_counter() - t0
                    st.session_state["last_results"] = results
                    st.session_state["last_rgb"] = rgb
                    st.session_state["last_model"] = model_id
                    st.session_state["last_wall"] = wall
                    st.session_state["last_tolerance"] = float(tolerance)

    results = st.session_state.get("last_results")
    last_rgb = st.session_state.get("last_rgb")
    if results is None or last_rgb is None:
        st.info("Upload an image and click **Identify**.")
        return

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Model", st.session_state.get("last_model", model_id))
    with col_b:
        st.metric("Faces detected", len(results))
    with col_c:
        st.metric("Gallery embeddings", st.session_state.get("gallery_n", 0))
    with col_d:
        st.metric("Wall time (s)", f"{st.session_state.get('last_wall', 0):.3f}")

    st.divider()

    disp_tol = st.session_state.get("last_tolerance", tolerance)

    for fr in results:
        st.subheader(f"Face #{fr.index}")
        badge = "KNOWN" if fr.best.is_match else "UNKNOWN"
        st.markdown(f"**{badge}** — best label: **{fr.best.best_name}**")

        ann = _annotate_image(last_rgb, results, fr.index)
        c1, c2 = st.columns([2, 1])
        with c1:
            st.image(
                ann,
                caption="Full image (highlighted box)",
                use_container_width=True,
            )
        with c2:
            st.image(fr.crop, caption="Face crop", use_container_width=True)

        st.progress(
            min(100, max(0, int(fr.best.match_pct))) / 100.0,
            text=f"Display match %: {fr.best.match_pct:.1f}%",
        )
        if fr.best.score_kind == "l2_distance":
            st.caption(
                f"L2 distance: {fr.best.raw_score:.4f} (lower is better) · threshold ≤ {disp_tol}"
            )
        else:
            st.caption(
                f"Cosine similarity: {fr.best.raw_score:.4f} (higher is better) · "
                f"threshold ≥ {disp_tol}"
            )
        st.caption(
            f"Timings — detect: {fr.timings.detect_sec:.3f}s, "
            f"encode: {fr.timings.encode_sec:.3f}s, "
            f"match: {fr.timings.match_sec:.3f}s, "
            f"sum: {fr.timings.total_sec:.3f}s · embedding dim: {fr.embedding_dim}"
        )

        with st.expander("Top-K candidates"):
            rows = []
            for rank, c in enumerate(fr.candidates, start=1):
                rows.append(
                    {
                        "rank": rank,
                        "name": c.best_name,
                        "raw": round(c.raw_score, 6),
                        "match_pct": round(c.match_pct, 2),
                        "is_match": c.is_match,
                    }
                )
            st.dataframe(rows, use_container_width=True)

        dbg = {
            "model_id": st.session_state.get("last_model", model_id),
            "embedding_dim": fr.embedding_dim,
            "box": {
                "top": fr.box.top,
                "right": fr.box.right,
                "bottom": fr.box.bottom,
                "left": fr.box.left,
            },
            "score_kind": fr.best.score_kind,
        }
        with st.expander("Debug JSON"):
            st.code(json.dumps(dbg, indent=2), language="json")


if __name__ == "__main__":
    main()
