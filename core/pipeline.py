"""
Face identification pipeline: build backend, load gallery, run identify on RGB images.
Used by ``face_identifier.py`` CLI and ``app.py`` Streamlit UI.
"""

from __future__ import annotations  # noqa: I001 — enables ``|`` unions on Python 3.9

import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
from PIL import Image

from backends import get_backend

if TYPE_CHECKING:
    from backends.base import FaceBackend

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Flat model ids for UI / ``build_backend``
MODEL_IDS = ("dlib", "insightface:buffalo_l", "insightface:buffalo_sc")


def known_faces_tree_mtime(known_dir: Path) -> float:
    """Latest mtime under ``known_dir`` (files only); 0 if missing."""
    if not known_dir.exists():
        return 0.0
    mt = known_dir.stat().st_mtime
    for p in known_dir.rglob("*"):
        if p.is_file():
            try:
                mt = max(mt, p.stat().st_mtime)
            except OSError:
                continue
    return mt


@dataclass
class FaceIdentifyTimings:
    """Seconds spent in each phase (shared detect/encode across faces; match is per face)."""

    detect_sec: float
    encode_sec: float
    match_sec: float
    total_sec: float


@dataclass
class FaceResult:
    """One detected face and how it matches the gallery."""

    index: int
    box: FaceBox
    best: MatchResult
    candidates: list[MatchResult]
    embedding_dim: int
    crop: np.ndarray
    timings: FaceIdentifyTimings


def build_backend(
    model_id: str,
    *,
    tolerance: float,
    num_jitters: int = 1,
) -> FaceBackend:
    """
    ``model_id``: ``dlib`` | ``insightface:buffalo_l`` | ``insightface:buffalo_sc``.
    """
    mid = model_id.strip().lower()
    if mid == "dlib":
        return get_backend("dlib", threshold=tolerance, num_jitters=num_jitters)
    if mid == "insightface:buffalo_l":
        return get_backend("insightface", threshold=tolerance, model_pack="buffalo_l")
    if mid == "insightface:buffalo_sc":
        return get_backend("insightface", threshold=tolerance, model_pack="buffalo_sc")
    raise ValueError(
        f"Unknown model_id {model_id!r}. Expected one of: {', '.join(MODEL_IDS)}"
    )


def load_known_faces(
    backend: FaceBackend,
    known_dir: Path,
    *,
    on_warning: Callable[[str], None] | None = None,
) -> tuple[list[np.ndarray], list[str], dict[str, int]]:
    """
    Load embeddings from ``known_dir/<PersonName>/*`` image files.

    Returns ``(embeddings, names_parallel_to_embeddings, counts_per_person)``.
    Skips images that fail to load. If ``on_warning`` is set, it receives a message per failure;
    if ``on_warning`` is ``None``, failures are skipped silently.
    """
    known_embeddings: list[np.ndarray] = []
    known_names: list[str] = []

    if not known_dir.exists():
        return known_embeddings, known_names, {}

    detect_encode = getattr(backend, "detect_and_encode", None)

    for person_dir in sorted(known_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                image = backend.load_image(img_path)
                if callable(detect_encode):
                    pairs = detect_encode(image)
                    for _box, enc in pairs:
                        known_embeddings.append(enc)
                        known_names.append(name)
                else:
                    boxes = backend.detect_faces(image)
                    if not boxes:
                        continue
                    encodings = backend.encode_faces(image, boxes)
                    for enc in encodings:
                        known_embeddings.append(enc)
                        known_names.append(name)
            except Exception as e:
                if on_warning:
                    on_warning(f"could not load {img_path}: {e}")

    counts = dict(Counter(known_names))
    return known_embeddings, known_names, counts


def _crop_face(image: np.ndarray, box: FaceBox) -> np.ndarray:
    pil = Image.fromarray(image)
    crop = pil.crop((box.left, box.top, box.right, box.bottom))
    return np.asarray(crop)


def identify(
    image: np.ndarray,
    backend: FaceBackend,
    known_embeddings: list[np.ndarray],
    known_names: list[str],
    threshold: float,
    top_k: int = 3,
    *,
    on_progress: Callable[[float, str], None] | None = None,
) -> list[FaceResult]:
    """
    Detect faces in RGB ``image``, match against gallery, return structured results (no disk I/O).

    If ``on_progress`` is set, it is called with ``(fraction_0_to_1, status_message)`` during work.
    """
    def _prog(p: float, msg: str) -> None:
        if on_progress is not None:
            on_progress(p, msg)

    _prog(0.05, "Starting detection...")
    detect_encode = getattr(backend, "detect_and_encode", None)

    if callable(detect_encode):
        _prog(0.12, "Running detector and embeddings...")
        t0 = time.perf_counter()
        pairs = detect_encode(image)
        detect_sec = time.perf_counter() - t0
        encode_sec = 0.0
        faces: list[tuple[FaceBox, np.ndarray]] = pairs
    else:
        _prog(0.12, "Locating faces...")
        t0 = time.perf_counter()
        boxes = backend.detect_faces(image)
        detect_sec = time.perf_counter() - t0
        _prog(0.25, "Encoding faces...")
        t1 = time.perf_counter()
        enc_list = backend.encode_faces(image, boxes) if boxes else []
        encode_sec = time.perf_counter() - t1
        faces = list(zip(boxes, enc_list))

    n = len(faces)
    if n == 0:
        _prog(1.0, "No faces detected.")
        return []

    _prog(0.4, f"Found {n} face(s); matching to gallery...")
    results: list[FaceResult] = []
    for idx, (box, emb) in enumerate(faces, start=1):
        frac = 0.4 + (idx / n) * 0.55
        _prog(min(frac, 0.95), f"Matching face {idx} of {n}...")
        t_m0 = time.perf_counter()
        best = backend.find_best_match(
            emb, known_embeddings, known_names, threshold
        )
        k = max(1, top_k)
        candidates = backend.rank_persons(
            emb, known_embeddings, known_names, threshold, k
        )
        match_sec = time.perf_counter() - t_m0
        emb_arr = np.asarray(emb)
        dim = int(emb_arr.size)
        crop = _crop_face(image, box)
        total_sec = detect_sec + encode_sec + match_sec
        results.append(
            FaceResult(
                index=idx,
                box=box,
                best=best,
                candidates=candidates,
                embedding_dim=dim,
                crop=crop,
                timings=FaceIdentifyTimings(
                    detect_sec=detect_sec,
                    encode_sec=encode_sec,
                    match_sec=match_sec,
                    total_sec=total_sec,
                ),
            )
        )
    _prog(1.0, "Done")
    return results
