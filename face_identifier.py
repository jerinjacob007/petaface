#!/usr/bin/env python3
"""
Face identification CLI. Detects faces, matches against ``known_faces/`` gallery,
writes matched crops under ``results/<name>/`` and unknown crops under ``unknown_faces/``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image

from backends import list_backends
from backends.base import FaceBox
from core.pipeline import (
    IMAGE_EXTS,
    build_backend,
    identify,
    load_known_faces,
)

KNOWN_FACES_DIR = Path("known_faces")
UNKNOWN_FACES_DIR = Path("unknown_faces")
INPUT_IMAGES_DIR = Path("input_images")
RESULTS_DIR = Path("results")


def ensure_folders() -> None:
    KNOWN_FACES_DIR.mkdir(exist_ok=True)
    UNKNOWN_FACES_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    print(
        f"Folders ready: {KNOWN_FACES_DIR}/, {UNKNOWN_FACES_DIR}/, {RESULTS_DIR}/"
    )


def save_face_crop(image_array, box: FaceBox, save_path: Path) -> None:
    pil_image = Image.fromarray(image_array)
    crop = pil_image.crop((box.left, box.top, box.right, box.bottom))
    crop.save(save_path)


def process_image(
    image_path: Path,
    backend,
    known_embeddings: list,
    known_names: list[str],
    threshold: float,
    top_k: int,
    unknown_counter: int,
    verbose: bool = False,
) -> int:
    """
    Run identification on one file; write crops. Returns updated unknown_counter.
    """
    try:
        image = backend.load_image(image_path)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return unknown_counter

    faces = identify(
        image, backend, known_embeddings, known_names, threshold, top_k=top_k
    )
    if not faces:
        if verbose:
            print(f"  [verbose] No faces detected in {image_path.name}")
        return unknown_counter

    if verbose:
        print(f"  [verbose] Detected {len(faces)} face(s) in {image_path.name}")

    base_name = image_path.stem
    for fr in faces:
        result = fr.best
        box = fr.box
        idx = fr.index
        if verbose:
            metric = (
                f"L2={result.raw_score:.6f}"
                if result.score_kind == "l2_distance"
                else f"cosine={result.raw_score:.6f}"
            )
            print(
                f"  [verbose] Face #{idx} box: top={box.top} left={box.left} "
                f"right={box.right} bottom={box.bottom}"
            )
            print(
                f"  [verbose]   best_name={result.best_name!r} is_match={result.is_match} "
                f"match_pct={result.match_pct:.2f}% {metric} "
                f"score_kind={result.score_kind} threshold={threshold}"
            )

        if result.is_match:
            name = result.best_name
            person_dir = RESULTS_DIR / name
            person_dir.mkdir(parents=True, exist_ok=True)
            save_path = person_dir / f"{base_name}_face{idx}_t{box.top}_l{box.left}.jpg"
            save_face_crop(image, box, save_path)
            metric_s = (
                f"distance: {result.raw_score:.3f}"
                if result.score_kind == "l2_distance"
                else f"cosine: {result.raw_score:.3f}"
            )
            print(
                f"  Known: {name} | match: {result.match_pct:.1f}% | "
                f"{metric_s} -> {save_path}"
            )
        else:
            UNKNOWN_FACES_DIR.mkdir(exist_ok=True)
            unknown_counter += 1
            save_path = (
                UNKNOWN_FACES_DIR / f"{base_name}_unknown_{unknown_counter}.jpg"
            )
            save_face_crop(image, box, save_path)
            metric_s = (
                f"distance: {result.raw_score:.3f}"
                if result.score_kind == "l2_distance"
                else f"cosine: {result.raw_score:.3f}"
            )
            print(
                f"  Unknown | best: {result.best_name} {result.match_pct:.1f}% "
                f"({metric_s}, threshold: {threshold}) -> {save_path}"
            )

    return unknown_counter


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Identify faces using ``known_faces/<Person>/`` as gallery. "
            "Matched crops go to ``results/<Person>/``; unknown crops to ``unknown_faces/``."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=str(INPUT_IMAGES_DIR),
        help="Path to an image file or folder of images (default: input_images/)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="dlib",
        choices=list_backends(),
        help="Recognition stack: dlib (face_recognition) or insightface (default: dlib)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help=(
            "Match threshold. For dlib: max L2 distance (lower stricter); typical 0.5–0.7, default 0.6. "
            "For insightface: min cosine similarity in [0,1] (higher stricter); typical 0.4–0.55, default 0.5."
        ),
    )
    parser.add_argument(
        "--num-jitters",
        type=int,
        default=1,
        help="Dlib only: face encoding re-samples (higher = slower, slightly more accurate)",
    )
    parser.add_argument(
        "--model-pack",
        type=str,
        default="buffalo_l",
        choices=("buffalo_l", "buffalo_sc"),
        help="InsightFace only: model pack (default: buffalo_l)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-K persons logged in verbose / ranking (default: 3)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-face boxes, scores, and gallery counts",
    )
    args = parser.parse_args()

    if args.backend == "dlib":
        default_tol = 0.6
    else:
        default_tol = 0.5
    tolerance = args.tolerance if args.tolerance is not None else default_tol

    if args.backend == "dlib":
        model_id = "dlib"
    else:
        model_id = f"insightface:{args.model_pack}"

    try:
        backend = build_backend(
            model_id,
            tolerance=tolerance,
            num_jitters=args.num_jitters,
        )
    except (KeyError, ValueError) as e:
        print(e)
        print("For InsightFace: uv sync --extra insightface")
        return

    ensure_folders()

    known_embeddings, known_names, _counts = load_known_faces(
        backend,
        KNOWN_FACES_DIR,
        on_warning=lambda msg: print(f"Warning: {msg}"),
    )
    known_counts = dict(Counter(known_names))
    print(f"Loaded {len(known_names)} known face encodings from {KNOWN_FACES_DIR}/")
    if args.verbose:
        print(f"  [verbose] model_id={model_id} tolerance={tolerance}")
        if args.backend == "dlib":
            print(f"  [verbose] num_jitters={args.num_jitters}")
        else:
            print(f"  [verbose] model_pack={args.model_pack}")
        print(f"  [verbose] Known faces per person: {known_counts}")

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input path does not exist: {input_path}")
        print("Create an 'input_images' folder and add photos, or pass an image path.")
        return

    unknown_counter = 0
    if input_path.is_file():
        if args.verbose:
            print(f"Processing {input_path.name}...")
        unknown_counter = process_image(
            input_path,
            backend,
            known_embeddings,
            known_names,
            tolerance,
            args.top_k,
            unknown_counter,
            verbose=args.verbose,
        )
    else:
        for img_path in sorted(input_path.iterdir()):
            if img_path.suffix.lower() in IMAGE_EXTS:
                print(f"Processing {img_path.name}...")
                unknown_counter = process_image(
                    img_path,
                    backend,
                    known_embeddings,
                    known_names,
                    tolerance,
                    args.top_k,
                    unknown_counter,
                    verbose=args.verbose,
                )


if __name__ == "__main__":
    main()
