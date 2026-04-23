"""
Shared types and backend protocol for pluggable face recognition.

Call-site ``threshold`` passed to ``find_best_match`` / ``rank_persons`` always wins;
constructor ``threshold`` on backends is only the default when None is passed (legacy).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Union

import numpy as np


@dataclass(frozen=True)
class FaceBox:
    """Face bounding box in pixel coordinates (top, right, bottom, left)."""

    top: int
    right: int
    bottom: int
    left: int


@dataclass
class MatchResult:
    """Result of comparing one face embedding to known embeddings."""

    best_name: str
    best_distance: float
    is_match: bool
    match_pct: float
    #: ``l2_distance`` = lower is better (face_recognition / dlib).
    #: ``cosine_similarity`` = higher is better (InsightFace).
    score_kind: Literal["l2_distance", "cosine_similarity"] = "l2_distance"
    #: Primary metric for display: L2 distance, or cosine similarity in [0, 1].
    raw_score: float = 0.0


class FaceBackend(Protocol):
    """Protocol that every face recognition backend must implement."""

    def load_image(self, path: Union[Path, str]) -> np.ndarray:
        """Load image from disk; return RGB numpy array (H, W, 3)."""
        ...

    def detect_faces(self, image: np.ndarray) -> list[FaceBox]:
        """Return list of face bounding boxes (top, right, bottom, left)."""
        ...

    def encode_faces(
        self, image: np.ndarray, boxes: list[FaceBox]
    ) -> list[np.ndarray]:
        """Return one embedding vector per box (shape is backend-defined)."""
        ...

    def find_best_match(
        self,
        embedding: np.ndarray,
        known_embeddings: list[np.ndarray],
        known_names: list[str],
        threshold: float,
    ) -> MatchResult:
        """Compare embedding to known list; return best name, distance, is_match, match_pct."""
        ...

    def rank_persons(
        self,
        embedding: np.ndarray,
        known_embeddings: list[np.ndarray],
        known_names: list[str],
        threshold: float,
        top_k: int,
    ) -> list[MatchResult]:
        """
        Return up to ``top_k`` unique person names, best gallery match per person,
        sorted by match quality (best first).
        """
        ...
