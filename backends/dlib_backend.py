"""
Dlib/face_recognition backend for pluggable face recognition.
"""

from pathlib import Path
from typing import Union

import face_recognition
import numpy as np

from backends.base import FaceBox, MatchResult


class DlibBackend:
    """Backend using the face_recognition (dlib) library."""

    def __init__(
        self,
        threshold: float = 0.6,
        num_jitters: int = 1,
        **kwargs: object,
    ) -> None:
        #: Default tolerance when ``find_best_match(..., threshold=None)`` (prefer explicit threshold).
        self.threshold = threshold
        self.num_jitters = num_jitters

    def load_image(self, path: Union[Path, str]) -> np.ndarray:
        return face_recognition.load_image_file(str(path))

    def detect_faces(self, image: np.ndarray) -> list[FaceBox]:
        locations = face_recognition.face_locations(image)
        return [
            FaceBox(top=t, right=r, bottom=b, left=l)
            for (t, r, b, l) in locations
        ]

    def encode_faces(
        self, image: np.ndarray, boxes: list[FaceBox]
    ) -> list[np.ndarray]:
        locations = [
            (b.top, b.right, b.bottom, b.left)
            for b in boxes
        ]
        return face_recognition.face_encodings(
            image, locations, num_jitters=self.num_jitters
        )

    def find_best_match(
        self,
        embedding: np.ndarray,
        known_embeddings: list[np.ndarray],
        known_names: list[str],
        threshold: float,
    ) -> MatchResult:
        if not known_embeddings:
            return MatchResult(
                best_name="—",
                best_distance=1.0,
                is_match=False,
                match_pct=0.0,
                score_kind="l2_distance",
                raw_score=1.0,
            )
        th = threshold if threshold is not None else self.threshold
        matches = face_recognition.compare_faces(
            known_embeddings, embedding, tolerance=th
        )
        face_distances = face_recognition.face_distance(known_embeddings, embedding)
        if True in matches:
            first_match_idx = matches.index(True)
            best_name = known_names[first_match_idx]
            best_distance = float(face_distances[first_match_idx])
            is_match = True
        else:
            best_idx = int(face_distances.argmin())
            best_name = known_names[best_idx]
            best_distance = float(face_distances[best_idx])
            is_match = False
        match_pct = max(0.0, min(100.0, (1.0 - best_distance) * 100.0))
        return MatchResult(
            best_name=best_name,
            best_distance=best_distance,
            is_match=is_match,
            match_pct=match_pct,
            score_kind="l2_distance",
            raw_score=best_distance,
        )

    def rank_persons(
        self,
        embedding: np.ndarray,
        known_embeddings: list[np.ndarray],
        known_names: list[str],
        threshold: float,
        top_k: int,
    ) -> list[MatchResult]:
        if not known_embeddings or top_k < 1:
            return []
        th = threshold if threshold is not None else self.threshold
        dists = face_recognition.face_distance(known_embeddings, embedding)
        best_dist: dict[str, float] = {}
        for i, name in enumerate(known_names):
            d = float(dists[i])
            if name not in best_dist or d < best_dist[name]:
                best_dist[name] = d
        sorted_names = sorted(best_dist.items(), key=lambda x: x[1])[:top_k]
        out: list[MatchResult] = []
        for name, d in sorted_names:
            is_match = d <= th
            match_pct = max(0.0, min(100.0, (1.0 - d) * 100.0))
            out.append(
                MatchResult(
                    best_name=name,
                    best_distance=d,
                    is_match=is_match,
                    match_pct=match_pct,
                    score_kind="l2_distance",
                    raw_score=d,
                )
            )
        return out
