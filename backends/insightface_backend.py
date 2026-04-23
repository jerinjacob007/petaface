"""
InsightFace backend (buffalo_l / buffalo_sc) for pluggable face recognition.
Optimized for CCTV and low-resolution. Optional dependency: pip install insightface onnxruntime.
"""

from pathlib import Path
from typing import Any, Union

import numpy as np

from backends.base import FaceBox, MatchResult


class InsightFaceBackend:
    """Backend using InsightFace (e.g. buffalo_l or buffalo_sc for CCTV)."""

    def __init__(
        self,
        threshold: float = 0.5,
        model_pack: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        **kwargs: Any,
    ) -> None:
        #: Default minimum cosine similarity when threshold is None (prefer explicit threshold).
        self.threshold = threshold
        self.model_pack = model_pack
        self.det_size = det_size
        self._app: Any = None
        #: Populated by ``detect_faces`` so ``encode_faces`` can align without re-inference.
        self._last_pairs: list[tuple[FaceBox, np.ndarray]] | None = None

    def _get_app(self) -> Any:
        if self._app is None:
            from insightface.app import FaceAnalysis

            self._app = FaceAnalysis(name=self.model_pack)
            self._app.prepare(ctx_id=0, det_size=self.det_size)
        return self._app

    def load_image(self, path: Union[Path, str]) -> np.ndarray:
        import cv2

        bgr = cv2.imread(str(path))
        if bgr is None:
            raise OSError(f"Could not load image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def detect_and_encode(self, image: np.ndarray) -> list[tuple[FaceBox, np.ndarray]]:
        """Single-pass detection + embedding (preferred over separate detect/encode)."""
        import cv2

        app = self._get_app()
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        faces = app.get(bgr)
        out: list[tuple[FaceBox, np.ndarray]] = []
        for f in faces:
            bbox = f.bbox.astype(int)
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
            box = FaceBox(top=top, right=right, bottom=bottom, left=left)
            out.append((box, np.array(f.embedding, dtype=np.float32)))
        return out

    def detect_faces(self, image: np.ndarray) -> list[FaceBox]:
        self._last_pairs = self.detect_and_encode(image)
        return [b for b, _ in self._last_pairs]

    def encode_faces(
        self, image: np.ndarray, boxes: list[FaceBox]
    ) -> list[np.ndarray]:
        """
        Use embeddings from the preceding ``detect_faces`` on the same ``image`` when counts match;
        otherwise run ``detect_and_encode`` again.
        """
        if not boxes:
            self._last_pairs = []
            return []
        if self._last_pairs is not None and len(self._last_pairs) == len(boxes):
            return [emb for _, emb in self._last_pairs]
        self._last_pairs = self.detect_and_encode(image)
        return [emb for _, emb in self._last_pairs]

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
                score_kind="cosine_similarity",
                raw_score=0.0,
            )
        th = threshold if threshold is not None else self.threshold
        emb = embedding.reshape(1, -1).astype(np.float32)
        known = np.stack(known_embeddings, axis=0).astype(np.float32)
        emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        known_n = known / (np.linalg.norm(known, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(known_n, emb_n.T).flatten()
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])
        best_name = known_names[best_idx]
        best_distance = 1.0 - best_sim
        match_pct = max(0.0, min(100.0, best_sim * 100.0))
        is_match = best_sim >= th
        return MatchResult(
            best_name=best_name,
            best_distance=best_distance,
            is_match=is_match,
            match_pct=match_pct,
            score_kind="cosine_similarity",
            raw_score=best_sim,
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
        emb = embedding.reshape(1, -1).astype(np.float32)
        known = np.stack(known_embeddings, axis=0).astype(np.float32)
        emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        known_n = known / (np.linalg.norm(known, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(known_n, emb_n.T).flatten()
        best_sim: dict[str, float] = {}
        for i, name in enumerate(known_names):
            s = float(similarities[i])
            if name not in best_sim or s > best_sim[name]:
                best_sim[name] = s
        sorted_names = sorted(best_sim.items(), key=lambda x: -x[1])[:top_k]
        out: list[MatchResult] = []
        for name, sim in sorted_names:
            is_match = sim >= th
            match_pct = max(0.0, min(100.0, sim * 100.0))
            out.append(
                MatchResult(
                    best_name=name,
                    best_distance=1.0 - sim,
                    is_match=is_match,
                    match_pct=match_pct,
                    score_kind="cosine_similarity",
                    raw_score=sim,
                )
            )
        return out
