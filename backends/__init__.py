"""
Pluggable face recognition backends. Use get_backend(name) to obtain a backend.
InsightFace is registered only if insightface + onnxruntime are installed.
"""

from typing import Any

from backends.base import FaceBackend, FaceBox, MatchResult
from backends.dlib_backend import DlibBackend

BACKENDS: dict[str, type[FaceBackend]] = {
    "dlib": DlibBackend,
}

try:
    from backends.insightface_backend import InsightFaceBackend

    BACKENDS["insightface"] = InsightFaceBackend
except ImportError:
    pass


def get_backend(name: str, **kwargs: Any) -> FaceBackend:
    """Return an instance of the named backend. Raises KeyError if unknown."""
    name = name.strip().lower()
    if name not in BACKENDS:
        available = ", ".join(sorted(BACKENDS))
        raise KeyError(f"Unknown backend {name!r}. Available: {available}")
    cls = BACKENDS[name]
    return cls(**kwargs)


def list_backends() -> list[str]:
    """Return list of registered backend names."""
    return sorted(BACKENDS.keys())


__all__ = [
    "FaceBackend",
    "FaceBox",
    "MatchResult",
    "BACKENDS",
    "get_backend",
    "list_backends",
]
