from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "FAISS is not installed. Add 'faiss-cpu' (Linux/Windows) or 'faiss-gpu' to your environment."
    ) from e


@dataclass(frozen=True)
class FaissRetrieverConfig:
    index_path: str
    vectors_path: str
    k: int = 8
    index_rate: float = 0.5   # 0 = no retrieval, 1 = only retrieved
    nprobe: int = 8           # effective for IVF indexes
    chunk_size: int = 4096    # process frames in chunks to limit RAM


class FaissRetriever:
    """
    RVC-like retrieval:
      1) search kNN in FAISS
      2) compute weighted average of neighbors
      3) blend with original features using index_rate
    """

    def __init__(self, cfg: FaissRetrieverConfig):
        self.cfg = cfg
        self.index = faiss.read_index(cfg.index_path)

        # memmap for large arrays
        self.vectors = np.load(cfg.vectors_path, mmap_mode="r")
        if self.vectors.ndim != 2:
            raise ValueError(f"content_vectors.npy must be 2D, got {self.vectors.shape}.")

        if int(self.index.d) != int(self.vectors.shape[1]):
            raise ValueError(
                f"FAISS index dim ({self.index.d}) != vectors dim ({self.vectors.shape[1]}). "
                "Rebuild index from the same content_vectors.npy."
            )

        # set nprobe if supported (IndexIVF etc.)
        if hasattr(self.index, "nprobe"):
            try:
                self.index.nprobe = int(cfg.nprobe)
            except Exception:
                pass

    @staticmethod
    def _weighted_knn_average(
        neighbors: np.ndarray,  # [T, K, D]
        distances: np.ndarray,  # [T, K]
        valid: np.ndarray,      # [T, K] bool
        eps: float = 1e-6,
    ) -> np.ndarray:
        # weight ~ 1/(dist+eps), normalize across K
        w = 1.0 / (distances + eps)
        w = w * valid.astype(np.float32)
        wsum = np.sum(w, axis=1, keepdims=True) + eps
        w = w / wsum
        return np.sum(neighbors * w[..., None], axis=1).astype(np.float32)  # [T, D]

    def apply(self, content: np.ndarray) -> np.ndarray:
        """
        content: [T, D] float32
        returns: blended content [T, D]
        """
        x = np.asarray(content, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"content must be 2D [T, D], got {x.shape}.")

        k = int(self.cfg.k)
        rate = float(self.cfg.index_rate)
        if rate <= 0.0:
            return x
        rate = max(0.0, min(1.0, rate))

        out = np.empty_like(x, dtype=np.float32)
        T = x.shape[0]
        cs = int(self.cfg.chunk_size)

        for s in range(0, T, cs):
            e = min(T, s + cs)
            q = x[s:e]  # [t, D]

            distances, indices = self.index.search(q, k)  # [t, K], [t, K]
            valid = indices >= 0

            # guard: replace -1 with 0 for indexing, then mask
            safe_idx = np.where(valid, indices, 0)
            neigh = self.vectors[safe_idx]  # [t, K, D]
            neigh = neigh * valid[..., None].astype(np.float32)

            retrieved = self._weighted_knn_average(neigh, distances, valid)  # [t, D]
            out[s:e] = (1.0 - rate) * q + rate * retrieved

        return out.astype(np.float32)
