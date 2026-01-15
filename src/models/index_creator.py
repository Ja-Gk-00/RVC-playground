# src/models/index_creator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None


@dataclass
class SearchResult:
    indices: np.ndarray   # [T, k]
    distances: np.ndarray # [T, k]


class IndexCreator:
    """
    - If faiss is available: IndexFlatL2
    - Else: numpy brute-force (OK for small datasets)

    Save format:
      - FAISS: faiss.write_index()
      - No-FAISS: np.savez_compressed(filepath, vectors=..., dim=...)
        (file extension can still be ".index")
    """

    def __init__(self, dimension: int):
        self.dim = int(dimension)
        self._vectors: Optional[np.ndarray] = None

        if faiss is not None:
            self.index = faiss.IndexFlatL2(self.dim)
        else:
            self.index = None

    def add(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Vectors shape mismatch: got {vectors.shape}, dim={self.dim}")
        v = vectors.astype(np.float32, copy=False)

        # Always keep a copy for reconstruction / numpy fallback
        self._vectors = v if self._vectors is None else np.concatenate([self._vectors, v], axis=0)

        if self.index is not None:
            self.index.add(v)

    def save(self, filepath: str) -> None:
        if self.index is not None and faiss is not None:
            faiss.write_index(self.index, filepath)
            return
        if self._vectors is None:
            raise RuntimeError("Nothing to save: index has no vectors.")
        np.savez_compressed(filepath, vectors=self._vectors.astype(np.float32), dim=np.int32(self.dim))

    def load(self, filepath: str) -> None:
        if faiss is not None:
            try:
                self.index = faiss.read_index(filepath)
                self.dim = int(self.index.d)
                self._vectors = None
                return
            except Exception:
                # fall through to npz
                pass

        z = np.load(filepath, allow_pickle=False)
        self.dim = int(z["dim"])
        self._vectors = z["vectors"].astype(np.float32)

        # keep index=None => numpy fallback
        self.index = None

    def search(self, query_vectors: np.ndarray, k: int = 8) -> SearchResult:
        q = query_vectors.astype(np.float32, copy=False)
        if q.ndim != 2 or q.shape[1] != self.dim:
            raise ValueError(f"Query shape mismatch: got {q.shape}, dim={self.dim}")

        if self.index is not None and faiss is not None:
            distances, indices = self.index.search(q, int(k))
            return SearchResult(indices=indices, distances=distances)

        if self._vectors is None or self._vectors.size == 0:
            raise RuntimeError("Numpy index has no vectors loaded.")

        # brute force L2
        # dist[t, n] = ||q[t]-v[n]||^2
        v = self._vectors
        dist = ((q[:, None, :] - v[None, :, :]) ** 2).sum(axis=2)  # [T, N]
        idx = np.argpartition(dist, kth=min(k, dist.shape[1] - 1), axis=1)[:, :k]
        # sort the k neighbors
        row = np.arange(dist.shape[0])[:, None]
        d_k = dist[row, idx]
        order = np.argsort(d_k, axis=1)
        idx = idx[row, order]
        d_k = d_k[row, order]
        return SearchResult(indices=idx.astype(np.int64), distances=d_k.astype(np.float32))
