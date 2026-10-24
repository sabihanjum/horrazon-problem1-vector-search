"""
ivf_index.py — Hand-rolled IVF (Inverted File) vector index.

Architecture
------------
* Build phase
    1. Use sklearn KMeans to assign each product to one of K partitions.
    2. Store cluster centroids, per-cluster product-ID lists, and
       per-cluster sub-vectors for candidate scoring.

* Search phase
    Three search strategies are exposed as independent methods:

    A. exact_scan(query, filter, k)
       Iterate over ALL filtered products; compute cosine similarity;
       return top-k.  Ground-truth baseline.

    B. partition_then_filter(query, filter, k, nprobe)
       1. Find the nprobe nearest centroids to the query.
       2. Collect all candidates from those partitions.
       3. Apply metadata filter AFTER collection (post-filtering).
       4. Rank by similarity; return top-k.

    C. filter_aware_partition_search(query, filter, k, nprobe)
       1. Pre-compute the global filtered mask.
       2. For each partition (in centroid-distance order) collect only
          the candidates that pass the filter.
       3. Stop early when enough qualifying candidates have been found
          (≥ min_candidates = max(10*k, 100)).
       4. Rank by similarity; return top-k.

Similarity metric: inner product on L2-normalised vectors (equivalent to
cosine similarity for unit vectors).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from src.metadata_filter import AndFilter


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    product_ids: List[str]
    scores: List[float]
    candidates_examined: int
    strategy: str
    latency_ms: float


# ---------------------------------------------------------------------------
# IVF Index
# ---------------------------------------------------------------------------

class IVFIndex:
    """
    Inverted File (IVF) index.

    Parameters
    ----------
    n_clusters : int
        Number of IVF partitions (Voronoi cells). Rule of thumb: sqrt(N).
    """

    def __init__(self, n_clusters: int = 224):
        self.n_clusters = n_clusters
        self.centroids: Optional[np.ndarray] = None      # (K, D)
        self.partition_ids: List[np.ndarray] = []        # K lists of int indices
        self.partition_vectors: List[np.ndarray] = []    # K arrays of (ni, D) float32
        self._vectors: Optional[np.ndarray] = None       # full (N, D) reference
        self._categories: Optional[np.ndarray] = None
        self._zones: Optional[np.ndarray] = None
        self._prices: Optional[np.ndarray] = None
        self._product_ids: Optional[List[str]] = None
        self._n: int = 0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        vectors: np.ndarray,         # (N, D) float32, L2-normalised
        categories: List[str],
        zones: List[str],
        prices: np.ndarray,          # (N,) float32
        product_ids: List[str],
        seed: int = 42,
        batch_size: int = 10_000,
    ) -> None:
        """
        Build the IVF index.
        sklearn MiniBatchKMeans is allowed (k-means only, no search).
        """
        n, d = vectors.shape
        self._n = n
        self._vectors = vectors
        self._categories = np.array(categories)
        self._zones = np.array(zones)
        self._prices = np.asarray(prices, dtype=np.float32)
        self._product_ids = product_ids

        print(f"[IVF] Building index: n={n}, dim={d}, K={self.n_clusters}")
        t0 = time.perf_counter()

        km = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=seed,
            batch_size=batch_size,
            n_init=5,
            max_iter=100,
        )
        labels = km.fit_predict(vectors)
        centroids = km.cluster_centers_.astype(np.float32)

        # L2-normalise centroids (so inner-product == cosine similarity)
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        self.centroids = centroids / (norms + 1e-9)

        # Build inverted lists
        self.partition_ids = [[] for _ in range(self.n_clusters)]
        for idx, label in enumerate(labels):
            self.partition_ids[label].append(idx)

        self.partition_vectors = []
        for k in range(self.n_clusters):
            ids_k = np.array(self.partition_ids[k], dtype=np.int32)
            self.partition_ids[k] = ids_k
            if len(ids_k):
                self.partition_vectors.append(vectors[ids_k])
            else:
                self.partition_vectors.append(np.empty((0, d), dtype=np.float32))

        elapsed = time.perf_counter() - t0
        sizes = [len(p) for p in self.partition_ids]
        print(
            f"[IVF] Built in {elapsed:.2f}s | "
            f"partition sizes: min={min(sizes)} mean={np.mean(sizes):.0f} max={max(sizes)}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _nearest_partitions(self, query: np.ndarray, nprobe: int) -> np.ndarray:
        """Return indices of the nprobe nearest centroids (descending similarity)."""
        sims = self.centroids @ query          # (K,)
        # argpartition is faster than full sort for large K
        top_k = min(nprobe, self.n_clusters)
        part_idx = np.argpartition(-sims, top_k - 1)[:top_k]
        return part_idx[np.argsort(-sims[part_idx])]

    def _score_candidates(
        self, query: np.ndarray, candidate_ids: np.ndarray, k: int
    ) -> Tuple[List[str], List[float]]:
        """Inner-product score over candidate_ids; return top-k."""
        if len(candidate_ids) == 0:
            return [], []
        vecs = self._vectors[candidate_ids]          # (m, D)
        sims = vecs @ query                          # (m,)
        top_k = min(k, len(candidate_ids))
        top_local = np.argpartition(-sims, top_k - 1)[:top_k]
        top_local = top_local[np.argsort(-sims[top_local])]
        ids_out = [self._product_ids[i] for i in candidate_ids[top_local]]
        scores_out = sims[top_local].tolist()
        return ids_out, scores_out

    # ------------------------------------------------------------------
    # Strategy A — Exact scan (ground-truth baseline)
    # ------------------------------------------------------------------

    def exact_scan(
        self,
        query: np.ndarray,
        filt: Optional[AndFilter],
        k: int = 10,
    ) -> SearchResult:
        t0 = time.perf_counter()

        if filt is not None:
            mask = filt.apply_mask(self._categories, self._zones, self._prices)
        else:
            mask = np.ones(self._n, dtype=bool)

        candidate_ids = np.where(mask)[0].astype(np.int32)
        ids_out, scores_out = self._score_candidates(query, candidate_ids, k)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return SearchResult(
            product_ids=ids_out,
            scores=scores_out,
            candidates_examined=len(candidate_ids),
            strategy="exact_scan",
            latency_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Strategy B — Partition then filter (post-filtering)
    # ------------------------------------------------------------------

    def partition_then_filter(
        self,
        query: np.ndarray,
        filt: Optional[AndFilter],
        k: int = 10,
        nprobe: int = 20,
    ) -> SearchResult:
        """
        Search nearest nprobe partitions, collect all their products,
        then apply the metadata filter.
        """
        t0 = time.perf_counter()

        part_indices = self._nearest_partitions(query, nprobe)
        candidate_lists = [self.partition_ids[p] for p in part_indices]
        candidate_ids = np.concatenate(candidate_lists) if candidate_lists else np.array([], dtype=np.int32)

        if filt is not None:
            candidate_ids = filt.apply_to_indices(
                candidate_ids, self._categories, self._zones, self._prices
            )

        ids_out, scores_out = self._score_candidates(query, candidate_ids, k)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return SearchResult(
            product_ids=ids_out,
            scores=scores_out,
            candidates_examined=len(candidate_ids),
            strategy="partition_then_filter",
            latency_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Strategy C — Filter-aware partition search (pre-filter aware)
    # ------------------------------------------------------------------

    def filter_aware_partition_search(
        self,
        query: np.ndarray,
        filt: Optional[AndFilter],
        k: int = 10,
        nprobe: int = 20,
        min_candidates: int = 100,
    ) -> SearchResult:
        """
        Pre-compute global filtered mask; then traverse partitions in
        centroid-distance order, collecting qualifying candidates from each.
        Stops early once we have >= min_candidates qualifying hits.
        """
        t0 = time.perf_counter()

        if filt is not None:
            global_mask = filt.apply_mask(self._categories, self._zones, self._prices)
        else:
            global_mask = np.ones(self._n, dtype=bool)

        part_indices = self._nearest_partitions(query, nprobe)

        collected: List[np.ndarray] = []
        total_qualifying = 0

        for p in part_indices:
            ids_k = self.partition_ids[p]
            if len(ids_k) == 0:
                continue
            qualifying = ids_k[global_mask[ids_k]]
            if len(qualifying):
                collected.append(qualifying)
                total_qualifying += len(qualifying)
            if total_qualifying >= min_candidates:
                break

        if collected:
            candidate_ids = np.concatenate(collected).astype(np.int32)
        else:
            candidate_ids = np.array([], dtype=np.int32)

        ids_out, scores_out = self._score_candidates(query, candidate_ids, k)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return SearchResult(
            product_ids=ids_out,
            scores=scores_out,
            candidates_examined=len(candidate_ids),
            strategy="filter_aware_partition_search",
            latency_ms=elapsed_ms,
        )
