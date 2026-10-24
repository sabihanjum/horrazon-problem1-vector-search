"""
ground_truth.py — Exact brute-force recall reference.

For every query:
  1. Apply the filter to all N products.
  2. Compute inner-product similarity over ALL matching products.
  3. Return true top-k.

This is Strategy A (exact_scan) but implemented separately so it is
never confused with the index's own exact_scan method during benchmarking.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.metadata_filter import AndFilter


def compute_ground_truth(
    query: np.ndarray,          # (D,) float32, L2-normalised
    vectors: np.ndarray,        # (N, D) float32
    categories: np.ndarray,     # (N,) str
    zones: np.ndarray,          # (N,) str
    prices: np.ndarray,         # (N,) float32
    product_ids: List[str],
    filt: Optional[AndFilter],
    k: int = 10,
) -> Tuple[List[str], List[float]]:
    """
    Returns (top_k_ids, top_k_scores) — exact ground truth for the query.
    """
    n = len(product_ids)

    if filt is not None:
        mask = filt.apply_mask(categories, zones, prices)
        candidate_ids = np.where(mask)[0].astype(np.int32)
    else:
        candidate_ids = np.arange(n, dtype=np.int32)

    if len(candidate_ids) == 0:
        return [], []

    vecs = vectors[candidate_ids]
    sims = vecs @ query
    top_k = min(k, len(candidate_ids))
    top_local = np.argpartition(-sims, top_k - 1)[:top_k]
    top_local = top_local[np.argsort(-sims[top_local])]
    ids_out = [product_ids[i] for i in candidate_ids[top_local]]
    scores_out = sims[top_local].tolist()
    return ids_out, scores_out


def recall_at_k(predicted: List[str], ground_truth: List[str], k: int = 10) -> float:
    """
    Recall@K = |predicted ∩ ground_truth| / |ground_truth|

    Both lists are treated as ordered; we take the first k from each.
    """
    pred_set = set(predicted[:k])
    gt_set = set(ground_truth[:k])
    if not gt_set:
        return 1.0   # vacuously correct (no qualifying products)
    return len(pred_set & gt_set) / len(gt_set)
