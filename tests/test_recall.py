"""
test_recall.py — Recall@10 ≥ 0.95 across selectivity groups.

These tests run the planner and each strategy against the brute-force ground
truth and assert the recall requirement.

Note: These tests use a 10,000-product dataset to stay fast in CI.
      The full 50,000-product benchmark is in benchmarks/benchmark.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ivf_index import IVFIndex
from src.metadata_filter import AndFilter, CategoryEq, ZoneEq, PriceRange
from src.stats_model import StatsModel
from src.planner import Planner
from src.ground_truth import compute_ground_truth, recall_at_k


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def recall_dataset():
    """10,000 products, 64-dim — fast CI recall fixture."""
    rng = np.random.default_rng(42)
    n, dim = 10_000, 64

    cats_list  = ["saree", "silk_saree", "cotton_saree", "kurta", "accessories",
                  "lehenga", "blouse", "kanjivaram", "bridal_saree", "kids_wear"]
    zones_list = ["Madurai", "Chennai", "Coimbatore", "Tiruppur", "Salem",
                  "Ranipet", "Ariyalur", "Kallakurichi", "Perambalur", "Theni"]

    zone_w = [0.15, 0.12, 0.10, 0.10, 0.10, 0.08, 0.08, 0.09, 0.09, 0.09]
    cat_w  = [0.14, 0.12, 0.12, 0.10, 0.10, 0.10, 0.08, 0.08, 0.08, 0.08]

    categories = rng.choice(cats_list, size=n, p=cat_w).tolist()
    zones      = rng.choice(zones_list, size=n, p=zone_w).tolist()
    prices     = rng.uniform(200, 10000, size=n).astype(np.float32)
    product_ids = [f"P{i:06d}" for i in range(n)]

    # Clustered vectors
    n_clusters_data = 30
    centres = rng.standard_normal((n_clusters_data, dim)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    labels  = rng.integers(0, n_clusters_data, size=n)
    vectors = centres[labels] + rng.standard_normal((n, dim)).astype(np.float32) * 0.3
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9

    index = IVFIndex(n_clusters=50)
    index.build(vectors, categories, zones, prices, product_ids, seed=42)

    stats = StatsModel()
    stats.build(categories, zones, prices)

    planner = Planner(index, stats)

    cats_arr  = np.array(categories)
    zones_arr = np.array(zones)
    return {
        "index": index, "stats": stats, "planner": planner,
        "vectors": vectors, "cats": cats_arr, "zones": zones_arr,
        "prices": prices, "pids": product_ids, "n": n, "dim": dim,
    }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

N_QUERIES = 20
RECALL_TARGET = 0.95


def avg_recall(ds, filt: Optional[AndFilter], strategy_fn, k=10, seed=77) -> float:
    rng = np.random.default_rng(seed)
    recalls = []
    for _ in range(N_QUERIES):
        q = rng.standard_normal(ds["dim"]).astype(np.float32)
        q /= np.linalg.norm(q) + 1e-9
        gt_ids, _ = compute_ground_truth(
            q, ds["vectors"], ds["cats"], ds["zones"],
            ds["prices"], ds["pids"], filt, k
        )
        result = strategy_fn(q, filt, k)
        recalls.append(recall_at_k(result.product_ids, gt_ids, k))
    return float(np.mean(recalls))


# ---------------------------------------------------------------------------
# BROAD: no filter
# ---------------------------------------------------------------------------

def test_recall_broad_planner(recall_dataset):
    ds = recall_dataset
    r = avg_recall(ds, None, ds["planner"].search)
    assert r >= RECALL_TARGET, f"Broad/planner recall={r:.3f} < {RECALL_TARGET}"


def test_recall_broad_post_filter(recall_dataset):
    ds = recall_dataset
    # nprobe must be high enough for a no-filter query against 50 partitions
    fn = lambda q, f, k: ds["index"].partition_then_filter(q, f, k, nprobe=47)
    r = avg_recall(ds, None, fn)
    assert r >= RECALL_TARGET, f"Broad/post-filter recall={r:.3f}"


# ---------------------------------------------------------------------------
# MEDIUM: single zone (Madurai ~15%)
# ---------------------------------------------------------------------------

def test_recall_medium_planner(recall_dataset):
    ds = recall_dataset
    filt = AndFilter([ZoneEq("Madurai")])
    r = avg_recall(ds, filt, ds["planner"].search)
    assert r >= RECALL_TARGET, f"Medium/planner recall={r:.3f}"


def test_recall_medium_post_filter(recall_dataset):
    ds = recall_dataset
    filt = AndFilter([ZoneEq("Madurai")])
    # nprobe=47 gives ~94% partition coverage for the 50-partition test index
    fn = lambda q, f, k: ds["index"].partition_then_filter(q, f, k, nprobe=47)
    r = avg_recall(ds, filt, fn)
    assert r >= RECALL_TARGET, f"Medium/post-filter recall={r:.3f}"


# ---------------------------------------------------------------------------
# SELECTIVE: category + zone (~1-2%)
# ---------------------------------------------------------------------------

def test_recall_selective_planner(recall_dataset):
    ds = recall_dataset
    filt = AndFilter([CategoryEq("saree"), ZoneEq("Madurai")])
    r = avg_recall(ds, filt, ds["planner"].search)
    assert r >= RECALL_TARGET, f"Selective/planner recall={r:.3f}"


def test_recall_selective_filter_aware(recall_dataset):
    ds = recall_dataset
    filt = AndFilter([CategoryEq("saree"), ZoneEq("Madurai")])
    # min_candidates=500 prevents early stopping; nprobe=47 covers 94% of partitions
    fn = lambda q, f, k: ds["index"].filter_aware_partition_search(
        q, f, k, nprobe=47, min_candidates=500
    )
    r = avg_recall(ds, filt, fn)
    assert r >= RECALL_TARGET, f"Selective/filter-aware recall={r:.3f}"


# ---------------------------------------------------------------------------
# VERY SELECTIVE: cat + small zone + price
# ---------------------------------------------------------------------------

def test_recall_very_selective_planner(recall_dataset):
    ds = recall_dataset
    filt = AndFilter([CategoryEq("saree"), ZoneEq("Ranipet"), PriceRange(0, 1500)])
    r = avg_recall(ds, filt, ds["planner"].search)
    # Very selective — vacuously 1.0 if few results (both are empty)
    assert r >= RECALL_TARGET, f"Very-selective/planner recall={r:.3f}"


# ---------------------------------------------------------------------------
# TINY: very few products — recall may be vacuously 1.0 if both return empty
# ---------------------------------------------------------------------------

def test_recall_tiny_exact_scan(recall_dataset):
    ds = recall_dataset
    filt = AndFilter([CategoryEq("bridal_saree"), ZoneEq("Ariyalur"), PriceRange(5000, 8000)])
    fn = ds["index"].exact_scan
    r = avg_recall(ds, filt, fn)
    assert r >= RECALL_TARGET, f"Tiny/exact-scan recall={r:.3f}"


# ---------------------------------------------------------------------------
# Consistency: exact_scan == ground_truth IDs
# ---------------------------------------------------------------------------

def test_exact_scan_equals_ground_truth(recall_dataset):
    ds = recall_dataset
    rng = np.random.default_rng(100)
    filt = AndFilter([ZoneEq("Madurai")])
    mismatches = 0
    for _ in range(10):
        q = rng.standard_normal(ds["dim"]).astype(np.float32)
        q /= np.linalg.norm(q) + 1e-9
        gt_ids, _ = compute_ground_truth(
            q, ds["vectors"], ds["cats"], ds["zones"],
            ds["prices"], ds["pids"], filt, k=10
        )
        result = ds["index"].exact_scan(q, filt, k=10)
        if result.product_ids != gt_ids:
            mismatches += 1
    # They should always agree (same algorithm, deterministic)
    assert mismatches == 0, f"exact_scan disagreed with ground_truth on {mismatches}/10 queries"
