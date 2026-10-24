"""
demo_queries.py — Interactive demo showing 3 queries with different planner choices.

Usage
-----
    python scripts/demo_queries.py [--data data/products.npz]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_data import load_dataset, SEED
from src.ivf_index import IVFIndex
from src.metadata_filter import AndFilter, CategoryEq, ZoneEq, PriceRange
from src.stats_model import StatsModel
from src.planner import Planner
from src.ground_truth import compute_ground_truth, recall_at_k


def demo(data_path: str = "data/products.npz"):
    print("Loading dataset …")
    ds = load_dataset(data_path)
    vectors = ds["vectors"]
    categories = np.array(ds["categories"])
    zones = np.array(ds["zones"])
    prices = ds["prices"]
    product_ids = ds["product_ids"]
    n, dim = vectors.shape

    # Build
    index = IVFIndex(n_clusters=224)
    index.build(vectors, ds["categories"], ds["zones"], prices, product_ids, seed=SEED)
    stats = StatsModel()
    stats.build(ds["categories"], ds["zones"], prices)
    planner = Planner(index, stats)

    rng = np.random.default_rng(SEED + 999)
    query = rng.standard_normal(dim).astype(np.float32)
    query /= np.linalg.norm(query) + 1e-9

    queries = [
        # 1. Broad — should pick exact_scan
        (None, "Q1: No filter (BROAD — ~100%)"),
        # 2. Medium — partition_then_filter
        (AndFilter([ZoneEq("Madurai")]), "Q2: zone=Madurai (MEDIUM — ~12%)"),
        # 3. Very selective — filter_aware_partition_search
        (AndFilter([CategoryEq("saree"), ZoneEq("Ranipet"), PriceRange(0, 1500)]),
         "Q3: saree + Ranipet + price<₹1500 (VERY SELECTIVE — ~0.1%)"),
    ]

    for filt, title in queries:
        print(f"\n{'─'*70}")
        print(f"  {title}")
        print('─'*70)
        print(planner.explain(filt))
        result = planner.search(query, filt, k=10)
        gt_ids, _ = compute_ground_truth(
            query, vectors, categories, zones, prices, product_ids, filt, k=10
        )
        recall = recall_at_k(result.product_ids, gt_ids, k=10)
        print(f"\nTop results   : {result.product_ids[:5]} …")
        print(f"Candidates ex.: {result.candidates_examined}")
        print(f"Latency       : {result.latency_ms:.2f} ms")
        print(f"Recall@10     : {recall:.3f}")

        # Verify every result satisfies the filter
        if filt is not None:
            pid_to_idx = {pid: i for i, pid in enumerate(product_ids)}
            for pid in result.product_ids:
                idx = pid_to_idx[pid]
                cat_a = np.array([categories[idx]])
                zone_a = np.array([zones[idx]])
                price_a = np.array([prices[idx]])
                assert filt.apply_mask(cat_a, zone_a, price_a)[0], \
                    f"Product {pid} does NOT satisfy filter!"
        print("✓ All results satisfy filter predicates")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/products.npz")
    args = parser.parse_args()
    demo(args.data)
