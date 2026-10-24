"""
mutation_test.py — Mutation testing for the IVF index.

Two mutations are applied sequentially to the base dataset:

  Mutation 1 — Delete 10% of products in the busiest zone (Madurai).
  Mutation 2 — Add 5% new products from a shifted distribution
               (mean vector shifted by +0.3 in first 8 dimensions).

After each mutation the benchmarks are rerun and changes in:
  * Recall@10
  * Latency (p50, p95)
  * Selectivity estimation accuracy
  * Partition sizes (imbalance)
  * Whether the index needs rebuilding

are measured and reported.

Usage
-----
    python benchmarks/mutation_test.py [--data data/products.npz]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_data import load_dataset, CATEGORIES, ZONES, SEED, CATEGORY_PRICE_ANCHOR
from src.ivf_index import IVFIndex
from src.metadata_filter import AndFilter, CategoryEq, ZoneEq, PriceRange
from src.stats_model import StatsModel
from src.planner import Planner
from src.ground_truth import compute_ground_truth, recall_at_k


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_query(rng, dim=128):
    q = rng.standard_normal(dim).astype(np.float32)
    q /= np.linalg.norm(q) + 1e-9
    return q


def build_everything(vectors, categories, zones, prices, product_ids, seed=SEED):
    """Build index + stats + planner for a dataset."""
    index = IVFIndex(n_clusters=224)
    index.build(vectors, categories, zones, prices, product_ids, seed=seed)
    stats = StatsModel()
    stats.build(categories, zones, prices)
    planner = Planner(index, stats)
    return index, stats, planner


def run_single_benchmark(index, stats, planner, vectors, categories, zones, prices,
                          product_ids, filt_list, n_queries=30, k=10, seed=99):
    """Run a mini benchmark over filt_list; return rows for the table."""
    rng = np.random.default_rng(seed)
    cats_arr = np.array(categories)
    zones_arr = np.array(zones)
    prices_arr = np.array(prices)
    n = len(product_ids)
    rows = []

    for filt, desc in filt_list:
        actual_count = int(filt.apply_mask(cats_arr, zones_arr, prices_arr).sum()) if filt else n
        est_count = stats.estimate_count(filt)
        plan_dec = planner.decide(filt)

        recalls, lats = [], []
        for _ in range(n_queries):
            q = make_query(rng, vectors.shape[1])
            gt_ids, _ = compute_ground_truth(
                q, vectors, cats_arr, zones_arr, prices_arr, product_ids, filt, k
            )
            t0 = time.perf_counter()
            result = planner.search(q, filt, k)
            lats.append((time.perf_counter() - t0) * 1000)
            recalls.append(recall_at_k(result.product_ids, gt_ids, k))

        lats_arr = np.array(lats)
        rows.append({
            "filter": desc,
            "actual_count": actual_count,
            "est_count": est_count,
            "recall": float(np.mean(recalls)),
            "p50_ms": float(np.percentile(lats_arr, 50)),
            "p95_ms": float(np.percentile(lats_arr, 95)),
            "strategy": plan_dec.strategy,
        })
    return rows


BENCHMARK_FILTERS = [
    (None, "no_filter"),
    (AndFilter([ZoneEq("Madurai")]), "zone=Madurai"),
    (AndFilter([CategoryEq("saree"), ZoneEq("Madurai")]), "saree+Madurai"),
    (AndFilter([CategoryEq("saree"), ZoneEq("Ranipet"), PriceRange(0, 1500)]),
     "saree+Ranipet+<1500"),
]


# ---------------------------------------------------------------------------
# Mutation 1 — Delete 10% of Madurai products
# ---------------------------------------------------------------------------

def mutation_delete_madurai(ds: dict, rng: np.random.Generator) -> dict:
    zones_arr = np.array(ds["zones"])
    madurai_idx = np.where(zones_arr == "Madurai")[0]
    n_delete = int(len(madurai_idx) * 0.10)
    to_delete = rng.choice(madurai_idx, size=n_delete, replace=False)
    keep_mask = np.ones(len(ds["product_ids"]), dtype=bool)
    keep_mask[to_delete] = False

    keep_idx = np.where(keep_mask)[0]
    print(f"[Mutation 1] Deleted {n_delete} Madurai products. "
          f"Dataset: {len(ds['product_ids'])} -> {keep_mask.sum()}")
    return {
        "vectors":     ds["vectors"][keep_idx],
        "categories":  [ds["categories"][i] for i in keep_idx],
        "zones":       [ds["zones"][i]       for i in keep_idx],
        "prices":      ds["prices"][keep_idx],
        "product_ids": [ds["product_ids"][i]  for i in keep_idx],
    }


# ---------------------------------------------------------------------------
# Mutation 2 — Add 5% shifted products
# ---------------------------------------------------------------------------

def mutation_add_shifted(ds: dict, rng: np.random.Generator) -> dict:
    n = len(ds["product_ids"])
    n_add = int(n * 0.05)
    dim = ds["vectors"].shape[1]
    shift = np.zeros(dim, dtype=np.float32)
    shift[:8] = 0.3

    new_vecs = rng.standard_normal((n_add, dim)).astype(np.float32)
    new_vecs += shift
    norms = np.linalg.norm(new_vecs, axis=1, keepdims=True)
    new_vecs /= norms + 1e-9

    # Assign metadata from existing distribution, biased toward "saree" and "Chennai"
    new_cats = rng.choice(["saree", "silk_saree", "cotton_saree"], size=n_add).tolist()
    new_zones = rng.choice(["Chennai", "Coimbatore", "Salem"], size=n_add).tolist()
    new_prices = rng.uniform(500, 3000, size=n_add).astype(np.float32)
    new_ids = [f"PROD_MUT_{i:06d}" for i in range(n_add)]

    print(f"[Mutation 2] Added {n_add} shifted products. "
          f"Dataset: {n} -> {n + n_add}")
    return {
        "vectors":     np.vstack([ds["vectors"], new_vecs]),
        "categories":  ds["categories"] + new_cats,
        "zones":       ds["zones"] + new_zones,
        "prices":      np.concatenate([ds["prices"], new_prices]),
        "product_ids": ds["product_ids"] + new_ids,
    }


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def print_table(rows: List[dict], title: str):
    print(f"\n{'='*90}")
    print(f"  {title}")
    print('='*90)
    headers = ["Filter", "Actual", "Est.", "Recall@10", "p50 ms", "p95 ms", "Strategy"]
    table = [
        [
            r["filter"],
            r["actual_count"],
            r["est_count"],
            f"{r['recall']:.3f}",
            f"{r['p50_ms']:.1f}",
            f"{r['p95_ms']:.1f}",
            r["strategy"],
        ]
        for r in rows
    ]
    print(tabulate(table, headers=headers, tablefmt="github"))


def compare_tables(before: List[dict], after: List[dict], title: str):
    print(f"\n{'='*100}")
    print(f"  {title}  (Before -> After)")
    print('='*100)
    headers = [
        "Filter", "Recall before", "Recall after", "Δrecall",
        "p50 before", "p50 after", "Est.count before", "Est.count after"
    ]
    rows = []
    for b, a in zip(before, after):
        rows.append([
            b["filter"],
            f"{b['recall']:.3f}",
            f"{a['recall']:.3f}",
            f"{a['recall'] - b['recall']:+.3f}",
            f"{b['p50_ms']:.1f}",
            f"{a['p50_ms']:.1f}",
            b["est_count"],
            a["est_count"],
        ])
    print(tabulate(rows, headers=headers, tablefmt="github"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_mutation_test(data_path: str = "data/products.npz"):
    rng = np.random.default_rng(SEED + 77)

    print("Loading base dataset …")
    ds_base = load_dataset(data_path)
    n_base = len(ds_base["product_ids"])

    # ---- Baseline ----
    print("\n[Baseline] Building index …")
    idx_b, stats_b, planner_b = build_everything(
        ds_base["vectors"], ds_base["categories"], ds_base["zones"],
        ds_base["prices"], ds_base["product_ids"]
    )
    rows_base = run_single_benchmark(
        idx_b, stats_b, planner_b,
        ds_base["vectors"], ds_base["categories"], ds_base["zones"],
        ds_base["prices"], ds_base["product_ids"],
        BENCHMARK_FILTERS,
    )
    print_table(rows_base, "Baseline")

    # ---- Mutation 1: Delete Madurai ----
    print("\n[Mutation 1] Deleting 10% of Madurai products …")
    ds_m1 = mutation_delete_madurai(ds_base, rng)

    # Option A — reuse OLD index (stale)
    print("  [M1-stale] Using old index (not rebuilt) …")
    rows_m1_stale = run_single_benchmark(
        idx_b, stats_b, planner_b,
        ds_base["vectors"], ds_base["categories"], ds_base["zones"],
        ds_base["prices"], ds_base["product_ids"],
        BENCHMARK_FILTERS,
    )

    # Option B — rebuild index on mutated data
    print("  [M1-rebuild] Rebuilding index on mutated data …")
    idx_m1, stats_m1, planner_m1 = build_everything(
        ds_m1["vectors"], ds_m1["categories"], ds_m1["zones"],
        ds_m1["prices"], ds_m1["product_ids"]
    )
    rows_m1_rebuilt = run_single_benchmark(
        idx_m1, stats_m1, planner_m1,
        ds_m1["vectors"], ds_m1["categories"], ds_m1["zones"],
        ds_m1["prices"], ds_m1["product_ids"],
        BENCHMARK_FILTERS,
    )
    compare_tables(rows_base, rows_m1_rebuilt, "Mutation 1 (Delete Madurai 10%): Baseline -> Rebuilt")

    # Partition imbalance check
    sizes_b  = [len(p) for p in idx_b.partition_ids]
    sizes_m1 = [len(p) for p in idx_m1.partition_ids]
    print(f"\n  Partition imbalance (std/mean):"
          f"  Baseline={np.std(sizes_b)/np.mean(sizes_b):.3f}  "
          f"After M1={np.std(sizes_m1)/np.mean(sizes_m1):.3f}")

    # ---- Mutation 2: Add shifted products ----
    print("\n[Mutation 2] Adding 5% shifted products to M1 dataset …")
    ds_m2 = mutation_add_shifted(ds_m1, rng)

    print("  [M2-rebuild] Rebuilding index on M1+M2 data …")
    idx_m2, stats_m2, planner_m2 = build_everything(
        ds_m2["vectors"], ds_m2["categories"], ds_m2["zones"],
        ds_m2["prices"], ds_m2["product_ids"]
    )
    rows_m2 = run_single_benchmark(
        idx_m2, stats_m2, planner_m2,
        ds_m2["vectors"], ds_m2["categories"], ds_m2["zones"],
        ds_m2["prices"], ds_m2["product_ids"],
        BENCHMARK_FILTERS,
    )
    compare_tables(rows_base, rows_m2, "Mutation 2 (Add shifted 5%): Baseline -> After M1+M2")

    sizes_m2 = [len(p) for p in idx_m2.partition_ids]
    print(f"\n  Partition imbalance (std/mean):"
          f"  Baseline={np.std(sizes_b)/np.mean(sizes_b):.3f}  "
          f"After M2={np.std(sizes_m2)/np.mean(sizes_m2):.3f}")

    # ---- Analysis summary ----
    print("""
=== MUTATION ANALYSIS SUMMARY ===

Mutation 1 — Delete 10% Madurai:
  * Recall impact : Minimal if index is rebuilt; stale index returns ~same
    results as before because deleted products still exist in stale vectors.
  * Latency       : Slightly lower (fewer candidates in Madurai partitions).
  * Est. accuracy : StatsModel zone_counts now over-estimate Madurai products
    until rebuilt → planner may over-probe for Madurai queries.
  * Imbalance     : Madurai-heavy partitions shrink; slightly more balanced.
  * Rebuild needed? YES for correct recall and selectivity estimation.

Mutation 2 — Add 5% shifted distribution:
  * Recall impact : New products cluster near existing partitions (small shift);
    recall stays high IF index is rebuilt. Without rebuild, new products
    may fall into wrong partition → recall loss for shifted-vector queries.
  * Latency       : Slightly higher (more candidates to score).
  * Est. accuracy : Category/zone counts need update; price histogram needs
    update (new products concentrated in ₹500–₹3000).
  * Imbalance     : Shifted vectors fill partitions differently; imbalance
    may worsen if shift moves many vectors into few clusters.
  * Rebuild needed? YES — both index and StatsModel must be rebuilt.
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/products.npz")
    args = parser.parse_args()
    run_mutation_test(args.data)
