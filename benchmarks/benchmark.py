"""
benchmark.py - Full benchmark: 5 selectivity groups x 4 strategies.

Usage
-----
    python benchmarks/benchmark.py [--data data/products.npz] [--k 10] [--n-queries 50]

Output
------
Prints a results table to stdout and saves results/benchmark_results.json.
Recall@10 >= 0.95 is the target for every group.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tabulate import tabulate

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_data import load_dataset, CATEGORIES, ZONES, SEED
from src.ivf_index import IVFIndex
from src.metadata_filter import AndFilter, CategoryEq, ZoneEq, PriceRange
from src.stats_model import StatsModel
from src.planner import Planner
from src.ground_truth import compute_ground_truth, recall_at_k


# ---------------------------------------------------------------------------
# Selectivity groups
# ---------------------------------------------------------------------------

def make_selectivity_groups(categories, zones, prices, rng: np.random.Generator):
    """
    Build 5 query filter groups covering different selectivity levels.
    Each group is a list of (filter, description) tuples.
    """
    cats = np.array(categories)
    zns = np.array(zones)
    prs = np.array(prices)
    n = len(prices)

    groups = {}

    # --- BROAD (~100%): no filter ---
    groups["broad"] = [
        (None, "no_filter"),
    ]

    # --- MEDIUM (~10%): single zone ---
    groups["medium"] = [
        (AndFilter([ZoneEq("Madurai")]),   "zone=Madurai"),
        (AndFilter([ZoneEq("Chennai")]),   "zone=Chennai"),
    ]

    # --- SELECTIVE (~1%): category in large zone ---
    groups["selective"] = [
        (AndFilter([CategoryEq("saree"), ZoneEq("Madurai")]),  "saree+Madurai"),
        (AndFilter([CategoryEq("kurta"), ZoneEq("Chennai")]),  "kurta+Chennai"),
    ]

    # --- VERY_SELECTIVE (~0.1%): category + small zone + price range ---
    groups["very_selective"] = [
        (AndFilter([CategoryEq("saree"), ZoneEq("Ranipet"), PriceRange(0, 1500)]),
         "saree+Ranipet+price<1500"),
        (AndFilter([CategoryEq("silk_saree"), ZoneEq("Kallakurichi")]),
         "silk_saree+Kallakurichi"),
    ]

    # --- TINY (handful): narrow category + tiny zone + tight price ---
    groups["tiny"] = [
        (AndFilter([CategoryEq("bridal_saree"), ZoneEq("Ariyalur"), PriceRange(5000, 8000)]),
         "bridal+Ariyalur+5000-8000"),
        (AndFilter([CategoryEq("kanjivaram"), ZoneEq("Perambalur"), PriceRange(3000, 5000)]),
         "kanjivaram+Perambalur+3000-5000"),
    ]

    return groups


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------

def make_query(rng: np.random.Generator, dim: int = 128) -> np.ndarray:
    q = rng.standard_normal(dim).astype(np.float32)
    q /= (np.linalg.norm(q) + 1e-9)
    return q


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    data_path: str = "data/products.npz",
    k: int = 10,
    n_queries: int = 50,
    n_clusters: int = 224,
    seed: int = SEED,
) -> Dict:
    rng = np.random.default_rng(seed + 1)   # separate from data seed

    # Load dataset
    print(f"Loading dataset from {data_path} ...")
    ds = load_dataset(data_path)
    vectors = ds["vectors"]
    categories = ds["categories"]
    zones = ds["zones"]
    prices = ds["prices"]
    product_ids = ds["product_ids"]
    n = len(product_ids)
    dim = vectors.shape[1]
    print(f"  N={n}, dim={dim}")

    # Build IVF index
    index = IVFIndex(n_clusters=n_clusters)
    index.build(vectors, categories, zones, prices, product_ids, seed=seed)

    # Build stats model
    stats = StatsModel()
    stats.build(categories, zones, prices)

    # Build planner
    planner = Planner(index, stats)

    # Build selectivity groups
    groups = make_selectivity_groups(categories, zones, prices, rng)

    # Pre-compute actual selectivity for each group
    cats_arr = np.array(categories)
    zones_arr = np.array(zones)
    prices_arr = np.array(prices)

    all_results = {}
    table_rows = []

    strategies = [
        ("planner",           None),
        ("post_filter",       "partition_then_filter"),
        ("pre_filter",        "filter_aware_partition_search"),
        ("exact_ground_truth","exact_scan"),
    ]

    for group_name, filters in groups.items():
        group_results = []

        for filt, filt_desc in filters:
            # Actual selectivity
            if filt is None:
                actual_count = n
            else:
                actual_count = int(filt.apply_mask(cats_arr, zones_arr, prices_arr).sum())
            actual_sel = actual_count / n

            # Planner decision for this filter
            plan_dec = planner.decide(filt)

            for strat_name, strat_fixed in strategies:
                latencies_ms = []
                recalls = []
                candidates_examined_list = []

                for qi in range(n_queries):
                    query = make_query(rng, dim)

                    # Ground truth
                    gt_ids, _ = compute_ground_truth(
                        query, vectors, cats_arr, zones_arr, prices_arr,
                        product_ids, filt, k
                    )

                    t0 = time.perf_counter()
                    if strat_name == "planner":
                        result = planner.search(query, filt, k)
                    elif strat_fixed == "partition_then_filter":
                        result = index.partition_then_filter(query, filt, k, nprobe=plan_dec.nprobe)
                    elif strat_fixed == "filter_aware_partition_search":
                        result = index.filter_aware_partition_search(query, filt, k, nprobe=plan_dec.nprobe)
                    else:  # exact_scan
                        result = index.exact_scan(query, filt, k)
                    elapsed_ms = (time.perf_counter() - t0) * 1000

                    recall = recall_at_k(result.product_ids, gt_ids, k)
                    recalls.append(recall)
                    latencies_ms.append(elapsed_ms)
                    candidates_examined_list.append(result.candidates_examined)

                latencies_arr = np.array(latencies_ms)
                row = {
                    "group": group_name,
                    "filter": filt_desc,
                    "actual_count": actual_count,
                    "actual_sel": actual_sel,
                    "strategy": strat_name,
                    "planner_strategy": plan_dec.strategy,
                    "recall_mean": float(np.mean(recalls)),
                    "p50_ms": float(np.percentile(latencies_arr, 50)),
                    "p95_ms": float(np.percentile(latencies_arr, 95)),
                    "avg_candidates": float(np.mean(candidates_examined_list)),
                    "recall_pass": float(np.mean(recalls)) >= 0.95,
                }
                group_results.append(row)

                status = "[PASS]" if row["recall_pass"] else "[FAIL]"
                table_rows.append([
                    group_name,
                    filt_desc,
                    f"{actual_sel:.4%}",
                    strat_name,
                    f"{row['recall_mean']:.3f} {status}",
                    f"{row['p50_ms']:.1f}",
                    f"{row['p95_ms']:.1f}",
                    f"{row['avg_candidates']:.0f}",
                    plan_dec.strategy,
                ])

        all_results[group_name] = group_results

    headers = [
        "Group", "Filter", "Actual sel.", "Strategy",
        "Recall@10", "p50 ms", "p95 ms", "Avg candidates", "Planner choice"
    ]
    print("\n" + "="*120)
    print("BENCHMARK RESULTS")
    print("="*120)
    print(tabulate(table_rows, headers=headers, tablefmt="github"))

    # Check recall targets
    fails = [r for grp in all_results.values() for r in grp if not r["recall_pass"]]
    if fails:
        print(f"\n[WARN] {len(fails)} group/strategy combinations failed Recall@10 >= 0.95:")
        for f in fails:
            print(f"  {f['group']} | {f['filter']} | {f['strategy']} -> recall={f['recall_mean']:.3f}")
    else:
        print("\n[PASS] All groups meet Recall@10 >= 0.95")

    # Save JSON
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "benchmark_results.json"
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nResults saved -> {out_path}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="data/products.npz")
    parser.add_argument("--k",          type=int, default=10)
    parser.add_argument("--n-queries",  type=int, default=50)
    parser.add_argument("--n-clusters", type=int, default=224)
    parser.add_argument("--seed",       type=int, default=SEED)
    args = parser.parse_args()

    run_benchmark(
        data_path=args.data,
        k=args.k,
        n_queries=args.n_queries,
        n_clusters=args.n_clusters,
        seed=args.seed,
    )
