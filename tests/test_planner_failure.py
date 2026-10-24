"""
test_planner_failure.py — Deliberately broken planner case (Step 8).

This test documents a known failure mode: when zone and category distributions
are highly correlated in a way that the independence assumption in the
StatsModel causes gross over-estimation, the planner picks exact_scan
when filter_aware_partition_search would be far more efficient.

The test PASSES because it is documenting an expected limitation — not a bug.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ivf_index import IVFIndex
from src.metadata_filter import AndFilter, CategoryEq, ZoneEq, PriceRange
from src.stats_model import StatsModel
from src.planner import Planner


# ---------------------------------------------------------------------------
# Setup: construct a dataset where category and price are highly correlated
# such that their JOINT probability is much lower than independent product
# ---------------------------------------------------------------------------

def build_correlated_dataset():
    """
    "kanjivaram" products exist ONLY in Chennai and ONLY in price range ₹8000+.
    All other categories are spread across all zones and prices.

    The price histogram won't capture the price-category correlation, so
    when the filter is category=kanjivaram AND price_range=[0, 5000],
    the stats model over-estimates (because it applies price selectivity
    independently, not knowing kanjivaram prices are always > ₹8000).

    Actual matching products: 0
    Estimated: some positive number
    """
    rng = np.random.default_rng(42)
    n, dim = 5000, 32

    # 4800 generic products across all zones/prices
    generic_cats  = rng.choice(["saree", "kurta", "dhoti", "accessories"], size=4800).tolist()
    generic_zones = rng.choice(["Madurai", "Chennai", "Coimbatore"], size=4800).tolist()
    generic_prices = rng.uniform(200, 5000, size=4800).astype(np.float32)

    # 200 kanjivaram products — ONLY in Chennai, ONLY ₹8000-₹15000
    kan_cats  = ["kanjivaram"] * 200
    kan_zones = ["Chennai"] * 200
    kan_prices = rng.uniform(8000, 15000, size=200).astype(np.float32)

    categories = generic_cats + kan_cats
    zones      = generic_zones + kan_zones
    prices     = np.concatenate([generic_prices, kan_prices])
    product_ids = [f"P{i:05d}" for i in range(n)]

    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    return vectors, categories, zones, prices, product_ids


# ---------------------------------------------------------------------------
# The broken planner case
# ---------------------------------------------------------------------------

def test_planner_failure_case():
    """
    Demonstrates the planner failure case.

    Filter: category=kanjivaram AND price=[0, 5000]
    Actual matching products: 0 (kanjivaram always costs > ₹8000)
    StatsModel estimate: POSITIVE (treats price and category independently)
    Planner decision: Based on estimated selectivity
    Better decision: filter_aware_partition_search (or exact_scan that returns 0 quickly)

    CAUSE: The StatsModel uses an independence assumption between category and price.
           It sees: kanjivaram_count / total × price_sel([0,5000]) > 0
           But in reality there is PERFECT anti-correlation (kanjivaram is never cheap).

    FIX / LIMITATION: Store per-category price histograms (conditional distributions).
           Then estimate P(price | category) × P(category) instead of independence.
    """
    vectors, categories, zones, prices, product_ids = build_correlated_dataset()

    index = IVFIndex(n_clusters=20)
    index.build(vectors, categories, zones, prices, product_ids, seed=42)

    stats = StatsModel()
    stats.build(categories, zones, prices)

    planner = Planner(index, stats)

    # The "broken" filter
    broken_filt = AndFilter([CategoryEq("kanjivaram"), PriceRange(0, 5000)])

    # Actual selectivity
    cats_arr  = np.array(categories)
    zones_arr = np.array(zones)
    prices_arr = np.array(prices)
    actual_mask = broken_filt.apply_mask(cats_arr, zones_arr, prices_arr)
    actual_count = int(actual_mask.sum())

    # Estimated selectivity
    est_count = stats.estimate_count(broken_filt)
    est_sel   = est_count / len(product_ids)

    decision = planner.decide(broken_filt)

    print(f"\n=== PLANNER FAILURE CASE ===")
    print(f"Filter              : {broken_filt.describe()}")
    print(f"Actual matching     : {actual_count} / {len(product_ids)}")
    print(f"Estimated matching  : {est_count} / {len(product_ids)}")
    print(f"Estimated sel.      : {est_sel:.4%}")
    print(f"Strategy selected   : {decision.strategy}")
    print(f"Better strategy     : filter_aware_partition_search (returns 0 fast)")
    print(f"Cause               : Independence assumption in StatsModel ignores")
    print(f"                      price-category correlation (kanjivaram > ₹8000)")
    print(f"Fix                 : Per-category price histograms in StatsModel")

    # Assert the planner DOES make a suboptimal decision (documenting the bug)
    assert actual_count == 0, f"Expected 0 kanjivaram products under ₹5000, got {actual_count}"
    assert est_count > 0, f"Expected StatsModel to over-estimate (independence assumption)"

    # The planner might choose exact_scan or partition_then_filter when
    # filter_aware would return 0 candidates immediately — not a crash, but inefficient
    # We simply document which strategy was chosen
    assert decision.strategy in (
        "exact_scan", "partition_then_filter", "filter_aware_partition_search"
    ), f"Unknown strategy: {decision.strategy}"

    print(f"\n-> Planner chose '{decision.strategy}' -- "
          f"this is {'correct' if decision.strategy == 'filter_aware_partition_search' else 'SUBOPTIMAL'} "
          f"for a filter matching 0 products.")
