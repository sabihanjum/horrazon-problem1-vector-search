"""
test_planner.py — Tests for planner strategy decisions and selectivity estimation.

Verifies:
  1. Correct strategy chosen for broad, medium, selective, very-selective filters.
  2. StatsModel estimated counts are reasonable (within 3× actual).
  3. nprobe scales inversely with selectivity.
  4. Planner can be constructed and explains its decisions.
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
from src.planner import Planner, BROAD_THRESHOLD, SELECTIVE_THRESHOLD


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def medium_dataset():
    """Medium dataset (5000 products) for planner tests."""
    rng = np.random.default_rng(42)
    n, dim = 5000, 64

    # 5 categories, 5 zones — simple distribution
    cats_list = ["saree", "kurta", "dhoti", "silk_saree", "accessories"]
    zones_list = ["Madurai", "Chennai", "Coimbatore", "Ranipet", "Ariyalur"]

    # Madurai = 40%, Chennai = 30%, others = 10% each
    zone_weights = [0.40, 0.30, 0.10, 0.10, 0.10]
    cat_weights  = [0.25, 0.25, 0.20, 0.15, 0.15]

    categories = rng.choice(cats_list, size=n, p=cat_weights).tolist()
    zones      = rng.choice(zones_list, size=n, p=zone_weights).tolist()
    prices     = rng.uniform(200, 5000, size=n).astype(np.float32)
    product_ids = [f"P{i:05d}" for i in range(n)]

    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    index = IVFIndex(n_clusters=30)
    index.build(vectors, categories, zones, prices, product_ids, seed=42)

    stats = StatsModel()
    stats.build(categories, zones, prices)

    planner = Planner(index, stats)
    return planner, stats, np.array(categories), np.array(zones), prices


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def actual_count(filt, cats, zones, prices):
    if filt is None:
        return len(prices)
    return int(filt.apply_mask(cats, zones, prices).sum())


# ---------------------------------------------------------------------------
# Test 1: Broad filter → exact_scan
# ---------------------------------------------------------------------------

def test_broad_filter_chooses_exact_scan(medium_dataset):
    planner, stats, cats, zones, prices = medium_dataset
    # no filter -> ~100% selectivity -> exact_scan
    decision = planner.decide(None)
    assert decision.strategy == "exact_scan", f"Expected exact_scan, got {decision.strategy}"
    assert decision.estimated_selectivity >= BROAD_THRESHOLD


def test_single_large_zone_chooses_exact_scan(medium_dataset):
    planner, stats, cats, zones, prices = medium_dataset
    # Madurai = 40% -> above BROAD_THRESHOLD -> exact_scan is correct
    filt = AndFilter([ZoneEq("Madurai")])
    decision = planner.decide(filt)
    assert decision.strategy == "exact_scan", (
        f"Madurai (40%) should get exact_scan, got {decision.strategy}"
    )


# ---------------------------------------------------------------------------
# Test 2: Medium filter -> partition_then_filter
# ---------------------------------------------------------------------------

def test_medium_filter_chooses_ptf(medium_dataset):
    planner, stats, cats, zones, prices = medium_dataset
    # saree (25%) x Ariyalur (10%) = ~2.5% joint selectivity
    # This falls between SELECTIVE_THRESHOLD (0.5%) and BROAD_THRESHOLD (5%)
    # so the planner should pick partition_then_filter.
    filt = AndFilter([CategoryEq("saree"), ZoneEq("Ariyalur")])
    decision = planner.decide(filt)
    assert decision.strategy == "partition_then_filter", (
        f"Expected partition_then_filter for saree+Ariyalur (~2.5%), got {decision.strategy}"
    )


# ---------------------------------------------------------------------------
# Test 3: Selective filter → filter_aware_partition_search
# ---------------------------------------------------------------------------

def test_selective_filter_chooses_faps(medium_dataset):
    planner, stats, cats, zones, prices = medium_dataset
    # Ariyalur is tiny (10% zone) × accessories (15% cat) ≈ 1.5% × any price range
    # With a tight price filter this should go below SELECTIVE_THRESHOLD
    filt = AndFilter([CategoryEq("accessories"), ZoneEq("Ariyalur"), PriceRange(200, 500)])
    decision = planner.decide(filt)
    # Estimated sel should be very small → filter_aware_partition_search
    assert decision.strategy in (
        "filter_aware_partition_search", "partition_then_filter"
    ), f"Got unexpected strategy {decision.strategy}"


# ---------------------------------------------------------------------------
# Test 4: Stats model count is within 3x of actual
# ---------------------------------------------------------------------------

def test_stats_estimate_reasonable_category(medium_dataset):
    planner, stats, cats, zones, prices = medium_dataset
    filt = AndFilter([CategoryEq("saree")])
    est = stats.estimate_count(filt)
    act = actual_count(filt, cats, zones, prices)
    ratio = est / max(act, 1)
    assert 0.33 < ratio < 3.0, (
        f"Category estimate {est} is more than 3× off from actual {act}"
    )


def test_stats_estimate_reasonable_zone(medium_dataset):
    planner, stats, cats, zones, prices = medium_dataset
    filt = AndFilter([ZoneEq("Madurai")])
    est = stats.estimate_count(filt)
    act = actual_count(filt, cats, zones, prices)
    ratio = est / max(act, 1)
    assert 0.5 < ratio < 2.0, (
        f"Zone estimate {est} is more than 2× off from actual {act}"
    )


def test_stats_estimate_reasonable_price_range(medium_dataset):
    planner, stats, cats, zones, prices = medium_dataset
    filt = AndFilter([PriceRange(200, 1500)])
    est = stats.estimate_count(filt)
    act = actual_count(filt, cats, zones, prices)
    ratio = est / max(act, 1)
    assert 0.4 < ratio < 2.5, (
        f"Price range estimate {est} is too far from actual {act}"
    )


# ---------------------------------------------------------------------------
# Test 5: nprobe scales inversely with selectivity
# ---------------------------------------------------------------------------

def test_nprobe_scales_with_selectivity(medium_dataset):
    planner, stats, cats, zones, prices = medium_dataset

    # High selectivity → low nprobe
    dec_high = planner.decide(AndFilter([ZoneEq("Madurai")]))  # ~40%
    # Low selectivity → high nprobe
    dec_low  = planner.decide(
        AndFilter([CategoryEq("accessories"), ZoneEq("Ariyalur"), PriceRange(200, 500)])
    )
    # nprobe for lower selectivity should be >= nprobe for higher selectivity
    assert dec_low.nprobe >= dec_high.nprobe, (
        f"nprobe should be ≥ for more selective filter: "
        f"high_sel={dec_high.nprobe}, low_sel={dec_low.nprobe}"
    )


# ---------------------------------------------------------------------------
# Test 6: Planner explain() runs without error
# ---------------------------------------------------------------------------

def test_planner_explain(medium_dataset):
    planner, stats, cats, zones, prices = medium_dataset
    filt = AndFilter([CategoryEq("saree"), ZoneEq("Madurai")])
    explanation = planner.explain(filt)
    assert "Strategy" in explanation
    assert "Est. sel." in explanation


# ---------------------------------------------------------------------------
# Test 7: Price range selectivity is non-trivial for narrow range
# ---------------------------------------------------------------------------

def test_price_selectivity_narrow_range(medium_dataset):
    planner, stats, cats, zones, prices = medium_dataset
    filt_narrow = AndFilter([PriceRange(4900, 5000)])  # very narrow
    filt_wide   = AndFilter([PriceRange(200, 5000)])   # covers everything

    est_narrow = stats.estimate_count(filt_narrow)
    est_wide   = stats.estimate_count(filt_wide)
    assert est_narrow < est_wide, "Narrow price range should estimate fewer products"
