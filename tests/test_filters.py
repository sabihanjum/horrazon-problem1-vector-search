"""
test_filters.py — Tests that every returned product satisfies every filter predicate.

Covers:
  1. Saree products in Madurai (Tamil Nadu proxy)
  2. Products below ₹1,500
  3. Sarees in Madurai below ₹1,500
  4. Edge cases (empty results, single predicate, three predicates)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metadata_filter import AndFilter, CategoryEq, ZoneEq, PriceRange
from src.ivf_index import IVFIndex

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_dataset():
    """Small deterministic dataset for filter unit tests."""
    rng = np.random.default_rng(0)
    n = 500
    dim = 32
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    categories = (["saree"] * 100 + ["silk_saree"] * 100 + ["kurta"] * 100
                  + ["dhoti"] * 100 + ["accessories"] * 100)
    zones = (["Madurai"] * 200 + ["Chennai"] * 150 + ["Ranipet"] * 100
             + ["Ariyalur"] * 50)
    # Interleave so each category appears in each zone
    categories_arr = np.array(categories)
    zones_arr = np.array(zones)
    np.random.default_rng(1).shuffle(categories_arr)
    np.random.default_rng(2).shuffle(zones_arr)

    prices = rng.uniform(200, 5000, size=n).astype(np.float32)
    product_ids = [f"P{i:04d}" for i in range(n)]

    index = IVFIndex(n_clusters=10)
    index.build(vectors, categories_arr.tolist(), zones_arr.tolist(),
                prices, product_ids, seed=0)
    return index, categories_arr, zones_arr, prices, product_ids


# ---------------------------------------------------------------------------
# Helper: verify every returned product satisfies the filter
# ---------------------------------------------------------------------------

def verify_results(result_ids, product_ids, categories, zones, prices, filt):
    pid_to_idx = {pid: i for i, pid in enumerate(product_ids)}
    for pid in result_ids:
        idx = pid_to_idx[pid]
        cat_arr = np.array([categories[idx]])
        zone_arr = np.array([zones[idx]])
        price_arr = np.array([prices[idx]])
        mask = filt.apply_mask(cat_arr, zone_arr, price_arr)
        assert mask[0], (
            f"Product {pid} (cat={categories[idx]}, zone={zones[idx]}, "
            f"price={prices[idx]:.0f}) does NOT satisfy {filt.describe()}"
        )


# ---------------------------------------------------------------------------
# Test 1: Saree products in Madurai
# ---------------------------------------------------------------------------

def test_saree_in_madurai(small_dataset):
    index, cats, zones, prices, pids = small_dataset
    filt = AndFilter([CategoryEq("saree"), ZoneEq("Madurai")])
    q = np.ones(32, dtype=np.float32) / np.sqrt(32)

    result = index.exact_scan(q, filt, k=10)
    assert len(result.product_ids) > 0, "No results for saree+Madurai filter"
    verify_results(result.product_ids, pids, cats.tolist(), zones.tolist(), prices, filt)


# ---------------------------------------------------------------------------
# Test 2: Products below ₹1,500
# ---------------------------------------------------------------------------

def test_products_below_1500(small_dataset):
    index, cats, zones, prices, pids = small_dataset
    filt = AndFilter([PriceRange(0, 1500)])
    q = np.ones(32, dtype=np.float32) / np.sqrt(32)

    result = index.exact_scan(q, filt, k=20)
    assert len(result.product_ids) > 0, "No results for price < ₹1,500"
    verify_results(result.product_ids, pids, cats.tolist(), zones.tolist(), prices, filt)


# ---------------------------------------------------------------------------
# Test 3: Sarees in Madurai below ₹1,500
# ---------------------------------------------------------------------------

def test_saree_madurai_below_1500(small_dataset):
    index, cats, zones, prices, pids = small_dataset
    filt = AndFilter([CategoryEq("saree"), ZoneEq("Madurai"), PriceRange(0, 1500)])
    q = np.ones(32, dtype=np.float32) / np.sqrt(32)

    result = index.exact_scan(q, filt, k=10)
    # May be 0 results in tiny dataset — that's OK if filter is correct
    verify_results(result.product_ids, pids, cats.tolist(), zones.tolist(), prices, filt)


# ---------------------------------------------------------------------------
# Test 4: Partition_then_filter also satisfies the filter
# ---------------------------------------------------------------------------

def test_partition_then_filter_satisfies_predicate(small_dataset):
    index, cats, zones, prices, pids = small_dataset
    filt = AndFilter([CategoryEq("saree"), ZoneEq("Madurai")])
    q = np.ones(32, dtype=np.float32) / np.sqrt(32)

    result = index.partition_then_filter(q, filt, k=10, nprobe=5)
    verify_results(result.product_ids, pids, cats.tolist(), zones.tolist(), prices, filt)


# ---------------------------------------------------------------------------
# Test 5: Filter-aware search also satisfies the filter
# ---------------------------------------------------------------------------

def test_filter_aware_satisfies_predicate(small_dataset):
    index, cats, zones, prices, pids = small_dataset
    filt = AndFilter([CategoryEq("saree"), ZoneEq("Madurai")])
    q = np.ones(32, dtype=np.float32) / np.sqrt(32)

    result = index.filter_aware_partition_search(q, filt, k=10, nprobe=5)
    verify_results(result.product_ids, pids, cats.tolist(), zones.tolist(), prices, filt)


# ---------------------------------------------------------------------------
# Test 6: PriceRange predicate correctness
# ---------------------------------------------------------------------------

def test_price_range_predicate():
    cats  = np.array(["saree", "saree", "saree"])
    zones = np.array(["Madurai", "Madurai", "Madurai"])
    prices = np.array([500.0, 1500.0, 2000.0], dtype=np.float32)

    filt = AndFilter([PriceRange(0, 1500)])
    mask = filt.apply_mask(cats, zones, prices)
    assert mask.tolist() == [True, True, False], mask


# ---------------------------------------------------------------------------
# Test 7: CategoryEq predicate correctness
# ---------------------------------------------------------------------------

def test_category_eq_predicate():
    cats  = np.array(["saree", "kurta", "saree"])
    zones = np.array(["Madurai", "Madurai", "Chennai"])
    prices = np.array([1000.0, 1000.0, 1000.0], dtype=np.float32)

    filt = AndFilter([CategoryEq("saree")])
    mask = filt.apply_mask(cats, zones, prices)
    assert mask.tolist() == [True, False, True], mask


# ---------------------------------------------------------------------------
# Test 8: ZoneEq predicate correctness
# ---------------------------------------------------------------------------

def test_zone_eq_predicate():
    cats  = np.array(["saree", "saree", "saree"])
    zones = np.array(["Madurai", "Chennai", "Madurai"])
    prices = np.array([1000.0, 1000.0, 1000.0], dtype=np.float32)

    filt = AndFilter([ZoneEq("Madurai")])
    mask = filt.apply_mask(cats, zones, prices)
    assert mask.tolist() == [True, False, True], mask


# ---------------------------------------------------------------------------
# Test 9: AndFilter with 3 predicates
# ---------------------------------------------------------------------------

def test_and_filter_three_predicates():
    cats  = np.array(["saree", "saree", "kurta", "saree"])
    zones = np.array(["Madurai", "Chennai", "Madurai", "Madurai"])
    prices = np.array([1200.0, 800.0, 700.0, 2000.0], dtype=np.float32)

    filt = AndFilter([CategoryEq("saree"), ZoneEq("Madurai"), PriceRange(0, 1500)])
    mask = filt.apply_mask(cats, zones, prices)
    assert mask.tolist() == [True, False, False, False], mask


# ---------------------------------------------------------------------------
# Test 10: AndFilter validates predicate count
# ---------------------------------------------------------------------------

def test_and_filter_rejects_0_predicates():
    with pytest.raises(ValueError):
        AndFilter([])


def test_and_filter_rejects_4_predicates():
    with pytest.raises(ValueError):
        AndFilter([CategoryEq("saree")] * 4)
