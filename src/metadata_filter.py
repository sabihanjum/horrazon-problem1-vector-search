"""
metadata_filter.py - Metadata predicate evaluation for product records.

Supported predicates
--------------------
  CategoryEq(category)          : product.category == category
  ZoneEq(zone)                  : product.zone == zone
  PriceRange(lo, hi)            : lo <= product.price <= hi

Up to 3 predicates can be AND-ed with AndFilter([p1, p2, p3]).

Usage
-----
    from src.metadata_filter import CategoryEq, ZoneEq, PriceRange, AndFilter
    f = AndFilter([CategoryEq("saree"), ZoneEq("Tamil Nadu"), PriceRange(0, 1500)])
    mask = f.apply_mask(categories, zones, prices)   # boolean array
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Predicate(ABC):
    """Abstract base for a single filter predicate."""

    @abstractmethod
    def apply_mask(
        self,
        categories: np.ndarray,  # str array (N,)
        zones: np.ndarray,        # str array (N,)
        prices: np.ndarray,       # float32  (N,)
    ) -> np.ndarray:
        """Return boolean mask of shape (N,) -- True means 'passes filter'."""

    @abstractmethod
    def describe(self) -> str:
        ...

    def __repr__(self) -> str:
        return self.describe()


# ---------------------------------------------------------------------------
# Concrete predicates
# ---------------------------------------------------------------------------

@dataclass
class CategoryEq(Predicate):
    """Equality filter on category field."""
    category: str

    def apply_mask(self, categories, zones, prices) -> np.ndarray:
        if isinstance(categories, list):
            categories = np.array(categories)
        return categories == self.category

    def describe(self) -> str:
        return f"category == '{self.category}'"


@dataclass
class ZoneEq(Predicate):
    """Equality filter on delivery-zone field."""
    zone: str

    def apply_mask(self, categories, zones, prices) -> np.ndarray:
        if isinstance(zones, list):
            zones = np.array(zones)
        return zones == self.zone

    def describe(self) -> str:
        return f"zone == '{self.zone}'"


@dataclass
class PriceRange(Predicate):
    """Inclusive price range filter [lo, hi]."""
    lo: float
    hi: float

    def __post_init__(self):
        if self.lo > self.hi:
            raise ValueError(f"PriceRange: lo={self.lo} > hi={self.hi}")

    def apply_mask(self, categories, zones, prices) -> np.ndarray:
        return (prices >= self.lo) & (prices <= self.hi)

    def describe(self) -> str:
        return f"price in [{self.lo:.0f}, {self.hi:.0f}]"


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

class AndFilter:
    """
    AND-join of up to 3 predicates.

    Predicates are applied left-to-right for short-circuit efficiency:
    the cheapest predicate (equality checks) should come before range checks.
    """

    def __init__(self, predicates: List[Predicate]):
        if len(predicates) < 1 or len(predicates) > 3:
            raise ValueError("AndFilter accepts 1-3 predicates.")
        self.predicates = predicates

    def apply_mask(
        self,
        categories: np.ndarray,
        zones: np.ndarray,
        prices: np.ndarray,
    ) -> np.ndarray:
        n = len(prices)
        mask = np.ones(n, dtype=bool)
        for pred in self.predicates:
            partial = pred.apply_mask(categories, zones, prices)
            mask &= partial
            # early bail-out: if nothing passes, stop
            if not mask.any():
                break
        return mask

    def apply_to_indices(
        self,
        indices: np.ndarray,  # int array of candidate product indices
        categories: np.ndarray,
        zones: np.ndarray,
        prices: np.ndarray,
    ) -> np.ndarray:
        """Filter a small candidate set by index -- used after partition search."""
        cat_sub = np.asarray(categories)[indices]
        zone_sub = np.asarray(zones)[indices]
        price_sub = np.asarray(prices)[indices]
        local_mask = self.apply_mask(cat_sub, zone_sub, price_sub)
        return indices[local_mask]

    def describe(self) -> str:
        return " AND ".join(p.describe() for p in self.predicates)

    def __repr__(self) -> str:
        return f"AndFilter({self.describe()})"
