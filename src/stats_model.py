"""
stats_model.py — Selectivity estimation statistics.

Structures built at index-build time:
  * category_counts        : {cat: int}
  * zone_counts            : {zone: int}
  * cat_zone_counts        : {(cat, zone): int}
  * price_histogram        : (bin_edges float64, bin_counts int64)
  * total                  : int

Used by the Planner to estimate how many products a filter will match
WITHOUT scanning the full dataset.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.metadata_filter import AndFilter, CategoryEq, ZoneEq, PriceRange


N_PRICE_BINS = 50


class StatsModel:
    """Pre-computed selectivity statistics."""

    def __init__(self):
        self.total: int = 0
        self.category_counts: Dict[str, int] = {}
        self.zone_counts: Dict[str, int] = {}
        self.cat_zone_counts: Dict[Tuple[str, str], int] = {}
        self.price_bin_edges: Optional[np.ndarray] = None    # (N_BINS+1,)
        self.price_bin_counts: Optional[np.ndarray] = None   # (N_BINS,)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        categories: List[str],
        zones: List[str],
        prices: np.ndarray,
    ) -> None:
        self.total = len(prices)

        # Category counts
        cats, cnts = np.unique(categories, return_counts=True)
        self.category_counts = dict(zip(cats.tolist(), cnts.tolist()))

        # Zone counts
        zns, zcnts = np.unique(zones, return_counts=True)
        self.zone_counts = dict(zip(zns.tolist(), zcnts.tolist()))

        # Joint category × zone counts
        cat_arr = np.array(categories)
        zone_arr = np.array(zones)
        unique_cats = list(self.category_counts.keys())
        unique_zones = list(self.zone_counts.keys())
        for cat in unique_cats:
            cat_mask = cat_arr == cat
            for zone in unique_zones:
                cnt = int(np.sum(cat_mask & (zone_arr == zone)))
                if cnt:
                    self.cat_zone_counts[(cat, zone)] = cnt

        # Price histogram
        self.price_bin_edges, self.price_bin_counts = self._build_price_hist(prices)

    def _build_price_hist(
        self, prices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        edges = np.percentile(prices, np.linspace(0, 100, N_PRICE_BINS + 1))
        edges = np.unique(edges)
        counts, _ = np.histogram(prices, bins=edges)
        return edges.astype(np.float64), counts.astype(np.int64)

    # ------------------------------------------------------------------
    # Estimate
    # ------------------------------------------------------------------

    def estimate_count(self, filt: Optional[AndFilter]) -> int:
        """
        Estimate the number of products that satisfy the filter.
        Uses independence assumption for combined predicates, then
        applies cat_zone_counts for the common (cat AND zone) case.
        """
        if filt is None:
            return self.total

        cat_pred: Optional[CategoryEq] = None
        zone_pred: Optional[ZoneEq] = None
        price_pred: Optional[PriceRange] = None

        for p in filt.predicates:
            if isinstance(p, CategoryEq):
                cat_pred = p
            elif isinstance(p, ZoneEq):
                zone_pred = p
            elif isinstance(p, PriceRange):
                price_pred = p

        # Start from total
        est = float(self.total)

        # --- Category × Zone joint estimate (avoids independence error) ---
        if cat_pred and zone_pred:
            key = (cat_pred.category, zone_pred.zone)
            joint = self.cat_zone_counts.get(key, 0)
            est = float(joint)
            cat_pred = None   # already consumed
            zone_pred = None
        elif cat_pred:
            est = float(self.category_counts.get(cat_pred.category, 0))
            cat_pred = None
        elif zone_pred:
            est = float(self.zone_counts.get(zone_pred.zone, 0))
            zone_pred = None

        # --- Price range: apply selectivity fraction on top ---
        if price_pred is not None:
            price_sel = self._price_selectivity(price_pred.lo, price_pred.hi)
            est = est * price_sel

        return max(1, int(round(est)))

    def _price_selectivity(self, lo: float, hi: float) -> float:
        """Fraction of products in [lo, hi] estimated from histogram."""
        if self.price_bin_edges is None:
            return 1.0
        edges = self.price_bin_edges
        counts = self.price_bin_counts
        total_in_hist = counts.sum()
        if total_in_hist == 0:
            return 1.0

        # Bin indices that overlap [lo, hi]
        # Each bin is (edges[i], edges[i+1]]
        n_bins = len(counts)
        overlap_count = 0
        for i in range(n_bins):
            bin_lo = edges[i]
            bin_hi = edges[i + 1]
            if bin_hi < lo or bin_lo > hi:
                continue
            # Full bin
            if lo <= bin_lo and bin_hi <= hi:
                overlap_count += counts[i]
            else:
                # Partial bin — linear interpolation
                bin_width = bin_hi - bin_lo
                overlap = min(hi, bin_hi) - max(lo, bin_lo)
                frac = overlap / bin_width if bin_width > 0 else 0.0
                overlap_count += counts[i] * frac

        return float(overlap_count) / float(total_in_hist)

    def selectivity_fraction(self, filt: Optional[AndFilter]) -> float:
        """Estimated fraction of total products matching the filter."""
        return self.estimate_count(filt) / max(1, self.total)
