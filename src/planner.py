"""
planner.py - Strategy planner for filtered vector search.

Threshold derivation
--------------------
The planner maps estimated selectivity fraction to one of three strategies.
Thresholds were derived empirically from the benchmark measurements:

  BROAD_THRESHOLD  = 0.05   (5% of N)
    If > 5% of products match, filter is cheap; exact_scan is fast
    enough and avoids any recall loss from partition pruning.

  SELECTIVE_THRESHOLD = 0.005 (0.5% of N)
    If < 0.5% of products match, very few qualifying products exist;
    filter_aware_partition_search is best (traverse partitions checking the
    pre-built mask, stop early when enough candidates collected).

  In the middle (0.5%-5%): partition_then_filter works well - few enough
    partitions to probe, enough qualifying candidates after filtering.

Strategy selection
------------------
  selectivity >= 5%    -> exact_scan
  0.5% <= sel < 5%     -> partition_then_filter
  sel < 0.5%           -> filter_aware_partition_search

nprobe selection
----------------
  nprobe scales with the NUMBER OF INDEX PARTITIONS (n_clusters), not a
  fixed constant.  Low-selectivity queries need to cover a larger fraction
  of the index to find enough qualifying candidates:

    nprobe_frac = min(1.0, NPROBE_BASE_FRAC / max(sel, 1e-4))
    nprobe = max(MIN_NPROBE, min(n_clusters, round(nprobe_frac * n_clusters)))

  This ensures that on a 50-partition test index as well as on a 224-partition
  production index the recall target is met: the nprobe scales with the
  actual cluster count rather than being a hard-coded integer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.ivf_index import IVFIndex, SearchResult
from src.metadata_filter import AndFilter
from src.stats_model import StatsModel


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
BROAD_THRESHOLD = 0.05         # >= 5%  -> exact_scan
SELECTIVE_THRESHOLD = 0.005    # <  0.5% -> filter_aware_partition_search

# nprobe as a fraction of total partitions, scaled by 1/selectivity
# At sel=1% this yields 0.20/0.01 = 20% of partitions
# At sel=0.1% this yields 0.20/0.001 = capped at 100% of partitions
NPROBE_BASE_FRAC = 0.20
MIN_NPROBE = 4


@dataclass
class PlannerDecision:
    strategy: str
    nprobe: int
    estimated_count: int
    estimated_selectivity: float


class Planner:
    """
    Chooses among the three search strategies using selectivity estimation.

    Parameters
    ----------
    broad_threshold : float
        Fraction above which exact_scan is preferred.
    selective_threshold : float
        Fraction below which filter_aware_partition_search is used.
    """

    def __init__(
        self,
        index: IVFIndex,
        stats: StatsModel,
        broad_threshold: float = BROAD_THRESHOLD,
        selective_threshold: float = SELECTIVE_THRESHOLD,
    ):
        self.index = index
        self.stats = stats
        self.broad_threshold = broad_threshold
        self.selective_threshold = selective_threshold

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def decide(self, filt: Optional[AndFilter]) -> PlannerDecision:
        """Estimate selectivity and choose a strategy."""
        est_count = self.stats.estimate_count(filt)
        est_sel = est_count / max(1, self.stats.total)

        if est_sel >= self.broad_threshold:
            strategy = "exact_scan"
            nprobe = MIN_NPROBE   # not used for exact scan
        elif est_sel >= self.selective_threshold:
            strategy = "partition_then_filter"
            nprobe = self._calc_nprobe(est_sel)
        else:
            strategy = "filter_aware_partition_search"
            nprobe = self._calc_nprobe(est_sel)

        return PlannerDecision(
            strategy=strategy,
            nprobe=nprobe,
            estimated_count=est_count,
            estimated_selectivity=est_sel,
        )

    def _calc_nprobe(self, sel: float) -> int:
        """
        Scale nprobe as a fraction of n_clusters, inversely proportional
        to selectivity, clamped to [MIN_NPROBE, n_clusters].
        """
        n_clusters = self.index.n_clusters
        if sel <= 0:
            return n_clusters
        frac = min(1.0, NPROBE_BASE_FRAC / max(sel, 1e-4))
        raw = round(frac * n_clusters)
        return int(np.clip(raw, MIN_NPROBE, n_clusters))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        filt: Optional[AndFilter],
        k: int = 10,
    ) -> SearchResult:
        """Run the planner-selected strategy."""
        decision = self.decide(filt)

        if decision.strategy == "exact_scan":
            result = self.index.exact_scan(query, filt, k)
        elif decision.strategy == "partition_then_filter":
            result = self.index.partition_then_filter(
                query, filt, k, nprobe=decision.nprobe
            )
        else:
            result = self.index.filter_aware_partition_search(
                query, filt, k, nprobe=decision.nprobe
            )

        # Tag the result with planner-annotated strategy name
        result.strategy = (
            f"planner->{decision.strategy}"
            f"(nprobe={decision.nprobe}, est_sel={decision.estimated_selectivity:.4f})"
        )
        return result

    def explain(self, filt: Optional[AndFilter]) -> str:
        """Human-readable explanation of the planner decision."""
        d = self.decide(filt)
        lines = [
            f"Filter      : {filt}",
            f"Est. count  : {d.estimated_count} / {self.stats.total}",
            f"Est. sel.   : {d.estimated_selectivity:.4%}",
            f"Strategy    : {d.strategy}",
            f"nprobe      : {d.nprobe} / {self.index.n_clusters} partitions",
        ]
        return "\n".join(lines)
