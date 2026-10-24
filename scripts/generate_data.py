"""
Data-generation script for Problem 1 — Madurai Pincode Search.

Generation command (fixed seed, reproducible):
    python scripts/generate_data.py --n 50000 --dim 128 --seed 42 --out data/products.npz

Design choices
--------------
* 50,000 products, 128-dimensional L2-normalised embeddings.
* 20 product categories (saree, silk_saree, cotton_saree, kurta, dhoti,
  veshti, blouse, lehenga, salwar, kurti, lungi, pavadai, half_saree,
  chudidhar, pattu_saree, kanjivaram, bridal_saree, kids_wear, men_shirt,
  accessories).
* 30 delivery zones drawn from real Tamil Nadu districts.
* Embeddings are clustered (K = 60 vector clusters) so categories and zones
  are correlated with vector space — simulating real product images that look
  similar to each other within a category.
* Price range ₹200–₹15,000 drawn from a log-normal distribution anchored per
  category so "silk_saree" is expensive and "accessories" is cheap.
* Random seed = 42 everywhere.
"""

import argparse
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
N_CATEGORIES = 20
N_ZONES = 30
VECTOR_DIM = 128
N_VECTOR_CLUSTERS = 60   # internal latent clusters (not the IVF index clusters)

CATEGORIES = [
    "saree", "silk_saree", "cotton_saree", "kurta", "dhoti",
    "veshti", "blouse", "lehenga", "salwar", "kurti",
    "lungi", "pavadai", "half_saree", "chudidhar", "pattu_saree",
    "kanjivaram", "bridal_saree", "kids_wear", "men_shirt", "accessories",
]

ZONES = [
    "Madurai", "Chennai", "Coimbatore", "Tiruchirappalli", "Salem",
    "Tirunelveli", "Vellore", "Erode", "Thoothukudi", "Thanjavur",
    "Dindigul", "Kanchipuram", "Pudukottai", "Karur", "Namakkal",
    "Ramanathapuram", "Virudhunagar", "Sivaganga", "Theni", "Tiruppur",
    "Nagapattinam", "Cuddalore", "Villupuram", "Dharmapuri", "Krishnagiri",
    "Perambalur", "Ariyalur", "Tiruvannamalai", "Kallakurichi", "Ranipet",
]

# Price anchor (median ₹) per category — log-normal will spread around this
CATEGORY_PRICE_ANCHOR = {
    "saree": 1200, "silk_saree": 5000, "cotton_saree": 800,
    "kurta": 600, "dhoti": 400, "veshti": 350, "blouse": 300,
    "lehenga": 3500, "salwar": 700, "kurti": 500, "lungi": 250,
    "pavadai": 900, "half_saree": 1100, "chudidhar": 800,
    "pattu_saree": 4500, "kanjivaram": 8000, "bridal_saree": 12000,
    "kids_wear": 400, "men_shirt": 500, "accessories": 200,
}

# Zone size weight — Madurai and Chennai are large, small towns are tiny
ZONE_WEIGHTS_RAW = [
    12, 10, 9, 7, 6, 5, 5, 4, 4, 4,   # Madurai … Thanjavur
    3,  3,  3, 3, 3, 2, 2, 2, 2, 2,   # Dindigul … Tiruppur
    1,  1,  1, 1, 1, 1, 1, 1, 1, 1,   # Nagapattinam … Ranipet (tiny)
]


def _make_cluster_centres(rng: np.random.Generator, n_clusters: int, dim: int) -> np.ndarray:
    """Random L2-normalised cluster centres."""
    centres = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    norms = np.linalg.norm(centres, axis=1, keepdims=True)
    return centres / norms


def _category_cluster_map(n_categories: int, n_clusters: int, rng: np.random.Generator) -> dict:
    """Map each category → a small set of preferred vector clusters."""
    clusters_per_cat = max(1, n_clusters // n_categories)
    shuffled = rng.permutation(n_clusters)
    mapping = {}
    for i, cat in enumerate(CATEGORIES):
        start = (i * clusters_per_cat) % n_clusters
        assigned = shuffled[start:start + clusters_per_cat].tolist()
        if not assigned:
            assigned = [shuffled[0]]
        mapping[cat] = assigned
    return mapping


def _zone_cluster_map(n_zones: int, n_clusters: int, rng: np.random.Generator) -> dict:
    """Map each zone → a small set of preferred vector clusters (overlapping with categories)."""
    clusters_per_zone = max(1, n_clusters // n_zones)
    shuffled = rng.permutation(n_clusters)
    mapping = {}
    for i, zone in enumerate(ZONES):
        start = (i * clusters_per_zone) % n_clusters
        assigned = shuffled[start:start + clusters_per_zone].tolist()
        if not assigned:
            assigned = [shuffled[0]]
        mapping[zone] = assigned
    return mapping


def _sample_price(category: str, rng: np.random.Generator) -> float:
    """Log-normal price around the category anchor, clamped to [100, 20000]."""
    anchor = CATEGORY_PRICE_ANCHOR[category]
    sigma = 0.4
    raw = rng.lognormal(mean=np.log(anchor), sigma=sigma)
    return float(np.clip(raw, 100.0, 20_000.0))


def generate_dataset(n: int, dim: int, seed: int) -> dict:
    """
    Returns a dict with keys:
        vectors   : float32 (n, dim) — L2-normalised
        categories: list[str]
        zones     : list[str]
        prices    : float32 (n,)
        product_ids: list[str]
    """
    rng = np.random.default_rng(seed)

    # Build latent structure
    centres = _make_cluster_centres(rng, N_VECTOR_CLUSTERS, dim)
    cat_cluster_map = _category_cluster_map(N_CATEGORIES, N_VECTOR_CLUSTERS, rng)
    zone_cluster_map = _zone_cluster_map(N_ZONES, N_VECTOR_CLUSTERS, rng)

    # Zone frequency weights
    zone_weights = np.array(ZONE_WEIGHTS_RAW[:N_ZONES], dtype=float)
    zone_weights /= zone_weights.sum()

    vectors = np.empty((n, dim), dtype=np.float32)
    categories = []
    zones = []
    prices = np.empty(n, dtype=np.float32)
    product_ids = []

    # Sample categories proportionally (some categories are rarer)
    cat_weights = np.array([
        10, 8, 9, 7, 5, 4, 5, 6, 7, 7,
        4,  5, 6, 6, 7, 5, 3, 6, 6, 4,
    ], dtype=float)
    cat_weights /= cat_weights.sum()

    for i in range(n):
        # Choose category
        cat_idx = rng.choice(N_CATEGORIES, p=cat_weights)
        cat = CATEGORIES[cat_idx]

        # Choose zone (zone-weight determines delivery coverage)
        zone_idx = rng.choice(N_ZONES, p=zone_weights)
        zone = ZONES[zone_idx]

        # Choose a vector cluster — biased toward category's clusters (70%) + zone's clusters (30%)
        cat_clusters = cat_cluster_map[cat]
        zone_clusters = zone_cluster_map[zone]
        all_prefs = cat_clusters + zone_clusters
        cluster_id = rng.choice(all_prefs)

        centre = centres[cluster_id]
        noise = rng.standard_normal(dim).astype(np.float32) * 0.25
        vec = centre + noise
        vec /= (np.linalg.norm(vec) + 1e-9)
        vectors[i] = vec

        categories.append(cat)
        zones.append(zone)
        prices[i] = _sample_price(cat, rng)
        product_ids.append(f"PROD_{i:06d}")

    return {
        "vectors": vectors,
        "categories": categories,
        "zones": zones,
        "prices": prices,
        "product_ids": product_ids,
    }


def save_dataset(dataset: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        vectors=dataset["vectors"],
        categories=np.array(dataset["categories"]),
        zones=np.array(dataset["zones"]),
        prices=dataset["prices"],
        product_ids=np.array(dataset["product_ids"]),
    )
    print(f"Saved {len(dataset['product_ids'])} products -> {out_path}")

    # Print summary statistics
    cats, counts = np.unique(dataset["categories"], return_counts=True)
    print("\nCategory distribution:")
    for c, cnt in sorted(zip(cats, counts), key=lambda x: -x[1]):
        print(f"  {c:<20} {cnt:>6}")

    zns, zcounts = np.unique(dataset["zones"], return_counts=True)
    print("\nZone distribution (top 10):")
    for z, cnt in sorted(zip(zns, zcounts), key=lambda x: -x[1])[:10]:
        print(f"  {z:<25} {cnt:>6}")


def load_dataset(path: str) -> dict:
    data = np.load(path, allow_pickle=False)
    return {
        "vectors": data["vectors"],
        "categories": data["categories"].tolist(),
        "zones": data["zones"].tolist(),
        "prices": data["prices"],
        "product_ids": data["product_ids"].tolist(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic product dataset.")
    parser.add_argument("--n", type=int, default=50_000, help="Number of products")
    parser.add_argument("--dim", type=int, default=128, help="Vector dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default="data/products.npz", help="Output path")
    args = parser.parse_args()

    print(f"Generating {args.n} products, dim={args.dim}, seed={args.seed} …")
    dataset = generate_dataset(args.n, args.dim, args.seed)
    save_dataset(dataset, args.out)
    print("\nDone.")
