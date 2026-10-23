"""
Optimized Implementation - Customer Transaction & Risk Analytics Pipeline
High-performance version utilizing optimal data structures and algorithms.
"""

import heapq
from collections import Counter
from functools import lru_cache


# Optimization 1: O(N) Hash-Set Lookup (vs O(N^2) Nested Loop)
def deduplicate_records(records):
    seen = set()
    unique = []
    for r in records:
        key = (r["user_id"], r["amount"], r["category"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


# Optimization 2: Memoization with @lru_cache (Eliminates O(2^N) Redundant Recomputation)
@lru_cache(maxsize=128)
def _risk_calc(n):
    if n <= 1:
        return 1.0
    return 0.6 * _risk_calc(n - 1) + 0.3 * _risk_calc(n - 2) + 0.05 * n


def compute_risk_scores(values):
    return [_risk_calc(v) for v in values]


# Optimization 3: O(N log k) Heap Selection & Squared Euclidean Distance (Avoids Sqrt & Full Sort)
def find_nearest_anomalies(points, query, k=10):
    qx, qy = query
    # Monotonic squared distance: preserves order without math.sqrt
    return heapq.nsmallest(
        k,
        points,
        key=lambda p: (p[0] - qx) ** 2 + (p[1] - qy) ** 2,
    )


# Optimization 4: Single-Pass O(T) Aggregation with C-Optimized Counter (vs O(U*T) .count())
def aggregate_tag_frequencies(records):
    return dict(Counter(tag for r in records for tag in r["tags"]))


def run_pipeline(records, values, points, query, k=10):
    unique = deduplicate_records(records)
    scores = compute_risk_scores(values)
    nearest = find_nearest_anomalies(points, query, k)
    tags = aggregate_tag_frequencies(records)
    return {
        "unique_count": len(unique),
        "total_risk": sum(scores),
        "nearest_anomalies": nearest,
        "tag_frequencies": tags,
    }


if __name__ == "__main__":
    from baseline import generate_data

    records, values, points, query = generate_data()
    res = run_pipeline(records, values, points, query)
    print(f"Optimized Done: {res['unique_count']} unique records, total risk {res['total_risk']:.2f}")
