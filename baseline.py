"""
Baseline Implementation - Customer Transaction & Risk Analytics Pipeline
Unoptimized version demonstrating common computational bottlenecks.
"""

import math
import random


def generate_data(seed=42, num_records=5000, num_values=600, num_points=80000):
    """Generate self-contained synthetic dataset with fixed seed."""
    rng = random.Random(seed)
    records = [
        {
            "id": i,
            "user_id": rng.randint(1, 200),
            "amount": rng.randint(10, 100),
            "category": f"cat_{rng.randint(1, 10)}",
            "tags": [f"tag_{rng.randint(1, 50)}" for _ in range(5)],
        }
        for i in range(num_records)
    ]
    values = [rng.randint(10, 22) for _ in range(num_values)]
    points = [(rng.random() * 100, rng.random() * 100) for _ in range(num_points)]
    query = (50.0, 50.0)
    return records, values, points, query


# Bottleneck 1: O(N^2) Nested Loop Duplicate Detection
def deduplicate_records(records):
    unique = []
    for r in records:
        is_dup = False
        for u in unique:
            if (
                r["user_id"] == u["user_id"]
                and r["amount"] == u["amount"]
                and r["category"] == u["category"]
            ):
                is_dup = True
                break
        if not is_dup:
            unique.append(r)
    return unique


# Bottleneck 2: Exponential O(2^N) Redundant Recursion without Memoization
def _risk_calc(n):
    if n <= 1:
        return 1.0
    return 0.6 * _risk_calc(n - 1) + 0.3 * _risk_calc(n - 2) + 0.05 * n


def compute_risk_scores(values):
    return [_risk_calc(v) for v in values]


# Bottleneck 3: O(N log N) Full Distance Sorting with Expensive math.sqrt
def find_nearest_anomalies(points, query, k=10):
    qx, qy = query
    distances = []
    for p in points:
        d = math.sqrt((p[0] - qx) ** 2 + (p[1] - qy) ** 2)
        distances.append((d, p))
    distances.sort(key=lambda item: item[0])
    return [p for _, p in distances[:k]]


# Bottleneck 4: O(U * T) Repeated List Scanning via .count() in Loop
def aggregate_tag_frequencies(records):
    all_tags = []
    for r in records:
        for t in r["tags"]:
            all_tags.append(t)
    unique_tags = []
    for t in all_tags:
        if t not in unique_tags:
            unique_tags.append(t)
    return {t: all_tags.count(t) for t in unique_tags}


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
    records, values, points, query = generate_data()
    res = run_pipeline(records, values, points, query)
    print(f"Baseline Done: {res['unique_count']} unique records, total risk {res['total_risk']:.2f}")
