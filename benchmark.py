"""
Benchmark & Profiling Suite
Performs:
1. Automated output verification (correctness check)
2. cProfile profiling for baseline and optimized code
3. Execution time benchmarking with time.perf_counter()
4. Peak memory measurement using tracemalloc
"""

import cProfile
import io
import pstats
import time
import tracemalloc

import baseline
import optimized


def verify_equivalence():
    """Verify that optimized produces exact same outputs as baseline."""
    print("=" * 70)
    print("STEP 1: AUTOMATED VERIFICATION OF FUNCTIONAL EQUIVALENCE")
    print("=" * 70)

    records, values, points, query = baseline.generate_data(seed=42)

    res_base = baseline.run_pipeline(records, values, points, query)
    res_opt = optimized.run_pipeline(records, values, points, query)

    # 1. Deduplication check
    assert (
        res_base["unique_count"] == res_opt["unique_count"]
    ), f"Dedup mismatch: {res_base['unique_count']} vs {res_opt['unique_count']}"

    # 2. Risk scores check
    assert (
        abs(res_base["total_risk"] - res_opt["total_risk"]) < 1e-6
    ), f"Risk mismatch: {res_base['total_risk']} vs {res_opt['total_risk']}"

    # 3. Nearest anomalies check
    assert (
        res_base["nearest_anomalies"] == res_opt["nearest_anomalies"]
    ), f"Anomalies mismatch: {res_base['nearest_anomalies']} vs {res_opt['nearest_anomalies']}"

    # 4. Tag frequencies check
    assert (
        res_base["tag_frequencies"] == res_opt["tag_frequencies"]
    ), f"Tags mismatch: {res_base['tag_frequencies']} vs {res_opt['tag_frequencies']}"

    print("[PASSED] Output equivalence verified across all 4 pipeline stages.")
    print(f"         - Unique records matched: {res_base['unique_count']}")
    print(f"         - Total risk sum matched: {res_base['total_risk']:.4f}")
    print(f"         - Top nearest anomalies matched: {len(res_base['nearest_anomalies'])} points")
    print(f"         - Unique tag categories matched: {len(res_base['tag_frequencies'])} tags")
    print("         Zero regression detected between baseline and optimized.\n")


def profile_code(name, pipeline_func, records, values, points, query):
    """Run cProfile and print top bottleneck functions."""
    print("=" * 70)
    print(f"STEP 2: cProfile PROFILING - {name.upper()}")
    print("=" * 70)

    pr = cProfile.Profile()
    pr.enable()
    pipeline_func(records, values, points, query)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(15)
    print(s.getvalue())


def measure_memory(pipeline_func, records, values, points, query):
    """Measure peak memory usage in megabytes using tracemalloc."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    pipeline_func(records, values, points, query)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def benchmark_components(runs=3):
    """Measure runtime per component and overall speedup."""
    print("=" * 70)
    print(f"STEP 3: COMPONENT-LEVEL BENCHMARKING (Averaged over {runs} runs)")
    print("=" * 70)

    records, values, points, query = baseline.generate_data(seed=42)

    modules = [("Baseline", baseline), ("Optimized", optimized)]
    results = {}

    for name, mod in modules:
        t_dedup, t_risk, t_knn, t_tags, t_total = [], [], [], [], []

        for _ in range(runs):
            # 1. Dedup
            t0 = time.perf_counter()
            mod.deduplicate_records(records)
            t_dedup.append(time.perf_counter() - t0)

            # 2. Risk
            t0 = time.perf_counter()
            mod.compute_risk_scores(values)
            t_risk.append(time.perf_counter() - t0)

            # 3. KNN
            t0 = time.perf_counter()
            mod.find_nearest_anomalies(points, query)
            t_knn.append(time.perf_counter() - t0)

            # 4. Tags
            t0 = time.perf_counter()
            mod.aggregate_tag_frequencies(records)
            t_tags.append(time.perf_counter() - t0)

            # Total
            t0 = time.perf_counter()
            mod.run_pipeline(records, values, points, query)
            t_total.append(time.perf_counter() - t0)

        results[name] = {
            "dedup": sum(t_dedup) / runs,
            "risk": sum(t_risk) / runs,
            "knn": sum(t_knn) / runs,
            "tags": sum(t_tags) / runs,
            "total": sum(t_total) / runs,
            "mem_mb": measure_memory(mod.run_pipeline, records, values, points, query),
        }

    b = results["Baseline"]
    o = results["Optimized"]

    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Pipeline Component':<25} | {'Baseline (s)':<14} | {'Optimized (s)':<14} | {'Speedup':<12}"
    print(header)
    print("-" * 80)

    components = [
        ("1. Record Deduplication", "dedup"),
        ("2. Risk Calculation", "risk"),
        ("3. Anomaly Detection (KNN)", "knn"),
        ("4. Tag Frequency Count", "tags"),
        ("FULL PIPELINE EXECUTION", "total"),
    ]

    for label, key in components:
        base_time = b[key]
        opt_time = o[key]
        speedup = base_time / opt_time if opt_time > 0 else float("inf")
        print(f"{label:<25} | {base_time:<14.5f} | {opt_time:<14.5f} | {speedup:<10.2f}x")

    print("-" * 80)
    print(f"{'Peak Memory Usage (MB)':<25} | {b['mem_mb']:<14.2f} | {o['mem_mb']:<14.2f} | {b['mem_mb']/o['mem_mb'] if o['mem_mb']>0 else 1.0:<10.2f}x")
    print("=" * 80)


if __name__ == "__main__":
    records, values, points, query = baseline.generate_data(seed=42)

    # 1. Verification
    verify_equivalence()

    # 2. cProfile Profiling
    profile_code("Baseline", baseline.run_pipeline, records, values, points, query)
    profile_code("Optimized", optimized.run_pipeline, records, values, points, query)

    # 3. Component Benchmarking
    benchmark_components(runs=3)
