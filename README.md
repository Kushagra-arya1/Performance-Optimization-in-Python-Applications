# Performance Optimization in Python Applications

**Repository:** `Performance-Optimization-in-Python-Applications`  
**Environment:** Python 3.14.6 | Windows 11 x64  
**Deliverables:** Self-contained code files (`baseline.py`, `optimized.py`, `benchmark.py`), [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md), and [performance_optimization.zip](performance_optimization.zip).

---

## 1. Executive Summary

This project demonstrates rigorous identification, diagnosis, and resolution of computational bottlenecks in a Python-based data processing and analytics pipeline. The workload consists of four core computational tasks:
1. **Record Deduplication** (Identity & duplicate matching across transactions)
2. **Compound Risk Scoring** (Recursive evaluation over mathematical recurrence models)
3. **Spatial Anomaly Detection** (K-Nearest Neighbors query against 80,000 coordinate vectors)
4. **Tag Frequency Aggregation** (Multi-label category tallying across records)

Applying core computer science principles—hash indexing, memoization, bounded-heap selection, and native C-accelerated standard library counters—resulted in an overall **45.41x end-to-end execution speedup** and an **89.8% reduction in peak memory consumption**, with **100% verified functional equivalence (zero regressions)**.

### Performance Summary
| Metric | Baseline (Unoptimized) | Optimized | Improvement / Speedup |
| :--- | :--- | :--- | :--- |
| **Full Pipeline Runtime** | **6.6480 s** | **0.1464 s** | **45.41x faster** |
| **Total Function Calls (cProfile)** | **6,911,534** | **114,971** | **98.34% reduction** |
| **Peak Memory Allocation** | **8.55 MB** | **0.87 MB** | **9.88x more efficient** |
| **Correctness Verification** | Baseline Reference | Identical Output | **100% Passed (0 regression)** |

---

## 2. Benchmark Comparison Table

Measurements averaged across runs using `time.perf_counter()` and `tracemalloc`:

| Pipeline Component | Baseline Time (s) | Optimized Time (s) | Speedup Factor | Complexity Shift |
| :--- | :--- | :--- | :--- | :--- |
| **1. Record Deduplication** | 3.78500 s | 0.01271 s | **297.77x** | $O(N^2) \to O(N)$ |
| **2. Risk Calculation** | 2.35855 s | 0.00053 s | **4,412.63x** | $O(2^N) \to O(1)$ cached |
| **3. Anomaly Detection (KNN)** | 0.30826 s | 0.11846 s | **2.60x** | $O(M \log M) \to O(M \log k)$ |
| **4. Tag Frequency Count** | 0.17039 s | 0.01442 s | **11.82x** | $O(U \cdot T) \to O(T)$ |
| **FULL PIPELINE TOTAL** | **6.64795 s** | **0.14638 s** | **45.41x** | — |
| **Peak Memory Usage (MB)** | **8.55 MB** | **0.87 MB** | **9.88x** | — |

---

## 3. Project Structure

```text
├── baseline.py                 # Original unoptimized implementation with 4 bottlenecks
├── optimized.py                # High-performance refactored implementation
├── benchmark.py                # Automated verification, cProfile, timing, and tracemalloc suite
├── PERFORMANCE_REPORT.md       # Detailed technical engineering report
├── performance_optimization.zip# Self-contained archive deliverable
└── README.md                   # Repository overview and instructions
```

---

## 4. Optimization Details

1. **Record Deduplication (`deduplicate_records`):**
   - *Baseline:* $O(N^2)$ linear scanning over a dynamic list of unique records.
   - *Optimized:* $O(N)$ hash-set membership checking via a tuple key `(user_id, amount, category)`.

2. **Risk Scoring (`compute_risk_scores`):**
   - *Baseline:* Redundant binary recursion generating 6,641,498 function calls for 600 evaluations.
   - *Optimized:* Memoization via `@functools.lru_cache(maxsize=128)` achieving $O(1)$ lookups after initial computation.

3. **Anomaly Detection (`find_nearest_anomalies`):**
   - *Baseline:* Expensive `math.sqrt()` float operations + full TimSort on 80,000 points.
   - *Optimized:* Monotonic squared Euclidean distance $(x_1 - x_2)^2 + (y_1 - y_2)^2$ + bounded max-heap selection via `heapq.nsmallest(k, points)`.

4. **Tag Frequency Aggregation (`aggregate_tag_frequencies`):**
   - *Baseline:* Iterative `.count()` over all tokens inside a loop over unique tags ($O(U \cdot T)$).
   - *Optimized:* Single-pass C-implemented `collections.Counter(tag for r in records for tag in r["tags"])`.

---

## 5. Verification & How to Run

The project is completely self-contained with no external package dependencies.

### Run Verification & Full Profiling:
```bash
python benchmark.py
```

### Run Standalone Modules:
```bash
python baseline.py
python optimized.py
```
