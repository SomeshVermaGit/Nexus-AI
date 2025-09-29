"""Speed benchmarking for vector search indices."""

import time
import numpy as np
from typing import List, Dict, Any
import json
from pathlib import Path


class SpeedBenchmark:
    """Benchmark search speed and throughput."""

    def __init__(self, index, name: str = "Index"):
        """Initialize benchmark.

        Args:
            index: Index to benchmark
            name: Name for the index
        """
        self.index = index
        self.name = name
        self.results = []

    def benchmark_indexing(
        self,
        vectors: np.ndarray,
        batch_sizes: List[int] = [1, 10, 100, 1000]
    ) -> Dict[str, Any]:
        """Benchmark indexing speed.

        Args:
            vectors: Vectors to index
            batch_sizes: Different batch sizes to test

        Returns:
            Benchmark results
        """
        results = {
            'name': self.name,
            'test': 'indexing',
            'batch_results': []
        }

        for batch_size in batch_sizes:
            num_vectors = min(batch_size, len(vectors))
            batch_vectors = vectors[:num_vectors]

            start_time = time.time()

            # Add vectors
            for vec in batch_vectors:
                self.index.add(vec)

            end_time = time.time()

            elapsed = end_time - start_time
            throughput = num_vectors / elapsed

            results['batch_results'].append({
                'batch_size': batch_size,
                'time_seconds': elapsed,
                'throughput_per_second': throughput
            })

            print(f"  Batch size {batch_size}: {elapsed:.3f}s ({throughput:.2f} vec/s)")

        self.results.append(results)
        return results

    def benchmark_search(
        self,
        query_vectors: np.ndarray,
        k_values: List[int] = [1, 10, 100]
    ) -> Dict[str, Any]:
        """Benchmark search speed.

        Args:
            query_vectors: Query vectors
            k_values: Different k values to test

        Returns:
            Benchmark results
        """
        results = {
            'name': self.name,
            'test': 'search',
            'k_results': []
        }

        for k in k_values:
            latencies = []

            for query in query_vectors:
                start_time = time.time()
                self.index.search(query, k=k)
                end_time = time.time()

                latencies.append((end_time - start_time) * 1000)  # ms

            avg_latency = np.mean(latencies)
            p50_latency = np.percentile(latencies, 50)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            qps = 1000 / avg_latency  # queries per second

            results['k_results'].append({
                'k': k,
                'avg_latency_ms': avg_latency,
                'p50_latency_ms': p50_latency,
                'p95_latency_ms': p95_latency,
                'p99_latency_ms': p99_latency,
                'qps': qps
            })

            print(f"  k={k}: avg={avg_latency:.2f}ms, p95={p95_latency:.2f}ms, QPS={qps:.2f}")

        self.results.append(results)
        return results

    def benchmark_concurrent_search(
        self,
        query_vectors: np.ndarray,
        num_threads: List[int] = [1, 2, 4, 8],
        k: int = 10
    ) -> Dict[str, Any]:
        """Benchmark concurrent search performance.

        Args:
            query_vectors: Query vectors
            num_threads: Different thread counts to test
            k: Number of results

        Returns:
            Benchmark results
        """
        import concurrent.futures

        results = {
            'name': self.name,
            'test': 'concurrent_search',
            'thread_results': []
        }

        def search_task(query):
            start = time.time()
            self.index.search(query, k=k)
            return time.time() - start

        for n_threads in num_threads:
            start_time = time.time()

            with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
                latencies = list(executor.map(search_task, query_vectors))

            end_time = time.time()

            total_time = end_time - start_time
            avg_latency = np.mean(latencies) * 1000  # ms
            throughput = len(query_vectors) / total_time

            results['thread_results'].append({
                'num_threads': n_threads,
                'total_time_seconds': total_time,
                'avg_latency_ms': avg_latency,
                'throughput_qps': throughput
            })

            print(f"  {n_threads} threads: {total_time:.2f}s, {throughput:.2f} QPS")

        self.results.append(results)
        return results

    def save_results(self, output_path: str) -> None:
        """Save benchmark results to file.

        Args:
            output_path: Path to save results
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to {output_path}")


def run_comparative_benchmark(
    indices: Dict[str, Any],
    dim: int = 768,
    n_vectors: int = 10000,
    n_queries: int = 100
) -> None:
    """Run comparative benchmark across multiple indices.

    Args:
        indices: Dictionary of index_name -> index_instance
        dim: Vector dimension
        n_vectors: Number of vectors to index
        n_queries: Number of query vectors
    """
    print(f"\n{'='*60}")
    print(f"Running Comparative Benchmark")
    print(f"{'='*60}")
    print(f"Vectors: {n_vectors}, Queries: {n_queries}, Dimension: {dim}\n")

    # Generate test data
    print("Generating test data...")
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # Normalize

    query_vectors = np.random.randn(n_queries, dim).astype(np.float32)
    query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)

    all_results = []

    for name, index in indices.items():
        print(f"\n{'='*60}")
        print(f"Benchmarking: {name}")
        print(f"{'='*60}\n")

        benchmark = SpeedBenchmark(index, name=name)

        # Index vectors
        print("Indexing benchmark:")
        indexing_results = benchmark.benchmark_indexing(vectors, batch_sizes=[100, 1000])

        # Search benchmark
        print("\nSearch benchmark:")
        search_results = benchmark.benchmark_search(query_vectors[:100], k_values=[1, 10, 50])

        # Save results
        benchmark.save_results(f"benchmarks/results/{name}_speed_benchmark.json")

        all_results.append(benchmark.results)

    # Save comparative results
    output_path = Path("benchmarks/results/comparative_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("Benchmark Complete!")
    print(f"{'='*60}\n")