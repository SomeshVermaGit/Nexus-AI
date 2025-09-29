"""Recall benchmarking for approximate nearest neighbor search."""

import numpy as np
from typing import List, Dict, Any, Tuple
import json
from pathlib import Path


class RecallBenchmark:
    """Benchmark search recall (accuracy) for approximate methods."""

    def __init__(self, index, name: str = "Index"):
        """Initialize recall benchmark.

        Args:
            index: Index to benchmark
            name: Name for the index
        """
        self.index = index
        self.name = name
        self.results = []

    def compute_ground_truth(
        self,
        query_vectors: np.ndarray,
        index_vectors: np.ndarray,
        k: int = 100
    ) -> List[List[int]]:
        """Compute ground truth nearest neighbors using brute force.

        Args:
            query_vectors: Query vectors (N, dim)
            index_vectors: Indexed vectors (M, dim)
            k: Number of neighbors

        Returns:
            List of neighbor indices for each query
        """
        print(f"Computing ground truth for {len(query_vectors)} queries...")

        ground_truth = []

        for query in query_vectors:
            # Compute distances to all vectors
            distances = np.linalg.norm(index_vectors - query, axis=1)
            # Get top k indices
            top_k_indices = np.argsort(distances)[:k]
            ground_truth.append(top_k_indices.tolist())

        return ground_truth

    def compute_recall_at_k(
        self,
        retrieved: List[int],
        ground_truth: List[int],
        k: int
    ) -> float:
        """Compute recall@k metric.

        Args:
            retrieved: Retrieved neighbor IDs
            ground_truth: Ground truth neighbor IDs
            k: Number of results to consider

        Returns:
            Recall score
        """
        retrieved_set = set(retrieved[:k])
        ground_truth_set = set(ground_truth[:k])

        if not ground_truth_set:
            return 0.0

        intersection = retrieved_set.intersection(ground_truth_set)
        return len(intersection) / len(ground_truth_set)

    def benchmark_recall(
        self,
        query_vectors: np.ndarray,
        ground_truth: List[List[int]],
        k_values: List[int] = [1, 10, 50, 100]
    ) -> Dict[str, Any]:
        """Benchmark recall at different k values.

        Args:
            query_vectors: Query vectors
            ground_truth: Ground truth neighbors
            k_values: Different k values to test

        Returns:
            Benchmark results
        """
        results = {
            'name': self.name,
            'test': 'recall',
            'k_results': []
        }

        for k in k_values:
            recalls = []

            for query, gt in zip(query_vectors, ground_truth):
                # Search using index
                search_results = self.index.search(query, k=k)
                retrieved_ids = [doc_id for doc_id, _, _ in search_results]

                # Compute recall
                recall = self.compute_recall_at_k(retrieved_ids, gt, k)
                recalls.append(recall)

            avg_recall = np.mean(recalls)
            min_recall = np.min(recalls)
            max_recall = np.max(recalls)

            results['k_results'].append({
                'k': k,
                'recall': avg_recall,
                'min_recall': min_recall,
                'max_recall': max_recall
            })

            print(f"  Recall@{k}: {avg_recall:.4f} (min={min_recall:.4f}, max={max_recall:.4f})")

        self.results.append(results)
        return results

    def benchmark_recall_vs_speed(
        self,
        query_vectors: np.ndarray,
        ground_truth: List[List[int]],
        param_ranges: Dict[str, List],
        k: int = 10
    ) -> Dict[str, Any]:
        """Benchmark recall vs speed tradeoff.

        Args:
            query_vectors: Query vectors
            ground_truth: Ground truth neighbors
            param_ranges: Parameter ranges to test (e.g., {'ef_search': [10, 50, 100]})
            k: Number of results

        Returns:
            Benchmark results
        """
        import time

        results = {
            'name': self.name,
            'test': 'recall_vs_speed',
            'param_results': []
        }

        for param_name, param_values in param_ranges.items():
            for param_value in param_values:
                # Set parameter
                if hasattr(self.index, param_name):
                    setattr(self.index, param_name, param_value)

                recalls = []
                latencies = []

                for query, gt in zip(query_vectors, ground_truth):
                    start = time.time()
                    search_results = self.index.search(query, k=k)
                    latency = (time.time() - start) * 1000  # ms

                    retrieved_ids = [doc_id for doc_id, _, _ in search_results]
                    recall = self.compute_recall_at_k(retrieved_ids, gt, k)

                    recalls.append(recall)
                    latencies.append(latency)

                avg_recall = np.mean(recalls)
                avg_latency = np.mean(latencies)

                results['param_results'].append({
                    'parameter': param_name,
                    'value': param_value,
                    'recall': avg_recall,
                    'avg_latency_ms': avg_latency
                })

                print(f"  {param_name}={param_value}: recall={avg_recall:.4f}, latency={avg_latency:.2f}ms")

        self.results.append(results)
        return results

    def save_results(self, output_path: str) -> None:
        """Save benchmark results.

        Args:
            output_path: Path to save results
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to {output_path}")


def run_recall_benchmark(
    index,
    name: str,
    dim: int = 768,
    n_index_vectors: int = 10000,
    n_queries: int = 100,
    k_values: List[int] = [1, 10, 50, 100]
) -> None:
    """Run recall benchmark for an index.

    Args:
        index: Index to benchmark
        name: Index name
        dim: Vector dimension
        n_index_vectors: Number of vectors to index
        n_queries: Number of query vectors
        k_values: K values to test
    """
    print(f"\n{'='*60}")
    print(f"Running Recall Benchmark: {name}")
    print(f"{'='*60}")
    print(f"Index vectors: {n_index_vectors}, Queries: {n_queries}, Dimension: {dim}\n")

    # Generate test data
    print("Generating test data...")
    index_vectors = np.random.randn(n_index_vectors, dim).astype(np.float32)
    index_vectors = index_vectors / np.linalg.norm(index_vectors, axis=1, keepdims=True)

    query_vectors = np.random.randn(n_queries, dim).astype(np.float32)
    query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)

    # Index vectors
    print("Indexing vectors...")
    for i, vec in enumerate(index_vectors):
        index.add(vec, node_id=i)

    # Compute ground truth
    benchmark = RecallBenchmark(index, name=name)
    ground_truth = benchmark.compute_ground_truth(query_vectors, index_vectors, k=max(k_values))

    # Benchmark recall
    print("\nRecall benchmark:")
    benchmark.benchmark_recall(query_vectors, ground_truth, k_values=k_values)

    # Save results
    benchmark.save_results(f"benchmarks/results/{name}_recall_benchmark.json")

    print(f"\n{'='*60}")
    print("Benchmark Complete!")
    print(f"{'='*60}\n")