"""Inverted File Index (IVF) implementation.

IVF partitions the vector space using clustering (k-means) and creates
an inverted index mapping cluster centroids to vectors.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import pickle
from sklearn.cluster import KMeans


@dataclass
class IVFCell:
    """A cell in the inverted file index."""
    centroid: np.ndarray
    vector_ids: List[int]
    vectors: List[np.ndarray]


class IVFIndex:
    """Inverted File Index for fast approximate nearest neighbor search.

    Uses k-means clustering to partition the vector space and searches
    only the most relevant clusters.
    """

    def __init__(
        self,
        dim: int,
        n_clusters: int = 100,
        n_probe: int = 1,
        distance_metric: str = "cosine",
        use_pq: bool = False,
        pq_subvectors: int = 8
    ):
        """Initialize IVF index.

        Args:
            dim: Vector dimension
            n_clusters: Number of clusters (voronoi cells)
            n_probe: Number of clusters to search
            distance_metric: Distance metric ('cosine', 'euclidean', 'dot')
            use_pq: Whether to use product quantization for compression
            pq_subvectors: Number of subvectors for PQ
        """
        self.dim = dim
        self.n_clusters = n_clusters
        self.n_probe = n_probe
        self.distance_metric = distance_metric
        self.use_pq = use_pq
        self.pq_subvectors = pq_subvectors

        # Index structures
        self.centroids: Optional[np.ndarray] = None
        self.cells: Dict[int, IVFCell] = {}
        self.metadata_store: Dict[int, Dict] = {}
        self.next_id: int = 0
        self.is_trained: bool = False

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute distance between two vectors."""
        if self.distance_metric == "cosine":
            return 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        elif self.distance_metric == "euclidean":
            return np.linalg.norm(a - b)
        elif self.distance_metric == "dot":
            return -np.dot(a, b)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def _distance_batch(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute distances to multiple vectors efficiently."""
        if self.distance_metric == "cosine":
            # Normalize
            query_norm = query / (np.linalg.norm(query) + 1e-8)
            vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
            similarities = np.dot(vectors_norm, query_norm)
            return 1.0 - similarities
        elif self.distance_metric == "euclidean":
            return np.linalg.norm(vectors - query, axis=1)
        elif self.distance_metric == "dot":
            return -np.dot(vectors, query)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def train(self, vectors: np.ndarray, verbose: bool = True) -> None:
        """Train the index by clustering vectors to create centroids.

        Args:
            vectors: Training vectors (N, dim)
            verbose: Print training progress
        """
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Vector dimension {vectors.shape[1]} != index dimension {self.dim}")

        if verbose:
            print(f"Training IVF index with {len(vectors)} vectors...")

        # Normalize for cosine
        if self.distance_metric == "cosine":
            vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)

        # Run k-means clustering
        n_clusters = min(self.n_clusters, len(vectors))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(vectors)

        self.centroids = kmeans.cluster_centers_

        # Initialize cells
        for i in range(n_clusters):
            self.cells[i] = IVFCell(
                centroid=self.centroids[i],
                vector_ids=[],
                vectors=[]
            )

        self.is_trained = True

        if verbose:
            print(f"Training complete. Created {n_clusters} clusters.")

    def _find_nearest_cluster(self, vector: np.ndarray) -> int:
        """Find the nearest cluster for a vector.

        Args:
            vector: Query vector

        Returns:
            Cluster ID
        """
        distances = self._distance_batch(vector, self.centroids)
        return int(np.argmin(distances))

    def _find_nearest_clusters(self, vector: np.ndarray, n: int) -> List[int]:
        """Find the n nearest clusters for a vector.

        Args:
            vector: Query vector
            n: Number of clusters

        Returns:
            List of cluster IDs
        """
        distances = self._distance_batch(vector, self.centroids)
        return np.argsort(distances)[:n].tolist()

    def add(
        self,
        vector: np.ndarray,
        metadata: Optional[Dict] = None,
        vector_id: Optional[int] = None
    ) -> int:
        """Add vector to index.

        Args:
            vector: Vector to add
            metadata: Optional metadata
            vector_id: Optional custom vector ID

        Returns:
            Vector ID
        """
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")

        if vector.shape[0] != self.dim:
            raise ValueError(f"Vector dimension {vector.shape[0]} != index dimension {self.dim}")

        # Normalize for cosine
        if self.distance_metric == "cosine":
            vector = vector / (np.linalg.norm(vector) + 1e-8)

        # Assign ID
        if vector_id is None:
            vector_id = self.next_id
            self.next_id += 1

        # Find nearest cluster
        cluster_id = self._find_nearest_cluster(vector)

        # Add to cell
        self.cells[cluster_id].vector_ids.append(vector_id)
        self.cells[cluster_id].vectors.append(vector)

        # Store metadata
        if metadata:
            self.metadata_store[vector_id] = metadata

        return vector_id

    def add_batch(
        self,
        vectors: np.ndarray,
        metadata_list: Optional[List[Dict]] = None
    ) -> List[int]:
        """Add multiple vectors efficiently.

        Args:
            vectors: Vectors to add (N, dim)
            metadata_list: Optional list of metadata dicts

        Returns:
            List of vector IDs
        """
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")

        # Normalize for cosine
        if self.distance_metric == "cosine":
            vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)

        # Find clusters for all vectors
        vector_ids = []
        for i, vector in enumerate(vectors):
            cluster_id = self._find_nearest_cluster(vector)

            vector_id = self.next_id
            self.next_id += 1

            self.cells[cluster_id].vector_ids.append(vector_id)
            self.cells[cluster_id].vectors.append(vector)

            if metadata_list and i < len(metadata_list):
                self.metadata_store[vector_id] = metadata_list[i]

            vector_ids.append(vector_id)

        return vector_ids

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        n_probe: Optional[int] = None
    ) -> List[Tuple[int, float, Optional[Dict]]]:
        """Search for k nearest neighbors.

        Args:
            query: Query vector
            k: Number of nearest neighbors to return
            n_probe: Number of clusters to search (default: self.n_probe)

        Returns:
            List of (vector_id, distance, metadata) tuples
        """
        if not self.is_trained:
            raise RuntimeError("Index must be trained before searching")

        n_probe = n_probe or self.n_probe

        # Normalize query for cosine
        if self.distance_metric == "cosine":
            query = query / (np.linalg.norm(query) + 1e-8)

        # Find nearest clusters to search
        cluster_ids = self._find_nearest_clusters(query, n_probe)

        # Collect candidates from clusters
        candidates = []
        for cluster_id in cluster_ids:
            cell = self.cells[cluster_id]

            if not cell.vectors:
                continue

            # Compute distances to all vectors in cell
            vectors_array = np.array(cell.vectors)
            distances = self._distance_batch(query, vectors_array)

            for i, (vector_id, dist) in enumerate(zip(cell.vector_ids, distances)):
                metadata = self.metadata_store.get(vector_id)
                candidates.append((vector_id, float(dist), metadata))

        # Sort and return top k
        candidates.sort(key=lambda x: x[1])
        return candidates[:k]

    def delete(self, vector_id: int) -> bool:
        """Delete vector from index.

        Args:
            vector_id: Vector ID to delete

        Returns:
            True if deleted, False if not found
        """
        # Find and remove from cell
        for cell in self.cells.values():
            if vector_id in cell.vector_ids:
                idx = cell.vector_ids.index(vector_id)
                cell.vector_ids.pop(idx)
                cell.vectors.pop(idx)

                # Remove metadata
                self.metadata_store.pop(vector_id, None)
                return True

        return False

    def save(self, path: str) -> None:
        """Save index to disk.

        Args:
            path: Path to save index
        """
        # Convert vectors lists to arrays for efficient storage
        cells_serializable = {}
        for cid, cell in self.cells.items():
            cells_serializable[cid] = {
                'centroid': cell.centroid,
                'vector_ids': cell.vector_ids,
                'vectors': cell.vectors
            }

        data = {
            'dim': self.dim,
            'n_clusters': self.n_clusters,
            'n_probe': self.n_probe,
            'distance_metric': self.distance_metric,
            'use_pq': self.use_pq,
            'pq_subvectors': self.pq_subvectors,
            'centroids': self.centroids,
            'cells': cells_serializable,
            'metadata_store': self.metadata_store,
            'next_id': self.next_id,
            'is_trained': self.is_trained
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        """Load index from disk.

        Args:
            path: Path to load index from
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.dim = data['dim']
        self.n_clusters = data['n_clusters']
        self.n_probe = data['n_probe']
        self.distance_metric = data['distance_metric']
        self.use_pq = data['use_pq']
        self.pq_subvectors = data['pq_subvectors']
        self.centroids = data['centroids']
        self.metadata_store = data['metadata_store']
        self.next_id = data['next_id']
        self.is_trained = data['is_trained']

        # Reconstruct cells
        self.cells = {}
        for cid, cell_data in data['cells'].items():
            self.cells[cid] = IVFCell(
                centroid=cell_data['centroid'],
                vector_ids=cell_data['vector_ids'],
                vectors=cell_data['vectors']
            )

    def __len__(self) -> int:
        """Return number of vectors in index."""
        return sum(len(cell.vector_ids) for cell in self.cells.values())

    def get_stats(self) -> Dict:
        """Get index statistics.

        Returns:
            Dictionary with index stats
        """
        cell_sizes = [len(cell.vector_ids) for cell in self.cells.values()]

        return {
            'total_vectors': len(self),
            'n_clusters': len(self.cells),
            'avg_cell_size': np.mean(cell_sizes) if cell_sizes else 0,
            'max_cell_size': max(cell_sizes) if cell_sizes else 0,
            'min_cell_size': min(cell_sizes) if cell_sizes else 0,
            'empty_cells': sum(1 for size in cell_sizes if size == 0)
        }