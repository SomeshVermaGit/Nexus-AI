"""Product Quantization (PQ) for vector compression.

PQ divides vectors into subvectors and quantizes each independently,
achieving high compression rates while maintaining search quality.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import pickle
from sklearn.cluster import KMeans


@dataclass
class PQCodebook:
    """Codebook for a subvector."""
    centroids: np.ndarray  # (n_centroids, subdim)
    subdim: int


class ProductQuantizer:
    """Product Quantization for vector compression.

    Divides vectors into M subvectors and quantizes each independently using
    k-means clustering. This achieves compression while maintaining approximate
    distances.
    """

    def __init__(
        self,
        dim: int,
        M: int = 8,
        n_bits: int = 8,
        distance_metric: str = "euclidean"
    ):
        """Initialize Product Quantizer.

        Args:
            dim: Vector dimension
            M: Number of subvectors (must divide dim)
            n_bits: Bits per subvector (determines codebook size: 2^n_bits)
            distance_metric: Distance metric ('euclidean', 'cosine')
        """
        if dim % M != 0:
            raise ValueError(f"Dimension {dim} must be divisible by M={M}")

        self.dim = dim
        self.M = M
        self.subdim = dim // M
        self.n_bits = n_bits
        self.n_centroids = 2 ** n_bits
        self.distance_metric = distance_metric

        # Codebooks for each subvector
        self.codebooks: List[PQCodebook] = []
        self.is_trained: bool = False

    def train(self, vectors: np.ndarray, verbose: bool = True) -> None:
        """Train the product quantizer by clustering subvectors.

        Args:
            vectors: Training vectors (N, dim)
            verbose: Print training progress
        """
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Vector dimension {vectors.shape[1]} != {self.dim}")

        if verbose:
            print(f"Training PQ with {len(vectors)} vectors, M={self.M}, {self.n_bits} bits...")

        # Train codebook for each subvector
        self.codebooks = []

        for m in range(self.M):
            start_idx = m * self.subdim
            end_idx = (m + 1) * self.subdim

            # Extract subvectors
            subvectors = vectors[:, start_idx:end_idx]

            # Run k-means
            n_clusters = min(self.n_centroids, len(subvectors))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42 + m, n_init=10)
            kmeans.fit(subvectors)

            # Store codebook
            codebook = PQCodebook(
                centroids=kmeans.cluster_centers_,
                subdim=self.subdim
            )
            self.codebooks.append(codebook)

            if verbose and (m + 1) % max(1, self.M // 4) == 0:
                print(f"  Trained codebook {m + 1}/{self.M}")

        self.is_trained = True

        if verbose:
            compression_ratio = (self.dim * 32) / (self.M * self.n_bits)
            print(f"Training complete. Compression ratio: {compression_ratio:.1f}x")

    def encode(self, vector: np.ndarray) -> np.ndarray:
        """Encode a vector to PQ codes.

        Args:
            vector: Vector to encode (dim,)

        Returns:
            PQ codes (M,) - indices into codebooks
        """
        if not self.is_trained:
            raise RuntimeError("PQ must be trained before encoding")

        if vector.shape[0] != self.dim:
            raise ValueError(f"Vector dimension {vector.shape[0]} != {self.dim}")

        codes = np.zeros(self.M, dtype=np.uint8 if self.n_bits <= 8 else np.uint16)

        for m in range(self.M):
            start_idx = m * self.subdim
            end_idx = (m + 1) * self.subdim

            subvector = vector[start_idx:end_idx]
            centroids = self.codebooks[m].centroids

            # Find nearest centroid
            distances = np.linalg.norm(centroids - subvector, axis=1)
            codes[m] = np.argmin(distances)

        return codes

    def encode_batch(self, vectors: np.ndarray) -> np.ndarray:
        """Encode multiple vectors efficiently.

        Args:
            vectors: Vectors to encode (N, dim)

        Returns:
            PQ codes (N, M)
        """
        if not self.is_trained:
            raise RuntimeError("PQ must be trained before encoding")

        N = len(vectors)
        codes = np.zeros((N, self.M), dtype=np.uint8 if self.n_bits <= 8 else np.uint16)

        for m in range(self.M):
            start_idx = m * self.subdim
            end_idx = (m + 1) * self.subdim

            subvectors = vectors[:, start_idx:end_idx]
            centroids = self.codebooks[m].centroids

            # Compute distances to all centroids
            # (N, subdim) - (k, subdim) -> (N, k)
            distances = np.linalg.norm(
                subvectors[:, np.newaxis, :] - centroids[np.newaxis, :, :],
                axis=2
            )

            codes[:, m] = np.argmin(distances, axis=1)

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode PQ codes back to approximate vector.

        Args:
            codes: PQ codes (M,)

        Returns:
            Reconstructed vector (dim,)
        """
        if not self.is_trained:
            raise RuntimeError("PQ must be trained before decoding")

        vector = np.zeros(self.dim, dtype=np.float32)

        for m in range(self.M):
            start_idx = m * self.subdim
            end_idx = (m + 1) * self.subdim

            code = codes[m]
            centroid = self.codebooks[m].centroids[code]
            vector[start_idx:end_idx] = centroid

        return vector

    def decode_batch(self, codes: np.ndarray) -> np.ndarray:
        """Decode multiple PQ codes efficiently.

        Args:
            codes: PQ codes (N, M)

        Returns:
            Reconstructed vectors (N, dim)
        """
        if not self.is_trained:
            raise RuntimeError("PQ must be trained before decoding")

        N = len(codes)
        vectors = np.zeros((N, self.dim), dtype=np.float32)

        for m in range(self.M):
            start_idx = m * self.subdim
            end_idx = (m + 1) * self.subdim

            # Lookup centroids for all codes
            centroids = self.codebooks[m].centroids[codes[:, m]]
            vectors[:, start_idx:end_idx] = centroids

        return vectors

    def compute_distance_table(self, query: np.ndarray) -> np.ndarray:
        """Precompute distance table for fast asymmetric search.

        Args:
            query: Query vector (dim,)

        Returns:
            Distance table (M, n_centroids) - distances from query subvectors to all centroids
        """
        if not self.is_trained:
            raise RuntimeError("PQ must be trained")

        distance_table = np.zeros((self.M, self.n_centroids), dtype=np.float32)

        for m in range(self.M):
            start_idx = m * self.subdim
            end_idx = (m + 1) * self.subdim

            query_sub = query[start_idx:end_idx]
            centroids = self.codebooks[m].centroids

            # Compute distances
            distances = np.linalg.norm(centroids - query_sub, axis=1)
            distance_table[m, :len(distances)] = distances

        return distance_table

    def asymmetric_distance(
        self,
        query: np.ndarray,
        codes: np.ndarray,
        distance_table: Optional[np.ndarray] = None
    ) -> float:
        """Compute asymmetric distance between query and PQ-encoded vector.

        Args:
            query: Query vector (dim,)
            codes: PQ codes (M,)
            distance_table: Precomputed distance table (optional)

        Returns:
            Approximate distance
        """
        if distance_table is None:
            distance_table = self.compute_distance_table(query)

        # Sum distances across subvectors
        distance = 0.0
        for m in range(self.M):
            code = codes[m]
            distance += distance_table[m, code] ** 2

        return np.sqrt(distance)

    def asymmetric_distance_batch(
        self,
        query: np.ndarray,
        codes: np.ndarray
    ) -> np.ndarray:
        """Compute asymmetric distances for multiple codes efficiently.

        Args:
            query: Query vector (dim,)
            codes: PQ codes (N, M)

        Returns:
            Distances (N,)
        """
        distance_table = self.compute_distance_table(query)

        # Lookup distances and sum
        distances = np.zeros(len(codes), dtype=np.float32)

        for m in range(self.M):
            distances += distance_table[m, codes[:, m]] ** 2

        return np.sqrt(distances)

    def save(self, path: str) -> None:
        """Save PQ to disk.

        Args:
            path: Path to save
        """
        data = {
            'dim': self.dim,
            'M': self.M,
            'subdim': self.subdim,
            'n_bits': self.n_bits,
            'n_centroids': self.n_centroids,
            'distance_metric': self.distance_metric,
            'codebooks': self.codebooks,
            'is_trained': self.is_trained
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        """Load PQ from disk.

        Args:
            path: Path to load from
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.dim = data['dim']
        self.M = data['M']
        self.subdim = data['subdim']
        self.n_bits = data['n_bits']
        self.n_centroids = data['n_centroids']
        self.distance_metric = data['distance_metric']
        self.codebooks = data['codebooks']
        self.is_trained = data['is_trained']

    def estimate_memory(self, n_vectors: int) -> Dict[str, float]:
        """Estimate memory usage.

        Args:
            n_vectors: Number of vectors to store

        Returns:
            Dictionary with memory estimates in MB
        """
        # Original vectors
        original_mb = (n_vectors * self.dim * 4) / (1024 ** 2)

        # PQ codes
        bytes_per_code = 1 if self.n_bits <= 8 else 2
        compressed_mb = (n_vectors * self.M * bytes_per_code) / (1024 ** 2)

        # Codebooks
        codebook_mb = (self.M * self.n_centroids * self.subdim * 4) / (1024 ** 2)

        return {
            'original_mb': original_mb,
            'compressed_mb': compressed_mb,
            'codebook_mb': codebook_mb,
            'total_mb': compressed_mb + codebook_mb,
            'compression_ratio': original_mb / (compressed_mb + codebook_mb)
        }