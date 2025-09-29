"""Sharding logic for distributing vectors across multiple partitions."""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
import hashlib
import pickle
from pathlib import Path


@dataclass
class Shard:
    """A shard containing a portion of the vector index."""
    shard_id: int
    index: Any  # Vector index (HNSW, IVF, etc.)
    metadata: Dict[int, Dict] = field(default_factory=dict)
    doc_count: int = 0

    def add(self, vector: np.ndarray, metadata: Optional[Dict] = None, doc_id: Optional[int] = None) -> int:
        """Add vector to shard."""
        doc_id = self.index.add(vector, metadata=metadata, node_id=doc_id)
        if metadata:
            self.metadata[doc_id] = metadata
        self.doc_count += 1
        return doc_id

    def search(self, query: np.ndarray, k: int) -> List[Tuple[int, float, Optional[Dict]]]:
        """Search within shard."""
        return self.index.search(query, k=k)

    def delete(self, doc_id: int) -> bool:
        """Delete vector from shard."""
        success = self.index.delete(doc_id)
        if success and doc_id in self.metadata:
            del self.metadata[doc_id]
            self.doc_count -= 1
        return success


class ShardManager:
    """Manager for multiple shards with routing and coordination.

    Supports different sharding strategies:
    - hash: Consistent hashing based on doc ID
    - range: Range-based partitioning
    - random: Random assignment
    - round_robin: Distribute evenly across shards
    """

    def __init__(
        self,
        n_shards: int,
        index_factory,
        sharding_strategy: str = "hash",
        replication_factor: int = 1
    ):
        """Initialize shard manager.

        Args:
            n_shards: Number of shards
            index_factory: Factory function to create index for each shard
            sharding_strategy: Strategy for routing ('hash', 'range', 'random', 'round_robin')
            replication_factor: Number of replicas for each document
        """
        self.n_shards = n_shards
        self.index_factory = index_factory
        self.sharding_strategy = sharding_strategy
        self.replication_factor = replication_factor

        # Initialize shards
        self.shards: List[Shard] = []
        for i in range(n_shards):
            index = index_factory()
            shard = Shard(shard_id=i, index=index)
            self.shards.append(shard)

        # Round-robin counter
        self._round_robin_counter = 0

        # Range boundaries (for range sharding)
        self.range_boundaries: List[int] = []

    def _hash_shard_id(self, doc_id: int) -> int:
        """Compute shard ID using consistent hashing.

        Args:
            doc_id: Document ID

        Returns:
            Shard ID
        """
        # Use hash function for consistent mapping
        hash_value = int(hashlib.md5(str(doc_id).encode()).hexdigest(), 16)
        return hash_value % self.n_shards

    def _get_shard_ids(self, doc_id: int) -> List[int]:
        """Get shard IDs for a document (includes replicas).

        Args:
            doc_id: Document ID

        Returns:
            List of shard IDs
        """
        if self.sharding_strategy == "hash":
            primary_shard = self._hash_shard_id(doc_id)
        elif self.sharding_strategy == "random":
            primary_shard = np.random.randint(0, self.n_shards)
        elif self.sharding_strategy == "round_robin":
            primary_shard = self._round_robin_counter % self.n_shards
            self._round_robin_counter += 1
        elif self.sharding_strategy == "range":
            # Find appropriate range
            primary_shard = 0
            for i, boundary in enumerate(self.range_boundaries):
                if doc_id < boundary:
                    primary_shard = i
                    break
        else:
            raise ValueError(f"Unknown sharding strategy: {self.sharding_strategy}")

        # Add replicas
        shard_ids = [primary_shard]
        for i in range(1, self.replication_factor):
            replica_shard = (primary_shard + i) % self.n_shards
            shard_ids.append(replica_shard)

        return shard_ids

    def add(
        self,
        vector: np.ndarray,
        metadata: Optional[Dict] = None,
        doc_id: Optional[int] = None
    ) -> int:
        """Add vector to appropriate shard(s).

        Args:
            vector: Vector to add
            metadata: Optional metadata
            doc_id: Optional document ID

        Returns:
            Document ID
        """
        # Generate doc ID if not provided
        if doc_id is None:
            doc_id = self._generate_doc_id()

        # Get target shards
        shard_ids = self._get_shard_ids(doc_id)

        # Add to each shard (primary + replicas)
        for shard_id in shard_ids:
            self.shards[shard_id].add(vector, metadata=metadata, doc_id=doc_id)

        return doc_id

    def _generate_doc_id(self) -> int:
        """Generate unique document ID."""
        # Simple approach: use max existing ID + 1
        max_id = 0
        for shard in self.shards:
            if hasattr(shard.index, 'next_id'):
                max_id = max(max_id, shard.index.next_id)
            elif hasattr(shard.index, 'nodes'):
                if shard.index.nodes:
                    max_id = max(max_id, max(shard.index.nodes.keys()))

        return max_id + 1

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        search_all_shards: bool = True
    ) -> List[Tuple[int, float, Optional[Dict]]]:
        """Search across shards.

        Args:
            query: Query vector
            k: Number of results
            search_all_shards: Whether to search all shards or just one

        Returns:
            List of (doc_id, distance, metadata) tuples
        """
        if search_all_shards:
            # Search all shards and merge results
            all_results = []

            for shard in self.shards:
                shard_results = shard.search(query, k=k)
                all_results.extend(shard_results)

            # Deduplicate (in case of replication)
            seen = set()
            unique_results = []
            for doc_id, dist, metadata in all_results:
                if doc_id not in seen:
                    seen.add(doc_id)
                    unique_results.append((doc_id, dist, metadata))

            # Sort by distance and return top k
            unique_results.sort(key=lambda x: x[1])
            return unique_results[:k]
        else:
            # Search random shard (for load balancing)
            shard_id = np.random.randint(0, self.n_shards)
            return self.shards[shard_id].search(query, k=k)

    def delete(self, doc_id: int) -> bool:
        """Delete document from all shards.

        Args:
            doc_id: Document ID

        Returns:
            True if deleted from at least one shard
        """
        shard_ids = self._get_shard_ids(doc_id)
        deleted = False

        for shard_id in shard_ids:
            if self.shards[shard_id].delete(doc_id):
                deleted = True

        return deleted

    def get_stats(self) -> Dict:
        """Get statistics for all shards.

        Returns:
            Dictionary with shard statistics
        """
        stats = {
            'n_shards': self.n_shards,
            'replication_factor': self.replication_factor,
            'sharding_strategy': self.sharding_strategy,
            'shards': []
        }

        total_docs = 0
        for shard in self.shards:
            shard_stats = {
                'shard_id': shard.shard_id,
                'doc_count': shard.doc_count,
                'index_size': len(shard.index) if hasattr(shard.index, '__len__') else 0
            }
            stats['shards'].append(shard_stats)
            total_docs += shard.doc_count

        stats['total_docs'] = total_docs
        stats['avg_docs_per_shard'] = total_docs / self.n_shards if self.n_shards > 0 else 0

        return stats

    def rebalance(self) -> None:
        """Rebalance data across shards.

        Useful after adding/removing shards or when shards are unbalanced.
        """
        # Collect all documents from all shards
        all_docs = []

        for shard in self.shards:
            if hasattr(shard.index, 'nodes'):
                # HNSW index
                for node_id, node in shard.index.nodes.items():
                    all_docs.append((node_id, node.vector, node.metadata))
            elif hasattr(shard.index, 'cells'):
                # IVF index
                for cell in shard.index.cells.values():
                    for vid, vec in zip(cell.vector_ids, cell.vectors):
                        metadata = shard.index.metadata_store.get(vid)
                        all_docs.append((vid, vec, metadata))

        # Clear shards
        for i in range(self.n_shards):
            self.shards[i] = Shard(shard_id=i, index=self.index_factory())

        # Re-add documents
        for doc_id, vector, metadata in all_docs:
            self.add(vector, metadata=metadata, doc_id=doc_id)

    def save(self, base_path: str) -> None:
        """Save all shards to disk.

        Args:
            base_path: Base directory path
        """
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            'n_shards': self.n_shards,
            'sharding_strategy': self.sharding_strategy,
            'replication_factor': self.replication_factor,
            'round_robin_counter': self._round_robin_counter,
            'range_boundaries': self.range_boundaries
        }

        with open(base_path / 'shard_manager.pkl', 'wb') as f:
            pickle.dump(metadata, f)

        # Save each shard
        for shard in self.shards:
            shard_path = base_path / f'shard_{shard.shard_id}.pkl'
            shard.index.save(str(shard_path))

            # Save shard metadata
            shard_meta_path = base_path / f'shard_{shard.shard_id}_meta.pkl'
            with open(shard_meta_path, 'wb') as f:
                pickle.dump({
                    'metadata': shard.metadata,
                    'doc_count': shard.doc_count
                }, f)

    def load(self, base_path: str) -> None:
        """Load all shards from disk.

        Args:
            base_path: Base directory path
        """
        base_path = Path(base_path)

        # Load metadata
        with open(base_path / 'shard_manager.pkl', 'rb') as f:
            metadata = pickle.load(f)

        self.n_shards = metadata['n_shards']
        self.sharding_strategy = metadata['sharding_strategy']
        self.replication_factor = metadata['replication_factor']
        self._round_robin_counter = metadata['round_robin_counter']
        self.range_boundaries = metadata['range_boundaries']

        # Load each shard
        self.shards = []
        for i in range(self.n_shards):
            # Create index and load
            index = self.index_factory()
            shard_path = base_path / f'shard_{i}.pkl'
            index.load(str(shard_path))

            # Load shard metadata
            shard_meta_path = base_path / f'shard_{i}_meta.pkl'
            with open(shard_meta_path, 'rb') as f:
                shard_meta = pickle.load(f)

            shard = Shard(
                shard_id=i,
                index=index,
                metadata=shard_meta['metadata'],
                doc_count=shard_meta['doc_count']
            )
            self.shards.append(shard)