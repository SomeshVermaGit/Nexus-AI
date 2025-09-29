"""Hierarchical Navigable Small World (HNSW) index implementation.

HNSW is a graph-based algorithm for approximate nearest neighbor search.
It builds a multi-layer graph where higher layers provide long-range connections
and lower layers provide fine-grained search.
"""

import numpy as np
import heapq
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass, field
import pickle


@dataclass
class HNSWNode:
    """Node in the HNSW graph."""
    id: int
    vector: np.ndarray
    neighbors: Dict[int, Set[int]] = field(default_factory=dict)  # layer -> neighbor_ids
    metadata: Optional[Dict] = None


class HNSWIndex:
    """HNSW index for fast approximate nearest neighbor search.

    Based on the paper "Efficient and robust approximate nearest neighbor search using
    Hierarchical Navigable Small World graphs" by Malkov and Yashunin (2018).
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        max_M: int = None,
        max_M0: int = None,
        ml: float = 1.0,
        distance_metric: str = "cosine"
    ):
        """Initialize HNSW index.

        Args:
            dim: Vector dimension
            M: Number of bi-directional links for each node (except layer 0)
            ef_construction: Size of dynamic candidate list during construction
            ef_search: Size of dynamic candidate list during search
            max_M: Maximum number of connections for layers > 0 (default: M)
            max_M0: Maximum number of connections for layer 0 (default: 2*M)
            ml: Normalization factor for level generation
            distance_metric: Distance metric ('cosine', 'euclidean', 'dot')
        """
        self.dim = dim
        self.M = M
        self.max_M = max_M or M
        self.max_M0 = max_M0 or (2 * M)
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = ml
        self.distance_metric = distance_metric

        # Graph storage
        self.nodes: Dict[int, HNSWNode] = {}
        self.entry_point: Optional[int] = None
        self.max_layer: int = 0
        self.next_id: int = 0

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute distance between two vectors."""
        if self.distance_metric == "cosine":
            # Cosine distance: 1 - cosine_similarity
            return 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        elif self.distance_metric == "euclidean":
            return np.linalg.norm(a - b)
        elif self.distance_metric == "dot":
            return -np.dot(a, b)  # Negative for max-heap
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def _random_level(self) -> int:
        """Randomly determine layer for new element."""
        level = 0
        while np.random.random() < 0.5 and level < 16:
            level += 1
        return level

    def add(
        self,
        vector: np.ndarray,
        metadata: Optional[Dict] = None,
        node_id: Optional[int] = None
    ) -> int:
        """Add vector to index.

        Args:
            vector: Vector to add
            metadata: Optional metadata
            node_id: Optional custom node ID

        Returns:
            Node ID
        """
        if vector.shape[0] != self.dim:
            raise ValueError(f"Vector dimension {vector.shape[0]} != index dimension {self.dim}")

        # Normalize for cosine similarity
        if self.distance_metric == "cosine":
            vector = vector / np.linalg.norm(vector)

        # Assign ID and layer
        if node_id is None:
            node_id = self.next_id
            self.next_id += 1

        node_layer = self._random_level()

        # Create node
        node = HNSWNode(
            id=node_id,
            vector=vector,
            neighbors={layer: set() for layer in range(node_layer + 1)},
            metadata=metadata
        )
        self.nodes[node_id] = node

        # First element - set as entry point
        if self.entry_point is None:
            self.entry_point = node_id
            self.max_layer = node_layer
            return node_id

        # Search for nearest neighbors
        ep = [self.entry_point]

        # Greedy search through layers > node_layer
        for layer in range(self.max_layer, node_layer, -1):
            ep = self._search_layer(vector, ep, 1, layer)

        # Insert into layers <= node_layer
        for layer in range(node_layer, -1, -1):
            candidates = self._search_layer(vector, ep, self.ef_construction, layer)

            # Select M neighbors
            M = self.M if layer > 0 else self.max_M0
            neighbors = self._select_neighbors(vector, candidates, M, layer)

            # Add bidirectional links
            for neighbor_id in neighbors:
                node.neighbors[layer].add(neighbor_id)
                self.nodes[neighbor_id].neighbors[layer].add(node_id)

                # Prune neighbor's connections if needed
                max_conn = self.max_M if layer > 0 else self.max_M0
                if len(self.nodes[neighbor_id].neighbors[layer]) > max_conn:
                    neighbor_vector = self.nodes[neighbor_id].vector
                    neighbor_conns = list(self.nodes[neighbor_id].neighbors[layer])
                    neighbor_dists = [(nid, self._distance(neighbor_vector, self.nodes[nid].vector))
                                    for nid in neighbor_conns]
                    neighbor_dists.sort(key=lambda x: x[1])

                    # Keep only M closest
                    new_conns = {nid for nid, _ in neighbor_dists[:max_conn]}

                    # Remove pruned connections
                    for pruned_id in self.nodes[neighbor_id].neighbors[layer] - new_conns:
                        self.nodes[pruned_id].neighbors[layer].discard(neighbor_id)

                    self.nodes[neighbor_id].neighbors[layer] = new_conns

            ep = candidates

        # Update entry point if needed
        if node_layer > self.max_layer:
            self.max_layer = node_layer
            self.entry_point = node_id

        return node_id

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: List[int],
        num_closest: int,
        layer: int
    ) -> List[int]:
        """Search for nearest neighbors in a specific layer.

        Args:
            query: Query vector
            entry_points: Starting points for search
            num_closest: Number of closest elements to return
            layer: Layer to search

        Returns:
            List of closest node IDs
        """
        visited = set()
        candidates = []
        w = []

        # Initialize with entry points
        for ep_id in entry_points:
            dist = self._distance(query, self.nodes[ep_id].vector)
            heapq.heappush(candidates, (dist, ep_id))
            heapq.heappush(w, (-dist, ep_id))
            visited.add(ep_id)

        while candidates:
            current_dist, current_id = heapq.heappop(candidates)

            # If current is farther than worst in w, stop
            if current_dist > -w[0][0]:
                break

            # Check neighbors
            if layer in self.nodes[current_id].neighbors:
                for neighbor_id in self.nodes[current_id].neighbors[layer]:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        dist = self._distance(query, self.nodes[neighbor_id].vector)

                        if dist < -w[0][0] or len(w) < num_closest:
                            heapq.heappush(candidates, (dist, neighbor_id))
                            heapq.heappush(w, (-dist, neighbor_id))

                            # Keep only num_closest in w
                            if len(w) > num_closest:
                                heapq.heappop(w)

        return [node_id for _, node_id in w]

    def _select_neighbors(
        self,
        query: np.ndarray,
        candidates: List[int],
        M: int,
        layer: int
    ) -> Set[int]:
        """Select M neighbors using heuristic.

        Args:
            query: Query vector
            candidates: Candidate node IDs
            M: Number of neighbors to select
            layer: Current layer

        Returns:
            Set of selected neighbor IDs
        """
        # Simple heuristic: select M closest
        if len(candidates) <= M:
            return set(candidates)

        candidate_dists = [(nid, self._distance(query, self.nodes[nid].vector))
                          for nid in candidates]
        candidate_dists.sort(key=lambda x: x[1])

        return {nid for nid, _ in candidate_dists[:M]}

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef: Optional[int] = None
    ) -> List[Tuple[int, float, Optional[Dict]]]:
        """Search for k nearest neighbors.

        Args:
            query: Query vector
            k: Number of nearest neighbors to return
            ef: Size of dynamic candidate list (default: self.ef_search)

        Returns:
            List of (node_id, distance, metadata) tuples
        """
        if len(self.nodes) == 0:
            return []

        ef = ef or self.ef_search

        # Normalize query for cosine
        if self.distance_metric == "cosine":
            query = query / np.linalg.norm(query)

        # Start from entry point, search from top layer
        ep = [self.entry_point]

        for layer in range(self.max_layer, 0, -1):
            ep = self._search_layer(query, ep, 1, layer)

        # Search layer 0 with ef
        ep = self._search_layer(query, ep, max(ef, k), 0)

        # Get top k
        results = [(nid, self._distance(query, self.nodes[nid].vector), self.nodes[nid].metadata)
                  for nid in ep]
        results.sort(key=lambda x: x[1])

        return results[:k]

    def delete(self, node_id: int) -> bool:
        """Delete node from index.

        Args:
            node_id: Node ID to delete

        Returns:
            True if deleted, False if not found
        """
        if node_id not in self.nodes:
            return False

        node = self.nodes[node_id]

        # Remove all connections
        for layer in node.neighbors:
            for neighbor_id in node.neighbors[layer]:
                if neighbor_id in self.nodes:
                    self.nodes[neighbor_id].neighbors[layer].discard(node_id)

        # Remove node
        del self.nodes[node_id]

        # Update entry point if needed
        if node_id == self.entry_point:
            self.entry_point = next(iter(self.nodes.keys())) if self.nodes else None

        return True

    def save(self, path: str) -> None:
        """Save index to disk.

        Args:
            path: Path to save index
        """
        data = {
            'dim': self.dim,
            'M': self.M,
            'max_M': self.max_M,
            'max_M0': self.max_M0,
            'ef_construction': self.ef_construction,
            'ef_search': self.ef_search,
            'ml': self.ml,
            'distance_metric': self.distance_metric,
            'nodes': self.nodes,
            'entry_point': self.entry_point,
            'max_layer': self.max_layer,
            'next_id': self.next_id
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
        self.M = data['M']
        self.max_M = data['max_M']
        self.max_M0 = data['max_M0']
        self.ef_construction = data['ef_construction']
        self.ef_search = data['ef_search']
        self.ml = data['ml']
        self.distance_metric = data['distance_metric']
        self.nodes = data['nodes']
        self.entry_point = data['entry_point']
        self.max_layer = data['max_layer']
        self.next_id = data['next_id']

    def __len__(self) -> int:
        """Return number of nodes in index."""
        return len(self.nodes)