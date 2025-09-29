"""Distributed storage coordination for multi-node deployment."""

import asyncio
import aiohttp
from typing import List, Dict, Optional, Any
import numpy as np
from dataclasses import dataclass
import json


@dataclass
class Node:
    """Represents a storage node in the distributed system."""
    node_id: str
    host: str
    port: int
    is_healthy: bool = True
    shard_ids: List[int] = None

    def __post_init__(self):
        if self.shard_ids is None:
            self.shard_ids = []

    @property
    def url(self) -> str:
        """Get base URL for the node."""
        return f"http://{self.host}:{self.port}"


class DistributedStore:
    """Distributed storage coordinator for multi-node vector database.

    Coordinates search and write operations across multiple nodes,
    handles node health monitoring, and provides fault tolerance.
    """

    def __init__(
        self,
        nodes: List[Node],
        replication_factor: int = 2,
        consistency_level: str = "quorum"
    ):
        """Initialize distributed store.

        Args:
            nodes: List of storage nodes
            replication_factor: Number of replicas for each document
            consistency_level: Consistency level ('one', 'quorum', 'all')
        """
        self.nodes = {node.node_id: node for node in nodes}
        self.replication_factor = replication_factor
        self.consistency_level = consistency_level

        # Health check interval
        self.health_check_interval = 30  # seconds

    async def _make_request(
        self,
        node: Node,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Make HTTP request to a node.

        Args:
            node: Target node
            endpoint: API endpoint
            method: HTTP method
            data: Optional request data

        Returns:
            Response data or None on failure
        """
        url = f"{node.url}{endpoint}"

        try:
            async with aiohttp.ClientSession() as session:
                if method == "GET":
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            return await resp.json()
                elif method == "POST":
                    async with session.post(url, json=data, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status in (200, 201):
                            return await resp.json()
                elif method == "DELETE":
                    async with session.delete(url, json=data, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status in (200, 204):
                            return {"success": True}

        except Exception as e:
            print(f"Request to {node.node_id} failed: {e}")
            node.is_healthy = False
            return None

        return None

    def _get_replica_nodes(self, doc_id: int) -> List[Node]:
        """Get nodes that should store replicas of a document.

        Args:
            doc_id: Document ID

        Returns:
            List of nodes
        """
        # Simple consistent hashing
        healthy_nodes = [node for node in self.nodes.values() if node.is_healthy]

        if not healthy_nodes:
            raise RuntimeError("No healthy nodes available")

        # Hash document ID to select nodes
        node_list = sorted(healthy_nodes, key=lambda n: n.node_id)
        start_idx = doc_id % len(node_list)

        replica_nodes = []
        for i in range(self.replication_factor):
            idx = (start_idx + i) % len(node_list)
            replica_nodes.append(node_list[idx])

        return replica_nodes

    async def add(
        self,
        vector: np.ndarray,
        metadata: Optional[Dict] = None,
        doc_id: Optional[int] = None
    ) -> Optional[int]:
        """Add document to distributed store.

        Args:
            vector: Vector to add
            metadata: Optional metadata
            doc_id: Optional document ID

        Returns:
            Document ID or None on failure
        """
        # Determine target nodes
        if doc_id is None:
            doc_id = self._generate_doc_id()

        replica_nodes = self._get_replica_nodes(doc_id)

        # Prepare request data
        data = {
            'doc_id': doc_id,
            'vector': vector.tolist(),
            'metadata': metadata
        }

        # Send to replica nodes
        tasks = []
        for node in replica_nodes:
            task = self._make_request(node, "/api/add", method="POST", data=data)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Check consistency level
        successful = sum(1 for r in results if r is not None)

        if self.consistency_level == "one" and successful >= 1:
            return doc_id
        elif self.consistency_level == "quorum" and successful >= (len(replica_nodes) // 2 + 1):
            return doc_id
        elif self.consistency_level == "all" and successful == len(replica_nodes):
            return doc_id
        else:
            return None

    async def search(
        self,
        query: np.ndarray,
        k: int = 10
    ) -> List[Dict]:
        """Search across all nodes.

        Args:
            query: Query vector
            k: Number of results

        Returns:
            List of results
        """
        # Query all healthy nodes
        healthy_nodes = [node for node in self.nodes.values() if node.is_healthy]

        if not healthy_nodes:
            raise RuntimeError("No healthy nodes available")

        data = {
            'query': query.tolist(),
            'k': k
        }

        # Send search requests to all nodes
        tasks = []
        for node in healthy_nodes:
            task = self._make_request(node, "/api/search", method="POST", data=data)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Merge results from all nodes
        all_results = []
        for node_results in results:
            if node_results and 'results' in node_results:
                all_results.extend(node_results['results'])

        # Deduplicate and sort by distance
        seen = set()
        unique_results = []

        for result in all_results:
            doc_id = result['doc_id']
            if doc_id not in seen:
                seen.add(doc_id)
                unique_results.append(result)

        unique_results.sort(key=lambda x: x.get('distance', float('inf')))

        return unique_results[:k]

    async def delete(self, doc_id: int) -> bool:
        """Delete document from distributed store.

        Args:
            doc_id: Document ID

        Returns:
            True if deleted successfully
        """
        replica_nodes = self._get_replica_nodes(doc_id)

        data = {'doc_id': doc_id}

        # Send delete requests
        tasks = []
        for node in replica_nodes:
            task = self._make_request(node, "/api/delete", method="DELETE", data=data)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Check consistency
        successful = sum(1 for r in results if r is not None)

        if self.consistency_level == "one":
            return successful >= 1
        elif self.consistency_level == "quorum":
            return successful >= (len(replica_nodes) // 2 + 1)
        elif self.consistency_level == "all":
            return successful == len(replica_nodes)

        return False

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all nodes.

        Returns:
            Dictionary mapping node_id to health status
        """
        tasks = []
        node_ids = []

        for node in self.nodes.values():
            task = self._make_request(node, "/health", method="GET")
            tasks.append(task)
            node_ids.append(node.node_id)

        results = await asyncio.gather(*tasks)

        health_status = {}
        for node_id, result in zip(node_ids, results):
            is_healthy = result is not None
            self.nodes[node_id].is_healthy = is_healthy
            health_status[node_id] = is_healthy

        return health_status

    async def start_health_monitor(self):
        """Start background health monitoring."""
        while True:
            await self.health_check()
            await asyncio.sleep(self.health_check_interval)

    def _generate_doc_id(self) -> int:
        """Generate unique document ID."""
        # Simple approach: timestamp-based ID
        import time
        return int(time.time() * 1000000)

    def add_node(self, node: Node) -> None:
        """Add a new node to the cluster.

        Args:
            node: Node to add
        """
        self.nodes[node.node_id] = node

    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the cluster.

        Args:
            node_id: Node ID to remove

        Returns:
            True if removed
        """
        if node_id in self.nodes:
            del self.nodes[node_id]
            return True
        return False

    def get_cluster_stats(self) -> Dict:
        """Get cluster statistics.

        Returns:
            Dictionary with cluster stats
        """
        total_nodes = len(self.nodes)
        healthy_nodes = sum(1 for node in self.nodes.values() if node.is_healthy)

        return {
            'total_nodes': total_nodes,
            'healthy_nodes': healthy_nodes,
            'unhealthy_nodes': total_nodes - healthy_nodes,
            'replication_factor': self.replication_factor,
            'consistency_level': self.consistency_level,
            'nodes': [
                {
                    'node_id': node.node_id,
                    'host': node.host,
                    'port': node.port,
                    'is_healthy': node.is_healthy,
                    'num_shards': len(node.shard_ids)
                }
                for node in self.nodes.values()
            ]
        }

    async def rebalance(self) -> None:
        """Rebalance data across nodes.

        Useful after adding/removing nodes.
        """
        # This would implement data migration logic
        # For now, just a placeholder
        print("Rebalancing cluster...")

        # 1. Collect all document IDs from all nodes
        # 2. Redistribute based on new node configuration
        # 3. Move documents between nodes as needed
        # 4. Verify replication factor is maintained

        pass