"""Hybrid search combining dense vectors and sparse (BM25) retrieval."""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from collections import Counter
import math
import re


@dataclass
class Document:
    """Document with text and vector representation."""
    id: int
    text: str
    vector: np.ndarray
    metadata: Optional[Dict] = None


class BM25:
    """BM25 scoring for sparse text retrieval.

    Uses Okapi BM25 algorithm for ranking documents based on term frequency.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initialize BM25.

        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b

        # Index structures
        self.doc_freqs: Dict[str, int] = {}  # term -> num docs containing term
        self.doc_term_freqs: Dict[int, Dict[str, int]] = {}  # doc_id -> {term -> freq}
        self.doc_lengths: Dict[int, int] = {}  # doc_id -> length
        self.avg_doc_length: float = 0.0
        self.num_docs: int = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def add_document(self, doc_id: int, text: str) -> None:
        """Add document to BM25 index.

        Args:
            doc_id: Document ID
            text: Document text
        """
        tokens = self._tokenize(text)
        term_freqs = Counter(tokens)

        # Update doc term frequencies
        self.doc_term_freqs[doc_id] = dict(term_freqs)
        self.doc_lengths[doc_id] = len(tokens)

        # Update document frequencies
        for term in term_freqs.keys():
            self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        # Update statistics
        self.num_docs += 1
        self.avg_doc_length = sum(self.doc_lengths.values()) / self.num_docs

    def score(self, query: str, doc_id: int) -> float:
        """Compute BM25 score for a document given a query.

        Args:
            query: Query string
            doc_id: Document ID

        Returns:
            BM25 score
        """
        if doc_id not in self.doc_term_freqs:
            return 0.0

        query_terms = self._tokenize(query)
        doc_term_freqs = self.doc_term_freqs[doc_id]
        doc_length = self.doc_lengths[doc_id]

        score = 0.0
        for term in query_terms:
            if term not in doc_term_freqs:
                continue

            # Term frequency in document
            tf = doc_term_freqs[term]

            # Document frequency
            df = self.doc_freqs.get(term, 0)

            # IDF component
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)

            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Search for top-k documents.

        Args:
            query: Query string
            k: Number of results

        Returns:
            List of (doc_id, score) tuples
        """
        scores = []
        for doc_id in self.doc_term_freqs.keys():
            score = self.score(query, doc_id)
            if score > 0:
                scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class HybridSearch:
    """Hybrid search combining dense vector search and sparse BM25 retrieval.

    Supports multiple fusion strategies for combining rankings.
    """

    def __init__(
        self,
        vector_index,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        fusion_method: str = "rrf",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75
    ):
        """Initialize hybrid search.

        Args:
            vector_index: Dense vector index (HNSW, IVF, etc.)
            dense_weight: Weight for dense retrieval
            sparse_weight: Weight for sparse retrieval
            fusion_method: Fusion method ('rrf', 'linear', 'max')
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
        """
        self.vector_index = vector_index
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.fusion_method = fusion_method

        self.bm25 = BM25(k1=bm25_k1, b=bm25_b)
        self.documents: Dict[int, Document] = {}

    def add_document(
        self,
        text: str,
        vector: np.ndarray,
        metadata: Optional[Dict] = None,
        doc_id: Optional[int] = None
    ) -> int:
        """Add document to hybrid index.

        Args:
            text: Document text
            vector: Document vector embedding
            metadata: Optional metadata
            doc_id: Optional custom document ID

        Returns:
            Document ID
        """
        # Add to vector index
        doc_id = self.vector_index.add(vector, metadata=metadata, node_id=doc_id)

        # Add to BM25 index
        self.bm25.add_document(doc_id, text)

        # Store document
        doc = Document(
            id=doc_id,
            text=text,
            vector=vector,
            metadata=metadata
        )
        self.documents[doc_id] = doc

        return doc_id

    def _normalize_scores(self, results: List[Tuple[int, float]]) -> Dict[int, float]:
        """Normalize scores to [0, 1] range.

        Args:
            results: List of (id, score) tuples

        Returns:
            Dictionary mapping id to normalized score
        """
        if not results:
            return {}

        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return {doc_id: 1.0 for doc_id, _ in results}

        normalized = {}
        for doc_id, score in results:
            normalized[doc_id] = (score - min_score) / (max_score - min_score)

        return normalized

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[int, float]],
        sparse_results: List[Tuple[int, float]],
        k: int = 60
    ) -> List[Tuple[int, float]]:
        """Reciprocal Rank Fusion (RRF) for combining rankings.

        Args:
            dense_results: Dense retrieval results
            sparse_results: Sparse retrieval results
            k: RRF parameter (default 60)

        Returns:
            Fused results
        """
        rrf_scores: Dict[int, float] = {}

        # Add dense results
        for rank, (doc_id, _) in enumerate(dense_results, 1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

        # Add sparse results
        for rank, (doc_id, _) in enumerate(sparse_results, 1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

        # Sort by RRF score
        results = [(doc_id, score) for doc_id, score in rrf_scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def _linear_fusion(
        self,
        dense_results: List[Tuple[int, float]],
        sparse_results: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """Linear combination of normalized scores.

        Args:
            dense_results: Dense retrieval results
            sparse_results: Sparse retrieval results

        Returns:
            Fused results
        """
        # Normalize scores
        dense_scores = self._normalize_scores(dense_results)
        sparse_scores = self._normalize_scores(sparse_results)

        # Combine scores
        all_doc_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        combined_scores = {}

        for doc_id in all_doc_ids:
            dense_score = dense_scores.get(doc_id, 0)
            sparse_score = sparse_scores.get(doc_id, 0)

            combined_scores[doc_id] = (
                self.dense_weight * dense_score +
                self.sparse_weight * sparse_score
            )

        # Sort by combined score
        results = [(doc_id, score) for doc_id, score in combined_scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        k: int = 10,
        dense_k: Optional[int] = None,
        sparse_k: Optional[int] = None
    ) -> List[Tuple[int, float, str, Optional[Dict]]]:
        """Hybrid search combining dense and sparse retrieval.

        Args:
            query_text: Query text for BM25
            query_vector: Query vector for dense search
            k: Number of final results
            dense_k: Number of dense results to retrieve (default: 2*k)
            sparse_k: Number of sparse results to retrieve (default: 2*k)

        Returns:
            List of (doc_id, score, text, metadata) tuples
        """
        dense_k = dense_k or (2 * k)
        sparse_k = sparse_k or (2 * k)

        # Dense retrieval
        dense_results = self.vector_index.search(query_vector, k=dense_k)
        dense_results = [(doc_id, dist) for doc_id, dist, _ in dense_results]

        # Sparse retrieval
        sparse_results = self.bm25.search(query_text, k=sparse_k)

        # Fuse results
        if self.fusion_method == "rrf":
            fused = self._reciprocal_rank_fusion(dense_results, sparse_results)
        elif self.fusion_method == "linear":
            fused = self._linear_fusion(dense_results, sparse_results)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        # Get top k with document info
        results = []
        for doc_id, score in fused[:k]:
            if doc_id in self.documents:
                doc = self.documents[doc_id]
                results.append((doc_id, score, doc.text, doc.metadata))

        return results

    def rerank(
        self,
        query: str,
        results: List[Tuple[int, float, str, Optional[Dict]]],
        rerank_model=None
    ) -> List[Tuple[int, float, str, Optional[Dict]]]:
        """Rerank results using a cross-encoder or reranking model.

        Args:
            query: Query text
            results: Initial results
            rerank_model: Optional reranking model

        Returns:
            Reranked results
        """
        if rerank_model is None:
            return results

        # TODO: Implement cross-encoder reranking
        # This would use a model like sentence-transformers cross-encoder
        # to score query-document pairs more accurately

        return results

    def delete_document(self, doc_id: int) -> bool:
        """Delete document from hybrid index.

        Args:
            doc_id: Document ID

        Returns:
            True if deleted
        """
        # Delete from vector index
        vector_deleted = self.vector_index.delete(doc_id)

        # Remove from BM25 (simplified - full implementation would rebuild index)
        if doc_id in self.bm25.doc_term_freqs:
            del self.bm25.doc_term_freqs[doc_id]
            del self.bm25.doc_lengths[doc_id]
            self.bm25.num_docs -= 1

        # Remove document
        doc_deleted = doc_id in self.documents
        if doc_deleted:
            del self.documents[doc_id]

        return vector_deleted or doc_deleted