"""Retriever for fetching relevant documents."""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class RetrievedDocument:
    """A retrieved document with score."""
    text: str
    score: float
    doc_id: Optional[int] = None
    metadata: Optional[Dict] = None
    chunk_id: Optional[int] = None


class Retriever:
    """Retriever for RAG pipeline.

    Handles retrieving relevant documents and reranking.
    """

    def __init__(
        self,
        index,
        embedding_generator,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        reranker=None
    ):
        """Initialize retriever.

        Args:
            index: Vector index or hybrid search
            embedding_generator: Embedding generator for queries
            top_k: Number of documents to retrieve
            score_threshold: Minimum score threshold
            reranker: Optional reranker model
        """
        self.index = index
        self.embedding_generator = embedding_generator
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[RetrievedDocument]:
        """Retrieve relevant documents for a query.

        Args:
            query: Query text
            k: Number of documents (overrides self.top_k)
            filter_metadata: Optional metadata filter

        Returns:
            List of retrieved documents
        """
        k = k or self.top_k

        # Generate query embedding
        query_embedding = self.embedding_generator.embed(query)

        # Search index
        if hasattr(self.index, 'search') and hasattr(self.index, 'bm25'):
            # Hybrid search
            results = self.index.search(
                query_text=query,
                query_vector=query_embedding,
                k=k * 2  # Retrieve more for reranking
            )

            retrieved = [
                RetrievedDocument(
                    text=text,
                    score=score,
                    doc_id=doc_id,
                    metadata=metadata
                )
                for doc_id, score, text, metadata in results
            ]
        else:
            # Pure vector search
            results = self.index.search(query_embedding, k=k * 2)

            retrieved = []
            for doc_id, distance, metadata in results:
                # Convert distance to similarity score
                score = 1.0 / (1.0 + distance)

                # Get text from metadata if available
                text = metadata.get('text', '') if metadata else ''

                retrieved.append(RetrievedDocument(
                    text=text,
                    score=score,
                    doc_id=doc_id,
                    metadata=metadata
                ))

        # Apply metadata filter if provided
        if filter_metadata:
            retrieved = self._filter_by_metadata(retrieved, filter_metadata)

        # Apply score threshold
        if self.score_threshold is not None:
            retrieved = [doc for doc in retrieved if doc.score >= self.score_threshold]

        # Rerank if reranker is available
        if self.reranker:
            retrieved = self._rerank(query, retrieved)

        # Return top k
        return retrieved[:k]

    def _filter_by_metadata(
        self,
        documents: List[RetrievedDocument],
        filter_metadata: Dict
    ) -> List[RetrievedDocument]:
        """Filter documents by metadata.

        Args:
            documents: Documents to filter
            filter_metadata: Metadata filter

        Returns:
            Filtered documents
        """
        filtered = []

        for doc in documents:
            if doc.metadata is None:
                continue

            # Check if all filter criteria match
            match = True
            for key, value in filter_metadata.items():
                if key not in doc.metadata or doc.metadata[key] != value:
                    match = False
                    break

            if match:
                filtered.append(doc)

        return filtered

    def _rerank(
        self,
        query: str,
        documents: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """Rerank documents using cross-encoder.

        Args:
            query: Query text
            documents: Documents to rerank

        Returns:
            Reranked documents
        """
        if not documents:
            return documents

        # Prepare query-document pairs
        pairs = [(query, doc.text) for doc in documents]

        # Get reranker scores
        scores = self.reranker.predict(pairs)

        # Update scores and sort
        for doc, score in zip(documents, scores):
            doc.score = float(score)

        documents.sort(key=lambda x: x.score, reverse=True)

        return documents

    def retrieve_with_context(
        self,
        query: str,
        k: Optional[int] = None,
        context_window: int = 1
    ) -> List[RetrievedDocument]:
        """Retrieve documents with surrounding context chunks.

        Args:
            query: Query text
            k: Number of documents
            context_window: Number of adjacent chunks to include

        Returns:
            Retrieved documents with context
        """
        # Retrieve initial documents
        retrieved = self.retrieve(query, k=k)

        # TODO: Implement context expansion
        # This would fetch adjacent chunks based on chunk_id

        return retrieved


class DenseRetriever(Retriever):
    """Dense retriever using only vector search."""

    def __init__(self, **kwargs):
        """Initialize dense retriever."""
        super().__init__(**kwargs)


class SparseRetriever:
    """Sparse retriever using only BM25."""

    def __init__(
        self,
        bm25_index,
        top_k: int = 5
    ):
        """Initialize sparse retriever.

        Args:
            bm25_index: BM25 index
            top_k: Number of documents to retrieve
        """
        self.bm25 = bm25_index
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[RetrievedDocument]:
        """Retrieve documents using BM25.

        Args:
            query: Query text
            k: Number of documents

        Returns:
            Retrieved documents
        """
        k = k or self.top_k

        results = self.bm25.search(query, k=k)

        retrieved = []
        for doc_id, score in results:
            # Get document text if available
            text = self.bm25.doc_term_freqs.get(doc_id, {})

            retrieved.append(RetrievedDocument(
                text=str(text),
                score=score,
                doc_id=doc_id
            ))

        return retrieved


class HybridRetriever(Retriever):
    """Hybrid retriever combining dense and sparse search."""

    def __init__(
        self,
        hybrid_search,
        embedding_generator,
        alpha: float = 0.5,
        **kwargs
    ):
        """Initialize hybrid retriever.

        Args:
            hybrid_search: Hybrid search index
            embedding_generator: Embedding generator
            alpha: Weight for dense vs sparse (0=sparse, 1=dense)
            **kwargs: Additional Retriever arguments
        """
        super().__init__(
            index=hybrid_search,
            embedding_generator=embedding_generator,
            **kwargs
        )
        self.alpha = alpha