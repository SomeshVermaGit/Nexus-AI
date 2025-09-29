"""GraphQL API implementation with Strawberry."""

import strawberry
from typing import List, Optional
import numpy as np
from datetime import datetime


# GraphQL Types
@strawberry.type
class Metadata:
    """Metadata key-value pairs."""
    key: str
    value: str


@strawberry.type
class SearchResult:
    """Search result item."""
    doc_id: int
    score: float
    text: Optional[str] = None
    metadata: Optional[List[Metadata]] = None


@strawberry.type
class SearchResponse:
    """Search response."""
    results: List[SearchResult]
    query_time_ms: float
    total_results: int


@strawberry.type
class IndexResponse:
    """Index operation response."""
    doc_ids: List[int]
    indexed_count: int
    success: bool


@strawberry.type
class DeleteResponse:
    """Delete operation response."""
    deleted_count: int
    success: bool


@strawberry.type
class Stats:
    """Index statistics."""
    total_documents: int
    index_type: str
    memory_usage_mb: float
    queries_per_second: float


@strawberry.type
class Health:
    """Health status."""
    status: str
    timestamp: str
    version: str
    index_size: int


# Input Types
@strawberry.input
class VectorDocumentInput:
    """Document input for indexing."""
    text: str
    vector: List[float]
    metadata: Optional[List[str]] = None


@strawberry.input
class SearchInput:
    """Search query input."""
    query_text: Optional[str] = None
    query_vector: List[float]
    k: int = 10
    hybrid: bool = False


# Queries
@strawberry.type
class Query:
    """GraphQL query operations."""

    @strawberry.field
    def health(self) -> Health:
        """Get health status."""
        # Access vector_db from context
        from .rest import vector_db

        return Health(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0",
            index_size=len(vector_db.index) if vector_db.index else 0
        )

    @strawberry.field
    def stats(self) -> Stats:
        """Get index statistics."""
        from .rest import vector_db

        if vector_db.index is None:
            raise Exception("Database not initialized")

        index_size = len(vector_db.index) if hasattr(vector_db.index, '__len__') else 0
        qps = vector_db.total_queries / max(1, vector_db.total_query_time) if vector_db.total_query_time > 0 else 0

        return Stats(
            total_documents=index_size,
            index_type=type(vector_db.index).__name__,
            memory_usage_mb=0.0,
            queries_per_second=qps
        )

    @strawberry.field
    def search(self, input: SearchInput) -> SearchResponse:
        """Search for similar documents."""
        import time
        from .rest import vector_db

        if vector_db.index is None:
            raise Exception("Database not initialized")

        start_time = time.time()

        try:
            query_vector = np.array(input.query_vector, dtype=np.float32)

            if input.hybrid and vector_db.hybrid_search and input.query_text:
                # Hybrid search
                results = vector_db.hybrid_search.search(
                    query_text=input.query_text,
                    query_vector=query_vector,
                    k=input.k
                )

                search_results = [
                    SearchResult(
                        doc_id=doc_id,
                        score=float(score),
                        text=text,
                        metadata=[Metadata(key=k, value=str(v)) for k, v in (metadata or {}).items()]
                    )
                    for doc_id, score, text, metadata in results
                ]
            else:
                # Pure vector search
                results = vector_db.index.search(query_vector, k=input.k)

                search_results = [
                    SearchResult(
                        doc_id=doc_id,
                        score=float(distance),
                        metadata=[Metadata(key=k, value=str(v)) for k, v in (metadata or {}).items()]
                    )
                    for doc_id, distance, metadata in results
                ]

            end_time = time.time()
            query_time_ms = (end_time - start_time) * 1000

            # Update stats
            vector_db.total_queries += 1
            vector_db.total_query_time += (end_time - start_time)

            return SearchResponse(
                results=search_results,
                query_time_ms=query_time_ms,
                total_results=len(search_results)
            )

        except Exception as e:
            raise Exception(f"Search failed: {str(e)}")

    @strawberry.field
    def get_document(self, doc_id: int) -> Optional[SearchResult]:
        """Get a specific document by ID."""
        from .rest import vector_db

        if vector_db.hybrid_search and doc_id in vector_db.hybrid_search.documents:
            doc = vector_db.hybrid_search.documents[doc_id]
            return SearchResult(
                doc_id=doc.id,
                score=0.0,
                text=doc.text,
                metadata=[Metadata(key=k, value=str(v)) for k, v in (doc.metadata or {}).items()]
            )

        return None


# Mutations
@strawberry.type
class Mutation:
    """GraphQL mutation operations."""

    @strawberry.mutation
    def index_documents(self, documents: List[VectorDocumentInput]) -> IndexResponse:
        """Index one or more documents."""
        from .rest import vector_db

        if vector_db.index is None:
            raise Exception("Database not initialized")

        try:
            doc_ids = []

            for doc in documents:
                vector = np.array(doc.vector, dtype=np.float32)

                # Parse metadata
                metadata = None
                if doc.metadata:
                    metadata = {}
                    for item in doc.metadata:
                        if '=' in item:
                            key, value = item.split('=', 1)
                            metadata[key] = value

                if vector_db.hybrid_search:
                    doc_id = vector_db.hybrid_search.add_document(
                        text=doc.text,
                        vector=vector,
                        metadata=metadata
                    )
                else:
                    doc_id = vector_db.index.add(
                        vector=vector,
                        metadata=metadata
                    )

                doc_ids.append(doc_id)

            return IndexResponse(
                doc_ids=doc_ids,
                indexed_count=len(doc_ids),
                success=True
            )

        except Exception as e:
            raise Exception(f"Indexing failed: {str(e)}")

    @strawberry.mutation
    def delete_documents(self, doc_ids: List[int]) -> DeleteResponse:
        """Delete documents from index."""
        from .rest import vector_db

        if vector_db.index is None:
            raise Exception("Database not initialized")

        try:
            deleted_count = 0

            for doc_id in doc_ids:
                if vector_db.hybrid_search:
                    success = vector_db.hybrid_search.delete_document(doc_id)
                else:
                    success = vector_db.index.delete(doc_id)

                if success:
                    deleted_count += 1

            return DeleteResponse(
                deleted_count=deleted_count,
                success=deleted_count > 0
            )

        except Exception as e:
            raise Exception(f"Deletion failed: {str(e)}")


# Schema
schema = strawberry.Schema(query=Query, mutation=Mutation)


# FastAPI integration
def create_graphql_app():
    """Create GraphQL endpoint for FastAPI."""
    from strawberry.fastapi import GraphQLRouter

    graphql_app = GraphQLRouter(schema)
    return graphql_app