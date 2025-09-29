"""REST API implementation with FastAPI."""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
from datetime import datetime
import asyncio

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


# Request/Response models
class VectorDocument(BaseModel):
    """Document with vector and metadata."""
    text: str = Field(..., description="Document text")
    vector: List[float] = Field(..., description="Document vector embedding")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")


class SearchRequest(BaseModel):
    """Search request."""
    query_text: Optional[str] = Field(None, description="Query text for hybrid search")
    query_vector: List[float] = Field(..., description="Query vector embedding")
    k: int = Field(10, ge=1, le=1000, description="Number of results")
    filter: Optional[Dict[str, Any]] = Field(None, description="Metadata filter")
    hybrid: bool = Field(False, description="Use hybrid search (dense + BM25)")


class SearchResult(BaseModel):
    """Search result item."""
    doc_id: int
    score: float
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    """Search response."""
    results: List[SearchResult]
    query_time_ms: float
    total_results: int


class IndexRequest(BaseModel):
    """Index document request."""
    documents: List[VectorDocument]


class IndexResponse(BaseModel):
    """Index response."""
    doc_ids: List[int]
    indexed_count: int
    success: bool


class DeleteRequest(BaseModel):
    """Delete request."""
    doc_ids: List[int]


class DeleteResponse(BaseModel):
    """Delete response."""
    deleted_count: int
    success: bool


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    index_size: int


class StatsResponse(BaseModel):
    """Statistics response."""
    total_documents: int
    index_type: str
    memory_usage_mb: float
    queries_per_second: float
    avg_query_latency_ms: float


# Dependency injection
class VectorDB:
    """Vector database instance (singleton)."""

    def __init__(self):
        self.index = None
        self.hybrid_search = None
        self.total_queries = 0
        self.total_query_time = 0.0

    def initialize(self, index, hybrid_search=None):
        """Initialize with index."""
        self.index = index
        self.hybrid_search = hybrid_search


# Global instance
vector_db = VectorDB()


def get_db() -> VectorDB:
    """Get database dependency."""
    if vector_db.index is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    return vector_db


# Create FastAPI app
def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    app = FastAPI(
        title="Neural Search API",
        description="High-performance vector search engine with RAG capabilities",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Rate limiter
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0",
            index_size=len(vector_db.index) if vector_db.index else 0
        )

    @app.get("/stats", response_model=StatsResponse)
    async def get_stats(db: VectorDB = Depends(get_db)):
        """Get index statistics."""
        index_size = len(db.index) if hasattr(db.index, '__len__') else 0

        qps = db.total_queries / max(1, db.total_query_time) if db.total_query_time > 0 else 0
        avg_latency = (db.total_query_time / db.total_queries * 1000) if db.total_queries > 0 else 0

        return StatsResponse(
            total_documents=index_size,
            index_type=type(db.index).__name__,
            memory_usage_mb=0.0,  # TODO: Implement memory tracking
            queries_per_second=qps,
            avg_query_latency_ms=avg_latency
        )

    @app.post("/api/index", response_model=IndexResponse, status_code=status.HTTP_201_CREATED)
    @limiter.limit("100/minute")
    async def index_documents(
        request: IndexRequest,
        background_tasks: BackgroundTasks,
        db: VectorDB = Depends(get_db)
    ):
        """Index one or more documents.

        Adds documents to the vector index for searching.
        """
        try:
            doc_ids = []

            for doc in request.documents:
                vector = np.array(doc.vector, dtype=np.float32)

                if db.hybrid_search:
                    doc_id = db.hybrid_search.add_document(
                        text=doc.text,
                        vector=vector,
                        metadata=doc.metadata
                    )
                else:
                    doc_id = db.index.add(
                        vector=vector,
                        metadata=doc.metadata
                    )

                doc_ids.append(doc_id)

            return IndexResponse(
                doc_ids=doc_ids,
                indexed_count=len(doc_ids),
                success=True
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

    @app.post("/api/search", response_model=SearchResponse)
    @limiter.limit("1000/minute")
    async def search(
        request: SearchRequest,
        db: VectorDB = Depends(get_db)
    ):
        """Search for similar documents.

        Performs vector similarity search and optionally hybrid search.
        """
        start_time = asyncio.get_event_loop().time()

        try:
            query_vector = np.array(request.query_vector, dtype=np.float32)

            if request.hybrid and db.hybrid_search and request.query_text:
                # Hybrid search
                results = db.hybrid_search.search(
                    query_text=request.query_text,
                    query_vector=query_vector,
                    k=request.k
                )

                search_results = [
                    SearchResult(
                        doc_id=doc_id,
                        score=float(score),
                        text=text,
                        metadata=metadata
                    )
                    for doc_id, score, text, metadata in results
                ]
            else:
                # Pure vector search
                results = db.index.search(query_vector, k=request.k)

                search_results = [
                    SearchResult(
                        doc_id=doc_id,
                        score=float(distance),
                        metadata=metadata
                    )
                    for doc_id, distance, metadata in results
                ]

            end_time = asyncio.get_event_loop().time()
            query_time_ms = (end_time - start_time) * 1000

            # Update stats
            db.total_queries += 1
            db.total_query_time += (end_time - start_time)

            return SearchResponse(
                results=search_results,
                query_time_ms=query_time_ms,
                total_results=len(search_results)
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    @app.delete("/api/delete", response_model=DeleteResponse)
    @limiter.limit("100/minute")
    async def delete_documents(
        request: DeleteRequest,
        db: VectorDB = Depends(get_db)
    ):
        """Delete documents from index.

        Removes documents by their IDs.
        """
        try:
            deleted_count = 0

            for doc_id in request.doc_ids:
                if db.hybrid_search:
                    success = db.hybrid_search.delete_document(doc_id)
                else:
                    success = db.index.delete(doc_id)

                if success:
                    deleted_count += 1

            return DeleteResponse(
                deleted_count=deleted_count,
                success=deleted_count > 0
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

    @app.post("/api/bulk_search")
    @limiter.limit("100/minute")
    async def bulk_search(
        queries: List[SearchRequest],
        db: VectorDB = Depends(get_db)
    ):
        """Perform multiple searches in batch."""
        results = []

        for query in queries:
            try:
                result = await search(query, db)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})

        return {"results": results}

    @app.get("/api/document/{doc_id}")
    async def get_document(
        doc_id: int,
        db: VectorDB = Depends(get_db)
    ):
        """Retrieve a specific document by ID."""
        if db.hybrid_search and doc_id in db.hybrid_search.documents:
            doc = db.hybrid_search.documents[doc_id]
            return {
                "doc_id": doc.id,
                "text": doc.text,
                "vector": doc.vector.tolist(),
                "metadata": doc.metadata
            }

        raise HTTPException(status_code=404, detail="Document not found")

    return app


# CLI for running server
if __name__ == "__main__":
    import uvicorn
    from ..index import HNSWIndex

    # Initialize index
    index = HNSWIndex(dim=768, M=16, ef_construction=200)
    vector_db.initialize(index)

    # Create app
    app = create_app()

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )