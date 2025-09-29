"""Complete RAG pipeline integrating all components."""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import time

from .chunking import DocumentChunker, Chunk
from .embeddings import EmbeddingGenerator
from .retriever import Retriever, RetrievedDocument
from .generator import AnswerGenerator, GeneratedAnswer


@dataclass
class RAGResponse:
    """Complete RAG response."""
    answer: str
    sources: List[Dict]
    query_time_ms: float
    num_retrieved: int
    confidence: Optional[float] = None
    reasoning: Optional[str] = None


class RAGPipeline:
    """Complete Retrieval Augmented Generation pipeline.

    Integrates chunking, embedding, retrieval, and generation.
    """

    def __init__(
        self,
        index,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        retriever: Optional[Retriever] = None,
        generator: Optional[AnswerGenerator] = None,
        chunker: Optional[DocumentChunker] = None,
        top_k: int = 5,
        verbose: bool = False
    ):
        """Initialize RAG pipeline.

        Args:
            index: Vector index or hybrid search
            embedding_generator: Embedding generator (auto-created if None)
            retriever: Retriever (auto-created if None)
            generator: Answer generator (auto-created if None)
            chunker: Document chunker (auto-created if None)
            top_k: Number of documents to retrieve
            verbose: Print pipeline steps
        """
        self.index = index
        self.top_k = top_k
        self.verbose = verbose

        # Initialize components
        if embedding_generator is None:
            embedding_generator = EmbeddingGenerator()

        if chunker is None:
            chunker = DocumentChunker(strategy="sentence", chunk_size=512)

        if retriever is None:
            retriever = Retriever(
                index=index,
                embedding_generator=embedding_generator,
                top_k=top_k
            )

        if generator is None:
            generator = AnswerGenerator()

        self.embedding_generator = embedding_generator
        self.chunker = chunker
        self.retriever = retriever
        self.generator = generator

    def add_document(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        doc_id: Optional[str] = None
    ) -> List[int]:
        """Add document to the RAG pipeline.

        Chunks the document, generates embeddings, and indexes.

        Args:
            text: Document text
            metadata: Optional metadata
            doc_id: Optional document ID

        Returns:
            List of chunk IDs
        """
        if self.verbose:
            print(f"Adding document (length: {len(text)} chars)...")

        # Chunk document
        chunks = self.chunker.chunk_with_metadata(text, doc_id=doc_id, metadata=metadata)

        if self.verbose:
            print(f"Created {len(chunks)} chunks")

        # Generate embeddings for each chunk
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_generator.embed_batch(chunk_texts)

        if self.verbose:
            print(f"Generated embeddings (dim: {embeddings.shape[1]})")

        # Index each chunk
        chunk_ids = []
        for chunk, embedding in zip(chunks, embeddings):
            # Add chunk metadata
            chunk_metadata = chunk.metadata or {}
            chunk_metadata['text'] = chunk.text
            chunk_metadata['chunk_id'] = chunk.chunk_id
            chunk_metadata['doc_id'] = chunk.doc_id

            # Index chunk
            if hasattr(self.index, 'add_document'):
                # Hybrid search
                chunk_id = self.index.add_document(
                    text=chunk.text,
                    vector=embedding,
                    metadata=chunk_metadata
                )
            else:
                # Pure vector index
                chunk_id = self.index.add(
                    vector=embedding,
                    metadata=chunk_metadata
                )

            chunk_ids.append(chunk_id)

        if self.verbose:
            print(f"Indexed {len(chunk_ids)} chunks")

        return chunk_ids

    def add_documents_batch(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[List[int]]:
        """Add multiple documents in batch.

        Args:
            documents: List of document dicts with 'text', 'metadata', 'doc_id'

        Returns:
            List of chunk ID lists for each document
        """
        all_chunk_ids = []

        for doc in documents:
            text = doc.get('text', '')
            metadata = doc.get('metadata')
            doc_id = doc.get('doc_id')

            chunk_ids = self.add_document(text, metadata, doc_id)
            all_chunk_ids.append(chunk_ids)

        return all_chunk_ids

    def query(
        self,
        question: str,
        k: Optional[int] = None,
        include_sources: bool = True,
        use_chain_of_thought: bool = False
    ) -> RAGResponse:
        """Query the RAG pipeline.

        Args:
            question: User question
            k: Number of documents to retrieve (overrides default)
            include_sources: Whether to include source documents
            use_chain_of_thought: Use chain-of-thought reasoning

        Returns:
            RAG response with answer and sources
        """
        start_time = time.time()

        if self.verbose:
            print(f"Query: {question}")

        # Retrieve relevant documents
        k = k or self.top_k
        retrieved_docs = self.retriever.retrieve(question, k=k)

        if self.verbose:
            print(f"Retrieved {len(retrieved_docs)} documents")

        # Generate answer
        if use_chain_of_thought:
            generated = self.generator.generate_with_chain_of_thought(
                question,
                retrieved_docs
            )
        else:
            generated = self.generator.generate(
                question,
                retrieved_docs
            )

        end_time = time.time()
        query_time_ms = (end_time - start_time) * 1000

        if self.verbose:
            print(f"Generated answer in {query_time_ms:.2f}ms")

        # Build response
        sources = generated.sources if include_sources else []

        return RAGResponse(
            answer=generated.answer,
            sources=sources,
            query_time_ms=query_time_ms,
            num_retrieved=len(retrieved_docs),
            confidence=generated.confidence,
            reasoning=generated.reasoning
        )

    def batch_query(
        self,
        questions: List[str],
        k: Optional[int] = None
    ) -> List[RAGResponse]:
        """Process multiple queries in batch.

        Args:
            questions: List of questions
            k: Number of documents to retrieve

        Returns:
            List of RAG responses
        """
        responses = []

        for question in questions:
            response = self.query(question, k=k)
            responses.append(response)

        return responses

    def update_document(
        self,
        doc_id: str,
        new_text: str,
        metadata: Optional[Dict] = None
    ) -> List[int]:
        """Update an existing document.

        Deletes old chunks and re-indexes new content.

        Args:
            doc_id: Document ID to update
            new_text: New document text
            metadata: Optional metadata

        Returns:
            List of new chunk IDs
        """
        # TODO: Delete old chunks with this doc_id
        # This requires tracking which chunk IDs belong to which documents

        # Re-add document
        return self.add_document(new_text, metadata, doc_id)

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its chunks.

        Args:
            doc_id: Document ID

        Returns:
            True if deleted
        """
        # TODO: Find and delete all chunks with this doc_id
        # This requires tracking chunk IDs by document

        if hasattr(self.index, 'delete_document'):
            return self.index.delete_document(doc_id)

        return False

    def get_stats(self) -> Dict:
        """Get pipeline statistics.

        Returns:
            Statistics dictionary
        """
        stats = {
            'index_type': type(self.index).__name__,
            'embedding_dim': self.embedding_generator.get_dimension(),
            'chunk_strategy': self.chunker.strategy,
            'chunk_size': self.chunker.chunk_size,
            'top_k': self.top_k
        }

        # Add index stats if available
        if hasattr(self.index, '__len__'):
            stats['total_chunks'] = len(self.index)

        if hasattr(self.index, 'get_stats'):
            stats['index_stats'] = self.index.get_stats()

        return stats


class StreamingRAGPipeline(RAGPipeline):
    """RAG pipeline with streaming response support."""

    def query_stream(self, question: str, k: Optional[int] = None):
        """Query with streaming response.

        Args:
            question: User question
            k: Number of documents to retrieve

        Yields:
            Answer tokens as they're generated
        """
        # Retrieve documents
        k = k or self.top_k
        retrieved_docs = self.retriever.retrieve(question, k=k)

        # Stream generation (requires streaming-capable generator)
        # This is a placeholder - actual implementation depends on LLM
        answer = self.generator.generate(question, retrieved_docs).answer

        # Simulate streaming
        for token in answer.split():
            yield token + " "
            time.sleep(0.05)  # Simulate delay