"""Basic usage example for Neural Search Engine."""

import sys
sys.path.append('..')

import numpy as np
from index import HNSWIndex, IVFIndex, HybridSearch
from rag import RAGPipeline, EmbeddingGenerator


def basic_vector_search_example():
    """Example of basic vector search using HNSW."""
    print("\n" + "="*60)
    print("Example 1: Basic Vector Search with HNSW")
    print("="*60 + "\n")

    # Create HNSW index
    index = HNSWIndex(dim=384, M=16, ef_construction=200)

    # Sample documents
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Neural networks are inspired by biological neural networks.",
        "Deep learning has revolutionized computer vision."
    ]

    print("Indexing documents...")
    # Simulate embeddings (in practice, use a real embedding model)
    for i, doc in enumerate(documents):
        # Random embedding for demo
        vector = np.random.randn(384).astype(np.float32)
        vector = vector / np.linalg.norm(vector)

        index.add(vector, metadata={'text': doc, 'doc_id': i})

    print(f"Indexed {len(documents)} documents\n")

    # Search
    print("Searching...")
    query_vector = np.random.randn(384).astype(np.float32)
    query_vector = query_vector / np.linalg.norm(query_vector)

    results = index.search(query_vector, k=3)

    print("Top 3 results:")
    for rank, (doc_id, distance, metadata) in enumerate(results, 1):
        print(f"{rank}. Distance: {distance:.4f}")
        print(f"   Text: {metadata.get('text', 'N/A')}\n")


def hybrid_search_example():
    """Example of hybrid search (dense + sparse)."""
    print("\n" + "="*60)
    print("Example 2: Hybrid Search (Dense + BM25)")
    print("="*60 + "\n")

    # Create HNSW index for dense vectors
    vector_index = HNSWIndex(dim=384, M=16, ef_construction=200)

    # Create hybrid search
    hybrid = HybridSearch(
        vector_index=vector_index,
        dense_weight=0.5,
        sparse_weight=0.5,
        fusion_method="rrf"
    )

    # Sample documents
    documents = [
        "Machine learning algorithms can learn from data.",
        "Deep neural networks have many layers.",
        "Natural language processing deals with text.",
        "Computer vision processes images and videos.",
        "Reinforcement learning learns from rewards."
    ]

    print("Indexing documents...")
    for i, doc in enumerate(documents):
        vector = np.random.randn(384).astype(np.float32)
        vector = vector / np.linalg.norm(vector)

        hybrid.add_document(text=doc, vector=vector, metadata={'source': f'doc_{i}'})

    print(f"Indexed {len(documents)} documents\n")

    # Search
    print("Searching for: 'neural networks'")
    query_text = "neural networks"
    query_vector = np.random.randn(384).astype(np.float32)
    query_vector = query_vector / np.linalg.norm(query_vector)

    results = hybrid.search(
        query_text=query_text,
        query_vector=query_vector,
        k=3
    )

    print("\nTop 3 results:")
    for rank, (doc_id, score, text, metadata) in enumerate(results, 1):
        print(f"{rank}. Score: {score:.4f}")
        print(f"   Text: {text}")
        print(f"   Metadata: {metadata}\n")


def rag_pipeline_example():
    """Example of RAG pipeline for question answering."""
    print("\n" + "="*60)
    print("Example 3: RAG Pipeline for Question Answering")
    print("="*60 + "\n")

    # Create index
    index = HNSWIndex(dim=384, M=16, ef_construction=200)

    # Note: In practice, use real embedding models
    # embedding_generator = EmbeddingGenerator()

    # Create RAG pipeline (with dummy components for demo)
    # rag = RAGPipeline(index=index, embedding_generator=embedding_generator)

    # Sample knowledge base
    documents = [
        {
            'text': "Python is a high-level programming language known for its simplicity and readability.",
            'metadata': {'source': 'programming_guide', 'topic': 'python'}
        },
        {
            'text': "Machine learning is a method of data analysis that automates analytical model building.",
            'metadata': {'source': 'ml_textbook', 'topic': 'machine_learning'}
        },
        {
            'text': "HNSW (Hierarchical Navigable Small World) is a graph-based algorithm for approximate nearest neighbor search.",
            'metadata': {'source': 'algorithms_paper', 'topic': 'search'}
        }
    ]

    print("Adding documents to knowledge base...")
    for doc in documents:
        # In practice, use: rag.add_document(doc['text'], metadata=doc['metadata'])
        vector = np.random.randn(384).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        index.add(vector, metadata=doc['metadata'])

    print(f"Added {len(documents)} documents\n")

    print("Querying RAG pipeline...")
    query = "What is Python?"

    # In practice: response = rag.query(query, k=2)
    print(f"Question: {query}")
    print("Answer: Python is a high-level programming language known for its simplicity...")
    print("Sources: [programming_guide]\n")


def performance_comparison():
    """Compare performance of different indices."""
    print("\n" + "="*60)
    print("Example 4: Performance Comparison")
    print("="*60 + "\n")

    dim = 128
    n_vectors = 1000
    n_queries = 10

    # Generate test data
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    queries = np.random.randn(n_queries, dim).astype(np.float32)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

    # Test HNSW
    print("Testing HNSW...")
    hnsw = HNSWIndex(dim=dim, M=16, ef_construction=200)

    import time
    start = time.time()
    for vec in vectors:
        hnsw.add(vec)
    hnsw_index_time = time.time() - start

    start = time.time()
    for query in queries:
        hnsw.search(query, k=10)
    hnsw_search_time = time.time() - start

    print(f"  Indexing: {hnsw_index_time:.3f}s")
    print(f"  Search (10 queries): {hnsw_search_time:.3f}s")
    print(f"  Avg query latency: {(hnsw_search_time/n_queries)*1000:.2f}ms\n")

    # Test IVF (needs training first)
    print("Testing IVF...")
    ivf = IVFIndex(dim=dim, n_clusters=10, n_probe=2)
    ivf.train(vectors[:500])  # Train on subset

    start = time.time()
    for vec in vectors:
        ivf.add(vec)
    ivf_index_time = time.time() - start

    start = time.time()
    for query in queries:
        ivf.search(query, k=10)
    ivf_search_time = time.time() - start

    print(f"  Indexing: {ivf_index_time:.3f}s")
    print(f"  Search (10 queries): {ivf_search_time:.3f}s")
    print(f"  Avg query latency: {(ivf_search_time/n_queries)*1000:.2f}ms\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Neural Search Engine - Examples")
    print("="*60)

    try:
        basic_vector_search_example()
        hybrid_search_example()
        rag_pipeline_example()
        performance_comparison()

        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()