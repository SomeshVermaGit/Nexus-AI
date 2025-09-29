# Neural Search Engine

A high-performance vector database and retrieval-augmented generation (RAG) system built from scratch. Features custom implementations of HNSW, IVF, and Product Quantization, hybrid search combining dense vectors with BM25, and a complete RAG pipeline.

## 🚀 Features

### Vector Indexing
- **HNSW (Hierarchical Navigable Small World)**: Graph-based approximate nearest neighbor search
- **IVF (Inverted File Index)**: Cluster-based fast retrieval with configurable probing
- **Product Quantization (PQ)**: Vector compression for memory-efficient storage
- **Hybrid Search**: Combines dense vector search with sparse BM25 retrieval

### Storage & Distribution
- **Sharding**: Distribute vectors across multiple partitions with configurable strategies
- **Persistence**: Save/load indices with versioning and snapshots
- **Distributed Coordination**: Multi-node deployment with replication and consistency levels

### RAG Pipeline
- **Document Chunking**: Multiple strategies (fixed, sentence, paragraph, semantic)
- **Embeddings**: HuggingFace model integration with caching
- **Retrieval**: Advanced retrieval with reranking and filtering
- **Generation**: LLM integration for answer generation with sources

### API & Interface
- **REST API**: FastAPI-based with rate limiting and authentication
- **GraphQL API**: Flexible query interface with Strawberry
- **React Dashboard**: Real-time monitoring, analytics, and search interface

### Performance
- **Benchmarking**: Speed and recall benchmarks vs Pinecone/Weaviate
- **Monitoring**: Prometheus metrics and Grafana dashboards
- **Optimization**: Batch operations, caching, and efficient indexing

## 📁 Project Structure

```
neural-search/
├── index/              # Vector indexing implementations
│   ├── hnsw.py        # HNSW algorithm
│   ├── ivf.py         # Inverted File Index
│   ├── pq.py          # Product Quantization
│   └── hybrid.py      # Hybrid search (BM25 + dense)
│
├── storage/            # Storage and persistence
│   ├── shard.py       # Sharding logic
│   ├── persistence.py # Save/load with versioning
│   └── distributed.py # Multi-node coordination
│
├── api/                # API endpoints
│   ├── rest.py        # FastAPI REST API
│   └── graphql_api.py # GraphQL API
│
├── rag/                # RAG pipeline
│   ├── chunking.py    # Document chunking
│   ├── embeddings.py  # Embedding generation
│   ├── retriever.py   # Retrieval logic
│   ├── generator.py   # Answer generation
│   └── pipeline.py    # Complete RAG pipeline
│
├── dashboard/          # React frontend
│   └── src/
│       ├── pages/     # Search, Analytics, Index, Settings
│       └── App.tsx
│
├── benchmarks/         # Performance benchmarks
│   ├── speed_benchmark.py
│   └── recall_benchmark.py
│
├── examples/           # Usage examples
│   └── basic_usage.py
│
├── deployment/         # Deployment configs
│   ├── kubernetes/
│   │   └── deployment.yaml
│   └── prometheus.yml
│
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 🔧 Installation

### Using Docker (Recommended)

```bash
# Start all services
docker-compose up -d

# Access API at http://localhost:8000
# Access Dashboard at http://localhost:3000
# Access Grafana at http://localhost:3001
```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run API server
python -m uvicorn api.rest:app --host 0.0.0.0 --port 8000

# Run dashboard (in dashboard/ directory)
cd dashboard
npm install
npm run dev
```

## 📖 Quick Start

### Basic Vector Search

```python
from index import HNSWIndex
import numpy as np

# Create index
index = HNSWIndex(dim=768, M=16, ef_construction=200)

# Add vectors
vector = np.random.randn(768).astype(np.float32)
vector = vector / np.linalg.norm(vector)  # Normalize

doc_id = index.add(vector, metadata={'text': 'Sample document'})

# Search
query = np.random.randn(768).astype(np.float32)
query = query / np.linalg.norm(query)

results = index.search(query, k=10)
for doc_id, distance, metadata in results:
    print(f"Doc {doc_id}: distance={distance:.4f}")
```

### Hybrid Search

```python
from index import HNSWIndex, HybridSearch

# Create hybrid search
vector_index = HNSWIndex(dim=768)
hybrid = HybridSearch(vector_index, dense_weight=0.5, sparse_weight=0.5)

# Add document
hybrid.add_document(
    text="Machine learning is a subset of AI.",
    vector=embedding,
    metadata={'source': 'textbook'}
)

# Search
results = hybrid.search(
    query_text="What is machine learning?",
    query_vector=query_embedding,
    k=5
)
```

### RAG Pipeline

```python
from rag import RAGPipeline, EmbeddingGenerator
from index import HNSWIndex

# Create pipeline
index = HNSWIndex(dim=384)
embedder = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
rag = RAGPipeline(index=index, embedding_generator=embedder)

# Add documents
rag.add_document("Python is a programming language.", metadata={'topic': 'python'})
rag.add_document("Machine learning uses algorithms.", metadata={'topic': 'ml'})

# Query
response = rag.query("What is Python?", k=3)
print(f"Answer: {response.answer}")
print(f"Sources: {response.sources}")
```

## 🌐 API Usage

### REST API

```bash
# Index a document
curl -X POST http://localhost:8000/api/index \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [{
      "text": "Sample document",
      "vector": [0.1, 0.2, ...],
      "metadata": {"source": "test"}
    }]
  }'

# Search
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query_vector": [0.1, 0.2, ...],
    "k": 10,
    "hybrid": true,
    "query_text": "sample query"
  }'

# Get stats
curl http://localhost:8000/stats
```

### GraphQL API

```graphql
# Search query
query {
  search(input: {
    queryVector: [0.1, 0.2, ...],
    k: 10,
    hybrid: true,
    queryText: "sample query"
  }) {
    results {
      docId
      score
      text
    }
    queryTimeMs
  }
}

# Index mutation
mutation {
  indexDocuments(documents: [{
    text: "Sample document",
    vector: [0.1, 0.2, ...],
    metadata: ["key=value"]
  }]) {
    docIds
    success
  }
}
```

## 🚢 Deployment

### Kubernetes

```bash
# Apply deployment
kubectl apply -f deployment/kubernetes/deployment.yaml

# Check status
kubectl get pods -l app=neural-search

# Scale
kubectl scale deployment neural-search-api --replicas=5
```

### Distributed Setup

```python
from storage import DistributedStore, Node

# Define nodes
nodes = [
    Node(node_id="node1", host="10.0.0.1", port=8000),
    Node(node_id="node2", host="10.0.0.2", port=8000),
    Node(node_id="node3", host="10.0.0.3", port=8000),
]

# Create distributed store
store = DistributedStore(
    nodes=nodes,
    replication_factor=2,
    consistency_level="quorum"
)

# Add document (automatically replicated)
await store.add(vector, metadata={'text': 'Sample'})

# Search across all nodes
results = await store.search(query_vector, k=10)
```

## 📊 Benchmarks

```bash
# Run speed benchmark
python benchmarks/speed_benchmark.py

# Run recall benchmark
python benchmarks/recall_benchmark.py

# Compare against other systems
python benchmarks/comparative_benchmark.py
```

## 🔍 Monitoring

- **Prometheus**: Metrics at http://localhost:9090
- **Grafana**: Dashboards at http://localhost:3001 (admin/admin)
- **API Docs**: Swagger UI at http://localhost:8000/docs

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=. tests/
```

## 🤝 Contributing

Contributions welcome! Please check out the [contribution guidelines](CONTRIBUTING.md).

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 📚 Documentation

Full documentation available in the `/docs` directory:
- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Performance Tuning](docs/performance.md)
- [Deployment Guide](docs/deployment.md)

## 🙏 Acknowledgments

- HNSW algorithm from Malkov & Yashunin (2018)
- Inspired by Pinecone, Weaviate, and FAISS
- Built with FastAPI, React, and HuggingFace Transformers