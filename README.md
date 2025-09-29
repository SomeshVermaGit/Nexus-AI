# Neural Search Engine

A high-performance vector database and retrieval-augmented generation (RAG) system built from scratch. Features custom implementations of HNSW, IVF, and Product Quantization, hybrid search combining dense vectors with BM25, and a complete RAG pipeline.

---

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

---

## 📁 Project Structure

```bash
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

