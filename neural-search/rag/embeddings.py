"""Embedding generation using HuggingFace models."""

import numpy as np
from typing import List, Optional, Union
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """Generate embeddings for text using pre-trained models."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True
    ):
        """Initialize embedding generator.

        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model
        self._load_model()

    def _load_model(self):
        """Load the embedding model."""
        try:
            # Try loading as sentence-transformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.model_type = "sentence_transformer"
            self.dim = self.model.get_sentence_embedding_dimension()
        except:
            # Fallback to transformers
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model_type = "transformers"

            # Get dimension from model config
            self.dim = self.model.config.hidden_size

    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text.

        Args:
            text: Single text or list of texts

        Returns:
            Embeddings array (dim,) or (N, dim)
        """
        if isinstance(text, str):
            text = [text]
            single = True
        else:
            single = False

        if self.model_type == "sentence_transformer":
            embeddings = self.model.encode(
                text,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True
            )
        else:
            embeddings = self._embed_transformers(text)

        if single:
            return embeddings[0]

        return embeddings

    def _embed_transformers(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using transformers library.

        Args:
            texts: List of texts

        Returns:
            Embeddings array (N, dim)
        """
        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)

                # Use [CLS] token or mean pooling
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    batch_embeddings = outputs.pooler_output
                else:
                    # Mean pooling
                    token_embeddings = outputs.last_hidden_state
                    attention_mask = inputs['attention_mask']
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

                # Normalize if requested
                if self.normalize:
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)

                embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts

        Returns:
            Embeddings array (N, dim)
        """
        return self.embed(texts)

    def get_dimension(self) -> int:
        """Get embedding dimension.

        Returns:
            Embedding dimension
        """
        return self.dim


class MultilingualEmbeddingGenerator(EmbeddingGenerator):
    """Embedding generator with multilingual support."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        **kwargs
    ):
        """Initialize multilingual embedding generator.

        Args:
            model_name: Multilingual model name
            **kwargs: Additional arguments for EmbeddingGenerator
        """
        super().__init__(model_name=model_name, **kwargs)


class CodeEmbeddingGenerator(EmbeddingGenerator):
    """Embedding generator specialized for code."""

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        **kwargs
    ):
        """Initialize code embedding generator.

        Args:
            model_name: Code model name
            **kwargs: Additional arguments for EmbeddingGenerator
        """
        super().__init__(model_name=model_name, **kwargs)


class EmbeddingCache:
    """Cache for embeddings to avoid recomputation."""

    def __init__(self, max_size: int = 10000):
        """Initialize embedding cache.

        Args:
            max_size: Maximum number of cached embeddings
        """
        self.cache: dict = {}
        self.max_size = max_size
        self.access_count: dict = {}

    def get(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding.

        Args:
            text: Text key

        Returns:
            Cached embedding or None
        """
        if text in self.cache:
            self.access_count[text] = self.access_count.get(text, 0) + 1
            return self.cache[text]
        return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        """Store embedding in cache.

        Args:
            text: Text key
            embedding: Embedding to cache
        """
        if len(self.cache) >= self.max_size:
            # Remove least accessed item
            least_accessed = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[least_accessed]
            del self.access_count[least_accessed]

        self.cache[text] = embedding
        self.access_count[text] = 1

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_count.clear()


class CachedEmbeddingGenerator(EmbeddingGenerator):
    """Embedding generator with caching."""

    def __init__(self, cache_size: int = 10000, **kwargs):
        """Initialize cached embedding generator.

        Args:
            cache_size: Maximum cache size
            **kwargs: Arguments for EmbeddingGenerator
        """
        super().__init__(**kwargs)
        self.cache = EmbeddingCache(max_size=cache_size)

    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings with caching.

        Args:
            text: Single text or list of texts

        Returns:
            Embeddings array
        """
        if isinstance(text, str):
            # Check cache
            cached = self.cache.get(text)
            if cached is not None:
                return cached

            # Generate and cache
            embedding = super().embed(text)
            self.cache.put(text, embedding)
            return embedding
        else:
            # For batches, generate all at once
            return super().embed(text)