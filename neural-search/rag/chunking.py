"""Document chunking strategies for RAG."""

from typing import List, Dict, Optional
from dataclasses import dataclass
import re


@dataclass
class Chunk:
    """A chunk of text with metadata."""
    text: str
    chunk_id: int
    doc_id: Optional[str] = None
    start_char: int = 0
    end_char: int = 0
    metadata: Optional[Dict] = None


class DocumentChunker:
    """Chunking strategies for splitting documents into retrievable segments."""

    def __init__(
        self,
        strategy: str = "fixed",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separator: str = "\n\n"
    ):
        """Initialize document chunker.

        Args:
            strategy: Chunking strategy ('fixed', 'sentence', 'paragraph', 'semantic')
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            separator: Separator for splitting (used in paragraph strategy)
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def chunk(self, text: str, doc_id: Optional[str] = None) -> List[Chunk]:
        """Chunk a document into segments.

        Args:
            text: Document text
            doc_id: Optional document ID

        Returns:
            List of chunks
        """
        if self.strategy == "fixed":
            return self._chunk_fixed(text, doc_id)
        elif self.strategy == "sentence":
            return self._chunk_sentence(text, doc_id)
        elif self.strategy == "paragraph":
            return self._chunk_paragraph(text, doc_id)
        elif self.strategy == "semantic":
            return self._chunk_semantic(text, doc_id)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")

    def _chunk_fixed(self, text: str, doc_id: Optional[str] = None) -> List[Chunk]:
        """Fixed-size chunking with overlap.

        Args:
            text: Document text
            doc_id: Optional document ID

        Returns:
            List of chunks
        """
        chunks = []
        chunk_id = 0
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Try to break at word boundary
            if end < len(text):
                # Find last space before end
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    start_char=start,
                    end_char=end
                ))
                chunk_id += 1

            # Move start with overlap
            start = end - self.chunk_overlap
            if start <= 0:
                start = end

        return chunks

    def _chunk_sentence(self, text: str, doc_id: Optional[str] = None) -> List[Chunk]:
        """Sentence-based chunking.

        Splits text into sentences and groups them into chunks.

        Args:
            text: Document text
            doc_id: Optional document ID

        Returns:
            List of chunks
        """
        # Simple sentence splitting (can be improved with nltk/spacy)
        sentence_endings = r'[.!?]+[\s\n]+'
        sentences = re.split(sentence_endings, text)

        chunks = []
        chunk_id = 0
        current_chunk = []
        current_length = 0
        char_pos = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_length = len(sentence)

            # Check if adding this sentence would exceed chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    start_char=char_pos - current_length,
                    end_char=char_pos
                ))
                chunk_id += 1

                # Start new chunk with overlap
                if self.chunk_overlap > 0 and len(current_chunk) > 1:
                    # Keep last sentence for overlap
                    overlap_sentence = current_chunk[-1]
                    current_chunk = [overlap_sentence, sentence]
                    current_length = len(overlap_sentence) + sentence_length
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

            char_pos += sentence_length

        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                doc_id=doc_id,
                start_char=char_pos - current_length,
                end_char=char_pos
            ))

        return chunks

    def _chunk_paragraph(self, text: str, doc_id: Optional[str] = None) -> List[Chunk]:
        """Paragraph-based chunking.

        Splits text by paragraphs and groups them into chunks.

        Args:
            text: Document text
            doc_id: Optional document ID

        Returns:
            List of chunks
        """
        paragraphs = text.split(self.separator)

        chunks = []
        chunk_id = 0
        current_chunk = []
        current_length = 0
        char_pos = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            para_length = len(paragraph)

            # Check if adding this paragraph would exceed chunk size
            if current_length + para_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = self.separator.join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    start_char=char_pos - current_length,
                    end_char=char_pos
                ))
                chunk_id += 1

                # Start new chunk
                current_chunk = [paragraph]
                current_length = para_length
            else:
                current_chunk.append(paragraph)
                current_length += para_length + len(self.separator)

            char_pos += para_length + len(self.separator)

        # Add final chunk
        if current_chunk:
            chunk_text = self.separator.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                doc_id=doc_id,
                start_char=char_pos - current_length,
                end_char=char_pos
            ))

        return chunks

    def _chunk_semantic(self, text: str, doc_id: Optional[str] = None) -> List[Chunk]:
        """Semantic chunking using embeddings.

        Groups semantically similar sentences together.
        This is a simplified version - full implementation would use embeddings.

        Args:
            text: Document text
            doc_id: Optional document ID

        Returns:
            List of chunks
        """
        # Fallback to sentence-based for now
        # Full implementation would:
        # 1. Split into sentences
        # 2. Embed each sentence
        # 3. Cluster similar sentences
        # 4. Group clusters into chunks

        return self._chunk_sentence(text, doc_id)

    def chunk_with_metadata(
        self,
        text: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """Chunk document and attach metadata to each chunk.

        Args:
            text: Document text
            doc_id: Optional document ID
            metadata: Metadata to attach to each chunk

        Returns:
            List of chunks with metadata
        """
        chunks = self.chunk(text, doc_id)

        # Attach metadata
        for chunk in chunks:
            chunk.metadata = metadata.copy() if metadata else {}
            chunk.metadata['chunk_id'] = chunk.chunk_id
            chunk.metadata['total_chunks'] = len(chunks)

        return chunks