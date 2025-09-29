"""Answer generation using LLMs."""

from typing import List, Optional, Dict
from dataclasses import dataclass


@dataclass
class GeneratedAnswer:
    """Generated answer with metadata."""
    answer: str
    sources: List[Dict]
    confidence: Optional[float] = None
    reasoning: Optional[str] = None


class AnswerGenerator:
    """Generate answers using LLMs with retrieved context."""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        max_context_length: int = 2048,
        temperature: float = 0.7,
        include_sources: bool = True
    ):
        """Initialize answer generator.

        Args:
            model_name: LLM model name
            max_context_length: Maximum context length
            temperature: Sampling temperature
            include_sources: Whether to include source citations
        """
        self.model_name = model_name
        self.max_context_length = max_context_length
        self.temperature = temperature
        self.include_sources = include_sources

        # Load model
        self._load_model()

    def _load_model(self):
        """Load the LLM model."""
        # This is a placeholder - actual implementation would load the model
        # Could use OpenAI API, HuggingFace models, or local models
        self.model = None

    def generate(
        self,
        query: str,
        retrieved_docs: List,
        system_prompt: Optional[str] = None
    ) -> GeneratedAnswer:
        """Generate answer given query and retrieved documents.

        Args:
            query: User query
            retrieved_docs: Retrieved documents from retriever
            system_prompt: Optional system prompt

        Returns:
            Generated answer
        """
        # Build context from retrieved documents
        context = self._build_context(retrieved_docs)

        # Build prompt
        prompt = self._build_prompt(query, context, system_prompt)

        # Generate answer
        answer_text = self._call_llm(prompt)

        # Extract sources
        sources = []
        if self.include_sources:
            sources = [
                {
                    'doc_id': doc.doc_id,
                    'text': doc.text[:200],  # Snippet
                    'score': doc.score,
                    'metadata': doc.metadata
                }
                for doc in retrieved_docs
            ]

        return GeneratedAnswer(
            answer=answer_text,
            sources=sources
        )

    def _build_context(self, retrieved_docs: List) -> str:
        """Build context string from retrieved documents.

        Args:
            retrieved_docs: Retrieved documents

        Returns:
            Context string
        """
        context_parts = []
        current_length = 0

        for i, doc in enumerate(retrieved_docs):
            doc_text = f"[Source {i+1}]: {doc.text}"

            # Check if adding this would exceed max length
            if current_length + len(doc_text) > self.max_context_length:
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        return "\n\n".join(context_parts)

    def _build_prompt(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Build prompt for LLM.

        Args:
            query: User query
            context: Context from retrieved documents
            system_prompt: Optional system prompt

        Returns:
            Complete prompt
        """
        if system_prompt is None:
            system_prompt = """You are a helpful AI assistant. Answer the user's question based on the provided context.
If you cannot answer based on the context, say so. Always cite your sources using [Source N] notation."""

        prompt = f"""{system_prompt}

Context:
{context}

Question: {query}

Answer:"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call LLM to generate answer.

        Args:
            prompt: Complete prompt

        Returns:
            Generated answer text
        """
        # Placeholder implementation
        # Actual implementation would call OpenAI API or run local model

        # Example with OpenAI (requires openai package):
        # import openai
        # response = openai.ChatCompletion.create(
        #     model=self.model_name,
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=self.temperature
        # )
        # return response.choices[0].message.content

        return "This is a placeholder answer. Please implement _call_llm() with your LLM of choice."

    def generate_with_chain_of_thought(
        self,
        query: str,
        retrieved_docs: List
    ) -> GeneratedAnswer:
        """Generate answer with chain-of-thought reasoning.

        Args:
            query: User query
            retrieved_docs: Retrieved documents

        Returns:
            Generated answer with reasoning
        """
        context = self._build_context(retrieved_docs)

        cot_prompt = f"""Let's solve this step by step:

Context:
{context}

Question: {query}

Reasoning steps:
1. First, let me identify the key information from the context.
2. Then, I'll analyze how it relates to the question.
3. Finally, I'll formulate a comprehensive answer.

Answer:"""

        answer_text = self._call_llm(cot_prompt)

        # Extract reasoning and answer (simplified)
        parts = answer_text.split("Answer:", 1)
        reasoning = parts[0].strip() if len(parts) > 1 else None
        answer = parts[1].strip() if len(parts) > 1 else answer_text

        sources = [
            {
                'doc_id': doc.doc_id,
                'text': doc.text[:200],
                'score': doc.score
            }
            for doc in retrieved_docs
        ]

        return GeneratedAnswer(
            answer=answer,
            sources=sources,
            reasoning=reasoning
        )


class OpenAIGenerator(AnswerGenerator):
    """Answer generator using OpenAI API."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        **kwargs
    ):
        """Initialize OpenAI generator.

        Args:
            api_key: OpenAI API key
            model_name: Model name
            **kwargs: Additional arguments
        """
        self.api_key = api_key
        super().__init__(model_name=model_name, **kwargs)

    def _load_model(self):
        """Load OpenAI client."""
        try:
            import openai
            openai.api_key = self.api_key
            self.client = openai
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    def _call_llm(self, prompt: str) -> str:
        """Call OpenAI API.

        Args:
            prompt: Prompt text

        Returns:
            Generated text
        """
        try:
            response = self.client.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"


class HuggingFaceGenerator(AnswerGenerator):
    """Answer generator using HuggingFace models."""

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: str = "cuda",
        **kwargs
    ):
        """Initialize HuggingFace generator.

        Args:
            model_name: HuggingFace model name
            device: Device to run on
            **kwargs: Additional arguments
        """
        self.device = device
        super().__init__(model_name=model_name, **kwargs)

    def _load_model(self):
        """Load HuggingFace model."""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")

    def _call_llm(self, prompt: str) -> str:
        """Call HuggingFace model.

        Args:
            prompt: Prompt text

        Returns:
            Generated text
        """
        try:
            import torch

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.max_context_length,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=200,
                    temperature=self.temperature,
                    do_sample=True
                )

            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return answer
        except Exception as e:
            return f"Error generating answer: {str(e)}"