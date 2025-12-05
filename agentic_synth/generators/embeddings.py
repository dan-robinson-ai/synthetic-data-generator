"""Embeddings generator for RAG systems."""

import json
from collections.abc import AsyncGenerator
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentic_synth.generators.base import BaseDataGenerator


class EmbeddingsConfig(BaseModel):
    """Configuration for embeddings generation."""

    content_type: Literal[
        "qa_pairs",
        "documents",
        "paragraphs",
        "sentences",
        "code_snippets",
        "product_descriptions",
        "reviews",
        "articles",
    ] = Field(
        default="documents",
        description="Type of content to generate",
    )
    count: int = Field(
        default=50,
        gt=0,
        description="Number of items to generate",
    )
    context: str = Field(
        default="",
        description="Domain context for generation",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Model to use for generating embeddings",
    )
    include_embeddings: bool = Field(
        default=True,
        description="Whether to generate actual embeddings",
    )
    min_length: int = Field(
        default=50,
        gt=0,
        description="Minimum content length in characters",
    )
    max_length: int = Field(
        default=500,
        gt=0,
        description="Maximum content length in characters",
    )
    categories: list[str] | None = Field(
        default=None,
        description="Categories to distribute content across",
    )


class EmbeddingsGenerator(BaseDataGenerator):
    """Generator for text content with embeddings for RAG systems."""

    CONTENT_TEMPLATES = {
        "qa_pairs": "question and answer pair",
        "documents": "document with title and content",
        "paragraphs": "informative paragraph",
        "sentences": "standalone sentence",
        "code_snippets": "code snippet with explanation",
        "product_descriptions": "product description with features",
        "reviews": "review with rating",
        "articles": "article with title, summary, and content",
    }

    def generate(
        self,
        config: EmbeddingsConfig,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Generate content with embeddings.

        Args:
            config: Embeddings configuration
            **kwargs: Additional parameters

        Returns:
            List of content items with embeddings
        """
        content_desc = self.CONTENT_TEMPLATES.get(config.content_type, "text content")
        categories_str = ""
        if config.categories:
            categories_str = f"Categories: {', '.join(config.categories)}"

        prompt = f"""Generate {config.count} unique {content_desc}s for a RAG (Retrieval Augmented Generation) system.

{f"Domain/Context: {config.context}" if config.context else ""}
{categories_str}

Requirements for each item:
- Content length: {config.min_length}-{config.max_length} characters
- Must be informative and suitable for semantic search
- Include relevant metadata
- Ensure diversity across the dataset

Structure based on content type ({config.content_type}):
"""

        # Add type-specific structure requirements
        if config.content_type == "qa_pairs":
            prompt += """
- question: The question text
- answer: The comprehensive answer
- category: Topic category
- difficulty: easy, medium, or hard"""
        elif config.content_type == "documents":
            prompt += """
- title: Document title
- content: Main document content
- category: Document category
- keywords: List of relevant keywords"""
        elif config.content_type == "code_snippets":
            prompt += """
- language: Programming language
- code: The code snippet
- description: What the code does
- use_case: When to use this code"""
        elif config.content_type == "product_descriptions":
            prompt += """
- name: Product name
- description: Product description
- features: List of key features
- category: Product category
- price_range: low, medium, high, or premium"""
        elif config.content_type == "reviews":
            prompt += """
- title: Review title
- content: Review text
- rating: 1-5 stars
- product_or_service: What is being reviewed
- pros: List of positives
- cons: List of negatives"""
        elif config.content_type == "articles":
            prompt += """
- title: Article title
- summary: Brief summary
- content: Full article content
- author: Author name
- tags: List of tags"""
        else:
            prompt += """
- id: Unique identifier
- content: The main text content
- metadata: Additional relevant information"""

        prompt += "\n\nReturn a JSON array of objects. Respond with ONLY the JSON array."

        # Check cache (without embeddings since those are separate)
        cache_key = f"{prompt}_no_embeddings"
        cached = self._check_cache(
            prompt=cache_key,
            schema={"content_type": config.content_type},
            config=config.model_dump(exclude={"include_embeddings"}),
        )
        if cached is not None:
            # If we need embeddings and they're not in cache, generate them
            if config.include_embeddings and cached and "embedding" not in cached[0]:
                return self._add_embeddings(cached, config)
            return cached

        system_prompt = (
            "You are a content generator for RAG systems. Generate high-quality, "
            "diverse content that will be useful for semantic search and retrieval. "
            "Ensure content is informative, well-structured, and covers various aspects "
            "of the specified domain."
        )

        result = self.provider.generate_with_schema(
            prompt=prompt,
            schema={"type": "array", "items": {"type": "object"}},
            system_prompt=system_prompt,
            **kwargs,
        )

        if isinstance(result, dict):
            result = result.get("data", result.get("items", [result]))

        # Store content in cache (without embeddings)
        self._store_cache(
            value=result,
            prompt=cache_key,
            schema={"content_type": config.content_type},
            config=config.model_dump(exclude={"include_embeddings"}),
        )

        # Add embeddings if requested
        if config.include_embeddings:
            result = self._add_embeddings(result, config)

        return result

    def _add_embeddings(
        self,
        items: list[dict[str, Any]],
        config: EmbeddingsConfig,
    ) -> list[dict[str, Any]]:
        """Add embeddings to content items."""
        # Extract text for embedding
        texts = []
        for item in items:
            if config.content_type == "qa_pairs":
                text = f"{item.get('question', '')} {item.get('answer', '')}"
            elif config.content_type == "code_snippets":
                text = f"{item.get('description', '')} {item.get('code', '')}"
            else:
                text = item.get("content", item.get("description", str(item)))
            texts.append(text[:8000])  # Limit text length

        # Check if provider supports embeddings
        if hasattr(self.provider, "create_embeddings"):
            try:
                embeddings = self.provider.create_embeddings(
                    texts=texts,
                    model=config.embedding_model,
                )
                for item, embedding in zip(items, embeddings):
                    item["embedding"] = embedding
            except Exception:
                # If embeddings fail, continue without them
                pass

        return items

    async def stream(
        self,
        config: EmbeddingsConfig,
        batch_size: int = 10,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream content generation with embeddings.

        Args:
            config: Embeddings configuration
            batch_size: Number of items per batch
            **kwargs: Additional parameters

        Yields:
            Content items with embeddings one at a time
        """
        generated = 0
        content_desc = self.CONTENT_TEMPLATES.get(config.content_type, "text content")

        while generated < config.count:
            batch_count = min(batch_size, config.count - generated)

            prompt = f"""Generate {batch_count} unique {content_desc}s for a RAG system.

{f"Context: {config.context}" if config.context else ""}

This is batch {generated // batch_size + 1}. Generate unique content different from previous batches.

Content length: {config.min_length}-{config.max_length} characters.

Return a JSON array of objects."""

            system_prompt = (
                "You are a content generator for RAG systems. Generate high-quality, "
                "diverse content suitable for semantic search."
            )

            response = await self.provider.agenerate(
                prompt=prompt,
                system_prompt=system_prompt,
                **kwargs,
            )

            # Parse response
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]

            try:
                batch = json.loads(cleaned.strip())
                if isinstance(batch, dict):
                    batch = batch.get("data", batch.get("items", [batch]))

                # Add embeddings if requested
                if config.include_embeddings and hasattr(self.provider, "acreate_embeddings"):
                    texts = []
                    for item in batch:
                        text = item.get("content", item.get("description", str(item)))[:8000]
                        texts.append(text)

                    try:
                        embeddings = await self.provider.acreate_embeddings(
                            texts=texts,
                            model=config.embedding_model,
                        )
                        for item, embedding in zip(batch, embeddings):
                            item["embedding"] = embedding
                    except Exception:
                        pass

                for item in batch:
                    yield item
                    generated += 1
                    if generated >= config.count:
                        break
            except json.JSONDecodeError:
                continue
