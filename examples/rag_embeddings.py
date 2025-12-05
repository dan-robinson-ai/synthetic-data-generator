"""RAG embeddings generation example."""

from agentic_synth import GeneratorConfig, SyntheticGenerator
from agentic_synth.generators import EmbeddingsConfig


def main() -> None:
    """Demonstrate embeddings generation for RAG systems."""

    config = GeneratorConfig(
        provider="openai",
        model="gpt-4o-mini",
    )

    with SyntheticGenerator(config) as generator:
        # Generate Q&A pairs for a knowledge base
        print("Generating Q&A pairs for knowledge base...")
        qa_config = EmbeddingsConfig(
            content_type="qa_pairs",
            count=5,
            context="Customer support FAQ for a cloud hosting platform",
            embedding_model="text-embedding-3-small",
            include_embeddings=True,
            categories=["billing", "technical", "account", "security"],
        )

        qa_pairs = generator.generate_embeddings(qa_config)

        print("\nQ&A Pairs:")
        for qa in qa_pairs:
            print(f"\nQ: {qa.get('question', 'N/A')[:80]}...")
            print(f"A: {qa.get('answer', 'N/A')[:80]}...")
            if "embedding" in qa:
                print(f"   Embedding dims: {len(qa['embedding'])}")

        # Generate documentation
        print("\n\nGenerating documentation articles...")
        doc_config = EmbeddingsConfig(
            content_type="articles",
            count=3,
            context="Technical documentation for a REST API",
            include_embeddings=True,
            min_length=100,
            max_length=300,
        )

        docs = generator.generate_embeddings(doc_config)

        print("\nDocumentation Articles:")
        for doc in docs:
            print(f"\nTitle: {doc.get('title', 'N/A')}")
            print(f"Summary: {doc.get('summary', 'N/A')[:100]}...")
            if "embedding" in doc:
                print(f"Embedding dims: {len(doc['embedding'])}")

        # Generate code snippets
        print("\n\nGenerating code snippets...")
        code_config = EmbeddingsConfig(
            content_type="code_snippets",
            count=3,
            context="Python examples for data processing library",
            include_embeddings=True,
        )

        snippets = generator.generate_embeddings(code_config)

        print("\nCode Snippets:")
        for snippet in snippets:
            print(f"\nLanguage: {snippet.get('language', 'N/A')}")
            print(f"Description: {snippet.get('description', 'N/A')[:80]}...")
            if "embedding" in snippet:
                print(f"Embedding dims: {len(snippet['embedding'])}")


if __name__ == "__main__":
    main()
