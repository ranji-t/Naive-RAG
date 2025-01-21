# Standard Imports
from typing import Any

# Third Party imports
from langchain_community.embeddings import OllamaEmbeddings


def get_ollama_embedder(config: dict[str, Any]) -> OllamaEmbeddings:
    # Create Embedding
    embedding_function = OllamaEmbeddings(
        model=config.get("embedder", {}).get("name"),
    )
    # Return Embeddings
    return embedding_function
