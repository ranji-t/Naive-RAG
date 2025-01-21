# Third Party imports
from langchain_ollama.embeddings import OllamaEmbeddings


def get_ollama_embedder(model: str) -> OllamaEmbeddings:
    # Create Embedding
    embedding_function = OllamaEmbeddings(
        model=model,
    )
    # Return Embeddings
    return embedding_function
