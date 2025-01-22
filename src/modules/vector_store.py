# Standard Imports
from typing import Any

# Third party imports
from chromadb import Embeddings, HttpClient
from chromadb.config import Settings
from langchain_chroma.vectorstores import Chroma


def get_chroma_store(
    embedding_function: Embeddings,
    chroma_config: dict[str, Any],
    collection_name: str,
) -> Chroma:
    # Create Chroma Store
    chroma_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        client=get_chroma_clinet(chroma_config=chroma_config),
    )
    # Return Chroma Store
    return chroma_store


def get_chroma_clinet(chroma_config: dict[str, Any]) -> HttpClient:
    # Create Settings for Chroma DB
    settings = Settings(
        chroma_client_auth_provider=chroma_config.get("chroma_client_auth_provider"),
        chroma_client_auth_credentials=chroma_config.get(
            "chroma_client_auth_credentials"
        ),
    )

    # Create data
    http_client = HttpClient(
        host=chroma_config.get("host"),
        port=chroma_config.get("port"),
        settings=settings,
    )

    return http_client
