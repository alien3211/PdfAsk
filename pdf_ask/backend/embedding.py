from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

ALLOWED_EMBEDDERS: dict[str, type[Embeddings]] = {"openAI": OpenAIEmbeddings}


class EmbedderNotAllowedError(Exception):
    """Exception raised when an embedder is not allowed."""

    pass


def get_embedding_instance(embedder_name: str, *args: Any, **kwargs: Any) -> Embeddings:
    """Retrieve an embedding instance based on the provided embedder name.

    Args:
        embedder_name (str): The name of the embedder to retrieve.
        *args: Additional arguments to pass to the embedder.
        **kwargs: Additional keyword arguments to pass to the embedder.

    Returns:
        Embeddings: An instance of the requested embedder.

    Raises:
        EmbedderNotAllowedException: If the embedder is not allowed.
    """
    if embedder_class := ALLOWED_EMBEDDERS.get(embedder_name):
        return embedder_class(*args, **kwargs)
    msg = f"{embedder_name} is not allowed"
    raise EmbedderNotAllowedError(msg)
