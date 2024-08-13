from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

ALLOWED_EMBEDDERS: dict[str, type[Embeddings]] = {"openAI": OpenAIEmbeddings}


class EmbeddedNotAllowedException(Exception):
    pass


def get_embeddings(embeddings_name: str, *args, **kwargs) -> Embeddings:
    if embeddings := ALLOWED_EMBEDDERS.get(embeddings_name):
        return embeddings(*args, **kwargs)
    msg = f"{embeddings_name:r} is not allowed"
    raise EmbeddedNotAllowedException(msg)
