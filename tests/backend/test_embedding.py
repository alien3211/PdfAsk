import os
from unittest.mock import patch

from langchain_openai import OpenAIEmbeddings

from pdf_ask.backend.embedding import (
    ALLOWED_EMBEDDERS,
    EmbedderNotAllowedError,
    get_embedding_instance,
)

import pytest


@patch.dict(os.environ, {"OPENAI_API_KEY": "TEST"})
def test_get_embedding_instance_valid():
    embedder = get_embedding_instance("openAI")
    assert isinstance(embedder, OpenAIEmbeddings)


def test_get_embedding_instance_invalid():
    with pytest.raises(EmbedderNotAllowedError):
        get_embedding_instance("invalidEmbedder")


def test_allowed_embedders():
    assert "openAI" in ALLOWED_EMBEDDERS
