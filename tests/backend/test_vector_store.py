# Python code

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from pdf_ask.backend.loader import LoaderProtocol
from pdf_ask.backend.vector_store import (
    FaissVectorStore,
    VectorStoreNotAllowedError,
    get_vector_store_class,
)

import pytest


@pytest.fixture
def mock_loader():
    return MagicMock(spec=LoaderProtocol)


@pytest.fixture
def mock_embeddings():
    mock = MagicMock(spec=Embeddings)
    mock.embed_query.return_value = [0.1, 0.2, 0.3]
    mock.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    return mock


@pytest.fixture
def vector_store(mock_loader, mock_embeddings, tmp_path):
    store_path = tmp_path / "vector_store"
    return FaissVectorStore(
        loader=mock_loader, embeddings=mock_embeddings, store_path=str(store_path)
    )


def test_initialization(vector_store):
    assert isinstance(vector_store, FaissVectorStore)
    assert vector_store.store_path.exists()


def test_list_documents(vector_store):
    assert vector_store.list_documents() == []


def test_add_file(vector_store, mock_loader):
    mock_loader.load_document.return_value = [
        Document(page_content="content", metadata={"source": "test_file"})
    ]
    vector_store.add_file("test_file")
    assert "test_file" in vector_store.documents_source


def test_similarity_search(vector_store, mock_loader):
    mock_loader.load_document.return_value = [
        Document(page_content="content", metadata={"source": "source"})
    ]
    vector_store.add_file("test_file")
    results = vector_store.similarity_search("query")
    assert len(results) > 0


def test_get_vector_store_class():
    assert get_vector_store_class("faiss") == FaissVectorStore
    with pytest.raises(VectorStoreNotAllowedError):
        get_vector_store_class("invalid_store")


def test_add_same_document(vector_store, mock_loader):
    mock_loader.load_document.return_value = [
        Document(page_content="content", metadata={"source": "test_file"})
    ]
    vector_store.add_file("test_file")
    assert "test_file" in vector_store.documents_source

    with pytest.raises(FileExistsError):
        vector_store.add_file("test_file")
