# Python code

from typing import Protocol, Self

from collections import defaultdict
from pathlib import Path

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from pdf_ask.backend.loader import LoaderProtocol


class VectorStoreProtocol(Protocol):
    def list_documents(self):
        """List all documents in the vector store.

        Returns:
            list: A list of document IDs.
        """
        ...

    def add_file(self: Self, file_path: str, force: bool = False) -> None:
        """Add a file to the vector store.

        Args:
            file_path (str): Path to the file.
            force (bool): Force overwrite if file exists.
        """

    def similarity_search(self: Self, query: str, top_k: int = 10) -> list[dict]:
        """Perform a similarity search on the vector store.

        Args:
            query (str): The search query.
            top_k (int): Number of top results to return.

        Returns:
            list[dict]: List of search results.
        """


class FaissVectorStore:
    def __init__(
        self, loader: LoaderProtocol, embeddings: Embeddings, store_path: str
    ) -> None:
        """Initialize the FaissVectorStore.

        Args:
            loader (LoaderProtocol): The document loader.
            embeddings (Embeddings): The embeddings model.
            store_path (str): Path to store the vector data.
        """
        self.store_path = Path(store_path)
        self.embeddings = embeddings
        self._vector_store = self._load_vector_store()
        self.documents_source = self._get_documents_source()
        self.loader = loader

    def _get_documents_source(self):
        """Get the source of documents.

        Returns:
            dict: A dictionary mapping sources to document IDs.
        """
        documents_source = defaultdict(list)
        for document_id, document in self._vector_store.docstore._dict.items():
            documents_source[document.metadata["source"]].append(document_id)
        return documents_source

    def _load_vector_store(self):
        """Load the vector store from the local path.

        Returns:
            FAISS: The loaded FAISS vector store.
        """
        if self.store_path.exists():
            return FAISS.load_local(
                self.store_path.as_posix(),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        return self._build_vector_store()

    def _build_vector_store(self):
        """Build a new vector store.

        Returns:
            FAISS: The newly built FAISS vector store.
        """
        index = faiss.IndexFlatL2(len(self.embeddings.embed_query("hello world")))
        vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        vector_store.save_local(self.store_path.as_posix())
        return vector_store

    def list_documents(self):
        """List all documents in the vector store.

        Returns:
            list: A list of document IDs.
        """
        return list(self._vector_store.docstore._dict)

    def list_sources(self):
        """List all sources in the vector store.

        Returns:
            list: A list of sources.
        """
        return list(self.documents_source)

    def add_file(self: Self, file_path: str, force: bool = False) -> None:
        """Add a file to the vector store.

        Args:
            file_path (str): Path to the file.
            force (bool): Force overwrite if file exists.
        """
        document_exists = file_path in self.list_sources()
        if document_exists:
            if force:
                self._remove_document(file_path)
            else:
                msg = f"File {file_path} already exists in the vector store. Use force=True to overwrite it."
                raise FileExistsError(msg)
        documents = self.loader.load_document(file_path=file_path)
        self._add_documents(documents=documents)

    def _remove_document(self, file_path):
        """Remove a document from the vector store.

        Args:
            file_path (str): Path to the file.
        """
        ids = self.documents_source.pop(file_path)
        self._vector_store.delete(ids=ids)

    def _add_documents(self: Self, documents: list[Document]) -> None:
        """Add documents to the vector store.

        Args:
            documents (list[Document]): List of documents to add.
        """
        ids = self._vector_store.add_documents(
            documents=documents, embeddings=self.embeddings
        )
        self._update_documents_source(ids, documents)
        self._vector_store.save_local(self.store_path.as_posix())

    def _update_documents_source(
        self: Self, ids: list[str], documents: list[Document]
    ) -> None:
        """Update the source of documents.

        Args:
            ids (list[str]): List of document IDs.
            documents (list[Document]): List of documents.
        """
        for _id, document in zip(ids, documents, strict=True):
            source = document.metadata["source"]
            if _id not in self.documents_source[source]:
                self.documents_source[source].append(_id)

    def similarity_search(self: Self, query: str, top_k: int = 10) -> list[dict]:
        """Perform a similarity search on the vector store.

        Args:
            query (str): The search query.
            top_k (int): Number of top results to return.

        Returns:
            list[dict]: List of search results.
        """
        if len(self.documents_source) == 0:
            msg = "No documents in the vector store."
            raise ValueError(msg)
        documents = self._vector_store.similarity_search(query, top_k)
        return [
            self._create_document_result(idx, document)
            for idx, document in enumerate(documents)
        ]

    @staticmethod
    def _create_document_result(idx: int, document: Document) -> dict:
        """Create a result dictionary for a document.

        Args:
            idx (int): Index of the document.
            document (Document): The document object.

        Returns:
            dict: A dictionary containing document content and ID.
        """
        return {"content": {document.page_content}, "id": idx}


class VectorStoreNotAllowedError(Exception):
    """Exception raised when a text splitter is not allowed."""

    pass


AVAILABLE_VECTOR_STORES: dict[str, type[VectorStoreProtocol]] = {
    "faiss": FaissVectorStore
}


def get_vector_store_class(vector_store_name):
    """Get the vector store by name.

    Args:
        vector_store_name (str): Name of the vector store.

    Returns:
        VectorStoreProtocol: The vector store class.
    """
    if splitter_class := AVAILABLE_VECTOR_STORES.get(vector_store_name):
        return splitter_class
    msg = f"{vector_store_name} is not allowed"
    raise VectorStoreNotAllowedError(msg)
