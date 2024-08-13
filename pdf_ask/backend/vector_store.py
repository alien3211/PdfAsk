from typing import Protocol, Self

from collections import defaultdict
from pathlib import Path

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from pdf_ask.backend.parse_document import LoaderProtocol


class VectorStoreProtocol(Protocol):
    def list_documents(self): ...

    def add_file(self: Self, file_path: str, force: bool = False) -> None: ...

    def similarity_search(self: Self, query: str, top_k: int = 10) -> list[dict]: ...


class FaissVectorStore(VectorStoreProtocol):
    def __init__(
        self, loader: LoaderProtocol, embeddings: Embeddings, store_path: str
    ) -> None:
        self.store_path = Path(store_path)
        self.embeddings = embeddings
        self._vector_store = self._load_vector_store()
        self.documents_source = self._get_documents_source()
        self.loader = loader

    def _get_documents_source(self):
        documents_source = defaultdict(list)

        for document_id, document in self._vector_store.docstore._dict.items():
            documents_source[document.metadata["source"]].append(document_id)

        return documents_source

    def _load_vector_store(self):
        if self.store_path.exists():
            return FAISS.load_local(
                self.store_path.as_posix(),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

        return self._build_vector_store()

    def _build_vector_store(self):
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
        return list(self._vector_store.docstore._dict)

    def add_file(self: Self, file_path: str, force: bool = False) -> None:
        document_exists = file_path in self.list_documents()
        if document_exists:
            if force:
                self._remove_document(file_path)
            else:
                msg = f"File {file_path} already exists in the vector store. Use force=True to overwrite it."
                raise FileExistsError(msg)

        documents = self.loader.load_document(file_path=file_path)
        self._add_documents(documents=documents)

    def _remove_document(self, file_path):
        ids = self.documents_source.pop(file_path)
        self._vector_store.delete(ids=ids)

    def _add_documents(self: Self, documents: list[Document]) -> None:
        ids = self._vector_store.add_documents(
            documents=documents, embeddings=self.embeddings
        )
        self._update_documents_source(ids, documents)
        self._vector_store.save_local(self.store_path.as_posix())

    def _update_documents_source(
        self: Self, ids: list[str], documents: list[Document]
    ) -> None:
        for _id, document in zip(ids, documents, strict=True):
            source = document.metadata["source"]
            if _id not in self.documents_source[source]:
                self.documents_source[source].append(_id)

    def similarity_search(self: Self, query: str, top_k: int = 10) -> list[dict]:
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
        return {"content": {document.page_content}, "id": idx}


AVAILABLE_VECTOR_STORES = {"faiss": FaissVectorStore}


def get_vector_store(name):
    return AVAILABLE_VECTOR_STORES[name]
