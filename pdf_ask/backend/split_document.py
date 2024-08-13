from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import TextSplitter
from pydantic import BaseModel

from pdf_ask.backend.parse_document import Document


class SplitDocument(BaseModel):
    documents: list[Document]


class SplitStrategy:
    def __init__(
        self, name: str, split_class: type[TextSplitter], *args, **kwargs
    ) -> None:
        self.name = name
        self.spliter = split_class(*args, **kwargs)

    def split_document(self, document: Document) -> SplitDocument:
        return SplitDocument(
            documents=self.spliter.create_documents(list(document.content))
        )


class RecursiveCharacterSplitStrategy(SplitStrategy):
    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 200) -> None:
        super().__init__("RecursiveCharacterSplit", RecursiveCharacterTextSplitter)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def params() -> dict[str, Any]:
        return {"chunk_size": int, "chunk_overlap": int}
