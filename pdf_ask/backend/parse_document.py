from typing import Protocol, Self

import re
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter

DOC_PARSER: dict[str, type[BaseLoader]] = {".pdf": PyMuPDFLoader, ".txt": TextLoader}


class ParseDocumentException(Exception):
    pass


def remove_special_characters(input_string):
    # Remove special characters and newline characters
    cleaned_string = re.sub(r"[^\w\s]", " ", input_string)  # Remove special characters
    return cleaned_string.replace("\n", " ")  # Remove newline characters


def parse_document(
    file_path: str, spliter: TextSplitter | None = None
) -> list[Document]:
    file_path = Path(file_path)
    if loader := DOC_PARSER.get(file_path.suffix):
        raw_documents = loader(file_path.as_posix()).load()
        for document in raw_documents:
            document.page_content = remove_special_characters(document.page_content)
        if spliter:
            return spliter.split_documents(raw_documents)
        return raw_documents

    msg = f"File type {file_path.suffix} not allowed"
    raise ParseDocumentException(msg)


class LoaderProtocol(Protocol):
    def __init__(self: Self, spliter: TextSplitter | None = None) -> None: ...

    def load_document(self: Self, file_path: str) -> list[Document]: ...


class LocalLoader(LoaderProtocol):
    def __init__(self: Self, spliter: TextSplitter | None = None) -> None:
        self.spliter = spliter

    def load_document(self: Self, file_path: str) -> list[Document]:
        return parse_document(file_path=file_path, spliter=self.spliter)
