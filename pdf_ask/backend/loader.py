from typing import Protocol, Self

import re
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter

DOC_PARSER: dict[str, type[BaseLoader]] = {".pdf": PyMuPDFLoader, ".txt": TextLoader}


class ParseDocumentError(Exception):
    """Error raised when document parsing fails."""

    pass


def clean_text(input_string: str) -> str:
    """Remove special characters and newline characters from a string.

    Args:
        input_string (str): The string to be cleaned.

    Returns:
        str: The cleaned string.
    """
    cleaned_string = re.sub(r"[^\w\s]", " ", input_string)
    return cleaned_string.replace("\n", " ")


def load_and_parse_document(
    file_path: str, splitter: TextSplitter | None = None
) -> list[Document]:
    """Load and parse a document from the given file path.

    Args:
        file_path (str): The path to the document file.
        splitter (TextSplitter, optional): An optional text splitter.

    Returns:
        list[Document]: A list of parsed documents.

    Raises:
        ParseDocumentException: If the file type is not supported.
    """
    file_path = Path(file_path)
    if loader := DOC_PARSER.get(file_path.suffix):
        raw_documents = loader(file_path.as_posix()).load()
        for document in raw_documents:
            document.page_content = clean_text(document.page_content)
        if splitter:
            return splitter.split_documents(raw_documents)
        return raw_documents

    msg = f"File type {file_path.suffix} not allowed"
    raise ParseDocumentError(msg)


class LoaderProtocol(Protocol):
    def __init__(self: Self, splitter: TextSplitter | None = None) -> None:
        """Initialize the loader with an optional text splitter."""
        ...

    def load_document(self: Self, file_path: str) -> list[Document]:
        """Load a document from the given file path.

        Args:
            file_path (str): The path to the document file.

        Returns:
            list[Document]: A list of loaded documents.
        """
        ...


class LocalLoader:
    def __init__(self: Self, splitter: TextSplitter | None = None) -> None:
        """Initialize the local loader with an optional text splitter."""
        self.splitter = splitter

    def load_document(self: Self, file_path: str) -> list[Document]:
        """Load and parse a document from the given file path.

        Args:
            file_path (str): The path to the document file.

        Returns:
            list[Document]: A list of parsed documents.
        """
        return load_and_parse_document(file_path=file_path, splitter=self.splitter)
