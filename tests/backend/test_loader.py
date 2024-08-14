from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from pdf_ask.backend.loader import (
    DOC_PARSER,
    ParseDocumentError,
    load_and_parse_document,
)

import pytest


@pytest.fixture
def mock_pdf_loader():
    mock_loader = MagicMock()
    mock_loader.return_value.load.return_value = [Document(page_content="PDF content")]
    return mock_loader


@pytest.fixture
def mock_txt_loader():
    mock_loader = MagicMock()
    mock_loader.return_value.load.return_value = [Document(page_content="TXT content")]
    return mock_loader


@patch.dict(
    DOC_PARSER,
    {
        ".pdf": MagicMock(
            return_value=MagicMock(load=lambda: [Document(page_content="PDF content")])
        )
    },
)
def test_load_and_parse_pdf():
    result = load_and_parse_document("test.pdf")
    assert len(result) == 1
    assert result[0].page_content == "PDF content"


@patch.dict(
    DOC_PARSER,
    {
        ".txt": MagicMock(
            return_value=MagicMock(load=lambda: [Document(page_content="TXT content")])
        )
    },
)
def test_load_and_parse_txt():
    result = load_and_parse_document("test.txt")
    assert len(result) == 1
    assert result[0].page_content == "TXT content"


def test_load_and_parse_unsupported_file():
    with pytest.raises(ParseDocumentError) as excinfo:
        load_and_parse_document("test.docx")
    assert "File type .docx not allowed" in str(excinfo.value)
