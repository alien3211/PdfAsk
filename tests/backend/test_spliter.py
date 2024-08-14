from pdf_ask.backend.spliter import (
    SimpleRecursiveCharacterTextSplitter,
    TextSplitterNotAllowedError,
    get_text_splitter_instance,
)

import pytest


def test_get_text_splitter_instance_allowed():
    splitter = get_text_splitter_instance("recursive")
    assert isinstance(splitter, SimpleRecursiveCharacterTextSplitter)


def test_get_text_splitter_instance_not_allowed():
    with pytest.raises(TextSplitterNotAllowedError):
        get_text_splitter_instance("non_recursive")
