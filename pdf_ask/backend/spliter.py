from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import TextSplitter


class SimpleRecursiveCharacterTextSplitter(RecursiveCharacterTextSplitter):
    """A simple recursive character text splitter with predefined settings."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Init function."""
        super().__init__(
            *args,
            chunk_size=200,
            chunk_overlap=5,
            length_function=len,
            is_separator_regex=False,
            **kwargs,
        )


ALLOWED_SPLITTER: dict[str, type[TextSplitter]] = {
    "recursive": SimpleRecursiveCharacterTextSplitter
}


class TextSplitterNotAllowedError(Exception):
    """Exception raised when a text splitter is not allowed."""

    pass


def get_text_splitter_instance(
    splitter_name: str, *args: Any, **kwargs: Any
) -> TextSplitter:
    """Get an instance of a text splitter by name.

    Args:
        splitter_name (str): The name of the text splitter.
        *args (Any): Additional arguments to pass to the text splitter.
        **kwargs (Any): Additional keyword arguments to pass to the text splitter.

    Returns:
        TextSplitter: An instance of the requested text splitter.

    Raises:
        TextSplitterNotAllowedError: If the requested text splitter is not allowed.
    """
    if splitter_class := ALLOWED_SPLITTER.get(splitter_name):
        return splitter_class(*args, **kwargs)
    msg = f"{splitter_name} is not allowed"
    raise TextSplitterNotAllowedError(msg)
