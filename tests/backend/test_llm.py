from unittest.mock import MagicMock, Mock

from pdf_ask.backend.llm import ChatMessage, Role, SimpleRAGChatBot
from pdf_ask.backend.vector_store import VectorStoreProtocol

import pytest


@pytest.fixture
def mock_vector_store():
    return Mock(spec=VectorStoreProtocol)


@pytest.fixture
def mock_llm():
    return Mock()


@pytest.fixture
def mock_prompt_template():
    return Mock()


@pytest.fixture
def chatbot(mock_llm, mock_vector_store, mock_prompt_template):
    # Mock the chain creation
    mock_chain = Mock()

    def mock_or(other):
        return mock_chain

    mock_prompt_template.from_template.return_value = mock_prompt_template
    mock_prompt_template.__or__ = mock_or

    bot = SimpleRAGChatBot(llm=mock_llm, vector_store=mock_vector_store)
    bot.prompt = mock_prompt_template
    bot.chain = mock_chain
    return bot


def test_get_response(chatbot, mock_vector_store, mock_llm):
    question = ChatMessage(role=Role.USER, text="What is AI?")
    history = [ChatMessage(role=Role.BOT, text="Hello! How can I help you today?")]

    mock_vector_store.similarity_search.return_value = [
        {"id": "1", "content": "AI stands for Artificial Intelligence."}
    ]

    # Create a mock response object with a content attribute
    mock_response = Mock()
    mock_response.content = "AI stands for Artificial Intelligence. [1]"
    chatbot.chain.invoke.return_value = mock_response

    response = chatbot.get_response(question, history)

    assert response.text == "AI stands for Artificial Intelligence. [1]"
    assert response.documents == {"[1]": "AI stands for Artificial Intelligence."}
