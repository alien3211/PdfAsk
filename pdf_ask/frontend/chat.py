import logging

import streamlit as st
from langchain_core.language_models.chat_models import BaseChatModel

from pdf_ask.backend.llm import ChatMessage, Role, SimpleRAGChatBot
from pdf_ask.frontend.documents import create_vector_store
from pdf_ask.frontend.session_state import ChatEnum, VectorStorEnum
from pdf_ask.frontend.tooltip import replace_text_with_tooltips

logger = logging.getLogger(__name__)


def init_chat_session_state():
    """Initialize the chat session state by setting up the chat history."""
    if ChatEnum.CHAT_HISTORY.value not in st.session_state:
        st.session_state[ChatEnum.CHAT_HISTORY.value] = [
            ChatMessage(Role.BOT, "How can I help you? 🤖")
        ]
        logger.info("Chat session state initialized.")


def clear_chat_history():
    """Clear the chat history in the session state."""
    st.session_state[ChatEnum.CHAT_HISTORY.value] = [
        ChatMessage(Role.BOT, "How can I help you? 🤖")
    ]
    logger.info("Chat history cleared.")


def add_message(role: Role, message: str, documents: list | None = None) -> ChatMessage:
    """Add a message to the chat history and return the ChatMessage object.

    Args:
        role (Role): The role of the message sender (USER or BOT).
        message (str): The message content.
        documents (list, optional): List of documents related to the message.

    Returns:
        ChatMessage: The created ChatMessage object.
    """
    chat_message = ChatMessage(role, message, documents)
    st.session_state[ChatEnum.CHAT_HISTORY.value].append(chat_message)
    logger.info(f"Message added to chat history: {message}")
    return chat_message


def display_chat_history():
    """Display the chat history from the session state."""
    for message in st.session_state[ChatEnum.CHAT_HISTORY.value]:
        _display_message(message)


def _display_message(message: ChatMessage) -> None:
    """Display a single chat message.

    Args:
        message (ChatMessage): The chat message to display.
    """
    with st.chat_message(message.role.value):
        text = message.text
        if documents := message.documents:
            text = replace_text_with_tooltips(text, documents)
        st.markdown(text, unsafe_allow_html=True)


def handle_user_question(bot):
    """Handle the user's question input, get a response from the bot, and display both.

    Args:
        bot (SimpleRAGChatBot): The chatbot instance to get responses from.
    """
    if question := st.chat_input("Ask a question"):
        user_message = add_message(Role.USER, question)
        _display_message(user_message)

        bot_response = bot.get_response(
            user_message, st.session_state[ChatEnum.CHAT_HISTORY.value]
        )
        bot_message = add_message(Role.BOT, bot_response.text, bot_response.documents)
        _display_message(bot_message)


def chat_interface(llm: BaseChatModel) -> None:
    """Main chat interface function to handle the chat logic.

    Args:
        llm (BaseChatModel): The language model to use for the chatbot.
    """
    logger.debug(f"Use {llm=}")
    if st.session_state[VectorStorEnum.CURRENT_VECTOR_STORE.value]:
        vector_store = create_vector_store(
            st.session_state[VectorStorEnum.CURRENT_VECTOR_STORE.value]
        )
        rag_bot = SimpleRAGChatBot(llm, vector_store)
        display_chat_history()
        handle_user_question(rag_bot)
    else:
        st.warning("Empty Vector store, please first add a vector store.", icon="⚠️")
        logger.warning("No vector store provided.")
