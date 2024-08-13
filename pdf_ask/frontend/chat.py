import streamlit as st
from langchain_openai import ChatOpenAI

from pdf_ask.backend.llm import ChatMessage, Role, SimpleRAGChatBot
from pdf_ask.frontend.documents import create_vector_store
from pdf_ask.frontend.session_state import VectorStorEnum
from pdf_ask.frontend.tooltip import replace_with_tooltips


def init_chat_session_state():
    # Chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    if "vector_store_dict" not in st.session_state:
        st.session_state.vector_store_dict = {}


def reset_chat_session_state():
    st.session_state.history = []


def add_message_and_return(role, message, documents=None):
    chat_message = ChatMessage(role, message, documents)
    st.session_state.history.append(chat_message)
    return chat_message


def display_chat_history():
    for message in st.session_state.history:
        _display_message(message)


def _display_message(message: ChatMessage) -> None:
    with st.chat_message(message.role.value):
        text = message.text
        if documents := message.documents:
            text = replace_with_tooltips(text, documents)
        st.markdown(text, unsafe_allow_html=True)


def ask_question(bot):
    if question := st.chat_input("Ask a question"):
        message = add_message_and_return(Role.USER, question)
        _display_message(message)

        answer = bot.get_response(message, st.session_state.history)
        message = add_message_and_return(Role.BOT, answer.text, answer.documents)
        _display_message(message)


def chat(llm: None) -> None:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    if st.session_state[VectorStorEnum.CURRENT_VECTOR_STORE.value]:
        vector_store = create_vector_store(
            st.session_state[VectorStorEnum.CURRENT_VECTOR_STORE.value]
        )
        rag_bot = SimpleRAGChatBot(llm, vector_store)
        display_chat_history()
        ask_question(rag_bot)
    else:
        st.markdown("# Please provide vector store.")
