import argparse

import streamlit as st
from dotenv import load_dotenv

from pdf_ask.frontend.chat import chat, init_chat_session_state, reset_chat_session_state
from pdf_ask.frontend.documents import display_documents_embedding, init_documents_session_state
from pdf_ask.frontend.session_state import VectorStorEnum

load_dotenv()
from langchain.globals import set_verbose

set_verbose(True)


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, help="Path to config file", default="config/config.yml")
    # parser.add_argument("--reload_db", help="Reload database", action="store_true")

    return parser.parse_args()


def display_info():
    st.title("RAG Chatbot")


def initialize_session_state():
    init_chat_session_state()
    init_documents_session_state()
    if "available_vector_stores" not in st.session_state:
        st.session_state.available_vector_stores = None
    if "current_embedding" not in st.session_state:
        st.session_state.current_embedding = None


def display_sitebar():
    with st.sidebar:
        st.selectbox(
            "Select resources",
            st.session_state[VectorStorEnum.AVAILABLE_VECTOR_STORES.value],
            key=VectorStorEnum.CURRENT_VECTOR_STORE.value,
        )  # , on_change=load_or_create_vector_store)
    with st.sidebar:
        st.button(
            "Reset conversation", type="primary", on_click=reset_chat_session_state
        )


def main(args):
    display_info()
    initialize_session_state()
    display_documents_embedding()
    display_sitebar()

    chat(None)


if __name__ == "__main__":
    args = parse_args()
    main(args)
