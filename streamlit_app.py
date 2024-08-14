import logging

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from pdf_ask.frontend.chat import (
    chat_interface,
    clear_chat_history,
    init_chat_session_state,
)
from pdf_ask.frontend.documents import (
    display_documents_embedding,
    init_documents_session_state,
)
from pdf_ask.frontend.session_state import VectorStorEnum

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def display_title():
    """Display the main title of the application."""
    st.title("Ask Your Data 🤖")


def initialize_session_state():
    """Initialize session state variables for chat and documents."""
    init_chat_session_state()
    init_documents_session_state()
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-35-turbo"
    if "model_temperature" not in st.session_state:
        st.session_state["model_temperature"] = 0.3


def display_sidebar():
    """Display the sidebar with options to select resources and reset conversation."""
    with st.sidebar:
        st.selectbox(
            "Select resources",
            st.session_state[VectorStorEnum.AVAILABLE_VECTOR_STORES.value],
            key=VectorStorEnum.CURRENT_VECTOR_STORE.value,
        )
        st.button("Reset conversation", type="primary", on_click=clear_chat_history)
        st.selectbox("OpenAI model:", ["gpt-4o", "gpt-35-turbo"], key="openai_model")
        st.slider("Temperature", 0.0, 1.0, 0.3, step=0.01, key="model_temperature")


@st.cache_resource
def get_llm_model(model_name, temperature):
    """Get the language model with the specified name and temperature.

    Args:
        model_name (str): The name of the model.
        temperature (float): The temperature setting for the model.

    Returns:
        ChatOpenAI: An instance of the ChatOpenAI model.
    """
    logger.info(f"Getting model {model_name} with temperature: {temperature}")
    return ChatOpenAI(name=model_name, temperature=temperature)


def main():
    """Main function to run the Streamlit application."""
    load_dotenv()
    display_title()
    initialize_session_state()
    display_documents_embedding()
    display_sidebar()

    llm = get_llm_model(
        st.session_state["openai_model"], st.session_state["model_temperature"]
    )
    chat_interface(llm)


if __name__ == "__main__":
    main()
