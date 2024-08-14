import logging
from pathlib import Path

import streamlit as st

from pdf_ask.backend.embedding import ALLOWED_EMBEDDERS, get_embedding_instance
from pdf_ask.backend.loader import LocalLoader
from pdf_ask.backend.spliter import ALLOWED_SPLITTER, get_text_splitter_instance
from pdf_ask.backend.vector_store import FaissVectorStore
from pdf_ask.frontend.session_state import DocumentsEnum, VectorStorEnum

logger = logging.getLogger(__name__)


def init_documents_session_state():
    """Initialize the session state for documents."""
    if DocumentsEnum.LOADER_KEY.value not in st.session_state:
        st.session_state[DocumentsEnum.LOADER_KEY.value] = 0
        st.session_state[DocumentsEnum.UPLOADED_FILES.value] = []

    if DocumentsEnum.RESOURCE_PATH.value not in st.session_state:
        st.session_state[DocumentsEnum.RESOURCE_PATH.value] = "resources"


def load_vector_store(
    vector_store_name: str, file_paths: list, force: bool = True
) -> None:
    """Load files into the vector store.

    Args:
        vector_store_name (str): Name of the vector store.
        file_paths (list): List of file paths to load.
        force (bool): Whether to forcefully add files to the vector store.
    """
    logger.info(f"Loading vector store: {vector_store_name}")
    resource_path = Path(st.session_state[DocumentsEnum.RESOURCE_PATH.value])
    vector_store_path = resource_path / vector_store_name
    vector_store = create_vector_store(vector_store_name)
    for file in file_paths:
        logger.info(f"Loading vector store: {vector_store_name}")
        bytes_data = file.read()
        file_name = vector_store_path / file.name
        with file_name.open("wb") as f:
            f.write(bytes_data)
        vector_store.add_file(file_name, force=force)
    clean_document()


def create_vector_store(vector_store_name):
    """Create a vector store.

    Args:
        vector_store_name (str): Name of the vector store.

    Returns:
        FaissVectorStore: An instance of FaissVectorStore.
    """
    logging.info(f"Creating vector store: {vector_store_name}")
    resource_path = Path(st.session_state[DocumentsEnum.RESOURCE_PATH.value])
    vector_store_path = resource_path / vector_store_name
    embedding = get_embedding_instance(
        st.session_state[DocumentsEnum.DOCUMENT_EMBEDDINGS_NAME.value]
    )
    loader = LocalLoader(
        get_text_splitter_instance(
            st.session_state[DocumentsEnum.TEXT_SPLITER_NAME.value]
        )
    )
    return FaissVectorStore(loader, embedding, vector_store_path.as_posix())


def get_files_by_extension(directory_path, extensions):
    """Get files by extension in each folder within the specified directory.

    Args:
        directory_path (str): The path to the directory containing folders.
        extensions (list): List of file extensions to filter by.

    Returns:
        dict: A dictionary where keys are folder names and values are dictionaries
              containing the folder path and a list of files with the specified extensions.
    """
    logging.info(f"Getting files by extension in directory: {directory_path}")
    folder_dict = {}
    path = Path(directory_path)
    path.mkdir(exist_ok=True, parents=True)
    for item in path.iterdir():
        if item.is_dir():
            file_list = [
                file
                for file in item.iterdir()
                if file.is_file() and file.suffix in extensions
            ]
            folder_dict[item.name] = {"path": item.as_posix(), "files": file_list}
    return folder_dict


def loader_clear():
    """Clear the loader by incrementing the loader key in session state."""
    st.session_state[DocumentsEnum.LOADER_KEY.value] += 1


def loader_unload():
    """Unload the loader by extending uploaded files in session state."""
    st.session_state[DocumentsEnum.UPLOADED_FILES.value].extend(
        st.session_state[st.session_state[DocumentsEnum.LOADER_KEY.value]]
    )
    loader_clear()


def file_remove(index):
    """Remove a file from uploaded files in session state by index.

    Args:
        index (int): Index of the file to remove.
    """
    st.session_state[DocumentsEnum.UPLOADED_FILES.value].pop(index)


def clean_document():
    """Reset loader key and uploaded files in session state."""
    st.session_state[DocumentsEnum.LOADER_KEY.value] = 0
    st.session_state[DocumentsEnum.UPLOADED_FILES.value] = []


def update_file_exist(dict_of_list: dict) -> None:
    """Update uploaded files in session state based on selected vector store.

    Args:
        dict_of_list (dict): Dictionary of vector store information.
    """
    if vector_store_info := dict_of_list.get(
        st.session_state[DocumentsEnum.SELECTED_VECTOR_STORE.value]
    ):
        st.session_state[DocumentsEnum.UPLOADED_FILES.value] = vector_store_info[
            "files"
        ]
    else:
        clean_document()


def show_uploaded_files():
    """Display uploaded files with remove buttons."""
    for index, file in enumerate(st.session_state[DocumentsEnum.UPLOADED_FILES.value]):
        loaded = st.columns([3, 10])
        loaded[0].button(
            "Remove", key=f"file{index}", on_click=file_remove, args=[index]
        )
        loaded[1].write(file.name)

    css = """
    <style>
        [data-testid="stFileUploadDropzone] + div {
            display: none;
        }
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)


def show_vector_store(vector_store_files: dict[str, list[dict[str, str]]]) -> None:
    """Display a select box for vector stores and handle creation or update.

    Args:
        vector_store_files (dict): Dictionary of vector store files.
    """
    vector_store_list = list(vector_store_files.keys())
    vector_store_list = ["<NEW VECTOR STORE>", *vector_store_list]

    existing_vector_store = st.selectbox(
        "Vector Store to Merge the Knowledge",
        vector_store_list,
        help="Which vector store to add the new documents or create new provide a new name.",
        key=DocumentsEnum.SELECTED_VECTOR_STORE.value,
        on_change=update_file_exist,
        args=[vector_store_files],
    )
    if existing_vector_store:
        if existing_vector_store == "<NEW VECTOR STORE>":
            vector_store_name = st.text_input(
                "Vector store name", placeholder="Enter new vector store name"
            )

            create_button = st.button("Create")
            if (
                vector_store_name
                and create_button
                and st.session_state[DocumentsEnum.UPLOADED_FILES.value]
            ):
                load_vector_store(
                    vector_store_name,
                    st.session_state[DocumentsEnum.UPLOADED_FILES.value],
                )
        else:
            st.button("Update")


def display_documents_embedding():
    """Display the document embedding interface."""
    available_extensions = [".pdf", ".txt"]
    vector_store_files = get_files_by_extension("resources", available_extensions)
    st.session_state[VectorStorEnum.AVAILABLE_VECTOR_STORES.value] = list(
        vector_store_files.keys()
    )
    with st.expander("Document Embedding"):
        st.markdown(
            "This page is used to upload the documents as the custom knowledge for the chatbot."
        )
        st.file_uploader(
            "Knowledge Documents",
            type=available_extensions,
            help=f"{','.join(available_extensions)} file",
            accept_multiple_files=True,
            key=st.session_state[DocumentsEnum.LOADER_KEY.value],
            on_change=loader_unload,
        )

        show_uploaded_files()

        row_1 = st.columns(2)
        with row_1[0]:
            st.selectbox(
                "Model Name of the Instruct Embeddings",
                ALLOWED_EMBEDDERS,
                key=DocumentsEnum.DOCUMENT_EMBEDDINGS_NAME.value,
            )
            st.selectbox(
                "Model Name of the Instruct Embeddings",
                ALLOWED_SPLITTER,
                key=DocumentsEnum.TEXT_SPLITER_NAME.value,
            )

        with row_1[1]:
            show_vector_store(vector_store_files)
