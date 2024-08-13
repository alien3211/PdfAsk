from pathlib import Path

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pdf_ask.backend.embedding import ALLOWED_EMBEDDERS, get_embeddings
from pdf_ask.backend.parse_document import LocalLoader
from pdf_ask.backend.vector_store import FaissVectorStore
from pdf_ask.frontend.session_state import DocumentsEnum, VectorStorEnum


def init_documents_session_state():
    if DocumentsEnum.LOADER_KEY.value not in st.session_state:
        st.session_state[DocumentsEnum.LOADER_KEY.value] = 0
        st.session_state[DocumentsEnum.UPLOADED_FILES.value] = []

    if DocumentsEnum.RESOURCE_PATH.value not in st.session_state:
        st.session_state[DocumentsEnum.RESOURCE_PATH.value] = "resources"


def load_vector_store(vector_store_name, file_paths, force: bool = True) -> None:
    resource_path = Path(st.session_state[DocumentsEnum.RESOURCE_PATH.value])
    vector_store_path = resource_path / vector_store_name
    vector_store = create_vector_store(vector_store_name)
    for file in file_paths:
        bytes_data = file.read()
        file_name = vector_store_path / file.name
        with file_name.open("wb") as f:
            f.write(bytes_data)
        vector_store.add_file(file_name, force=force)
    clean_document()


def create_vector_store(vector_store_name):
    resource_path = Path(st.session_state[DocumentsEnum.RESOURCE_PATH.value])
    vector_store_path = resource_path / vector_store_name
    embedding = get_embeddings(
        st.session_state[DocumentsEnum.DOCUMENT_EMBEDDINGS_NAME.value]
    )
    loader = LocalLoader(
        RecursiveCharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
    )
    return FaissVectorStore(loader, embedding, vector_store_path)


def get_files_by_extension(directory_path, extensions):
    """Get files by extension in each folder within the specified directory.

    Args:
        directory_path (str): The path to the directory containing folders.
        extensions (list): List of file extensions to filter by.

    Returns:
        dict: A dictionary where keys are folder names and values are dictionaries
              containing the folder path and a list of files with the specified extensions.
    """
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
    st.session_state[DocumentsEnum.LOADER_KEY.value] += 1


def loader_unload():
    st.session_state[DocumentsEnum.UPLOADED_FILES.value].extend(
        st.session_state[st.session_state[DocumentsEnum.LOADER_KEY.value]]
    )
    loader_clear()


def file_remove(index):
    st.session_state[DocumentsEnum.UPLOADED_FILES.value].pop(index)


def clean_document():
    st.session_state[DocumentsEnum.LOADER_KEY.value] = 0
    st.session_state[DocumentsEnum.UPLOADED_FILES.value] = []


def update_file_exist(dict_of_list: dict) -> None:
    if vector_store_info := dict_of_list.get(
        st.session_state[DocumentsEnum.SELECTED_VECTOR_STORE.value]
    ):
        st.session_state[DocumentsEnum.UPLOADED_FILES.value] = vector_store_info[
            "files"
        ]
    else:
        clean_document()


def show_uploaded_files():
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
    # List the existing vector stores
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

        with row_1[1]:
            show_vector_store(vector_store_files)
