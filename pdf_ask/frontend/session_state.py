from enum import Enum


class VectorStorEnum(Enum):
    """Session state for vector store."""

    CURRENT_VECTOR_STORE: str = "current_vector_store"
    AVAILABLE_VECTOR_STORES: str = "available_vector_stores"
    CURRENT_EMBEDDING: str = "current_embedding"
    AVAILABLE_CURRENT_EMBEDDING: str = "available_current_embedding"


class DocumentsEnum(Enum):
    """Document state."""

    LOADER_KEY: str = "loader_key"
    UPLOADED_FILES: str = "uploaded_files"
    DOCUMENT_EMBEDDINGS_NAME: str = "document_embeddings_name"
    SELECTED_VECTOR_STORE: str = "selected_vector_store"
    NEW_VECTOR_STORE_NAME: str = "new_vector_store_name"
    RESOURCE_PATH: str = "resource_path"
    TEXT_SPLITER_NAME: str = "text_spliter_name"


class ChatEnum(Enum):
    """Document state."""

    CHAT_HISTORY: str = "chat_history"
    UPLOADED_FILES: str = "uploaded_files"
    DOCUMENT_EMBEDDINGS_NAME: str = "document_embeddings_name"
    SELECTED_VECTOR_STORE: str = "selected_vector_store"
    NEW_VECTOR_STORE_NAME: str = "new_vector_store_name"
    RESOURCE_PATH: str = "resource_path"
