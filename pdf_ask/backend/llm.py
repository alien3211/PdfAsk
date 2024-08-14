from typing import Self

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from pdf_ask.backend.vector_store import VectorStoreProtocol

logger = logging.getLogger(__name__)


class Role(Enum):
    USER = "User"
    BOT = "Assistant"


@dataclass
class ChatMessage:
    role: Role
    text: str
    timestamp: str = field(init=False)
    documents: dict[str, str] | None = None

    def __post_init__(self):
        self.timestamp = datetime.now().strftime("%H:%M:%S")

    def __str__(self):
        return f"{self.timestamp} {self.role.value}: {self.text} {self.documents=}"


@dataclass
class LlmAnswer:
    text: str
    documents: dict[str, str] | None = None

    def __str__(self):
        return f"{self.text} {self.documents=}"


class SimpleRAGChatBot:
    """A simple Retrieval-Augmented Generation (RAG) chatbot.

    Attributes:
        llm: The language model used for generating responses.
        vector_store: The vector store used for similarity search.
        top_k: The number of top similar documents to retrieve.
        prompt: The chat prompt template.
        chain: The combined prompt and language model chain.
    """

    rag_prompt = """
You are an assistant for question-answering tasks. Use only the provided pieces of retrieved context to
answer the question. If the context does not contain the answer, respond with "I don't know."
If you use any piece of context to answer, include a reference in the format [n]
where "n" represents the nth piece of context, e.g., [1]. Additionally, aim to maintain continuity with the chat
history to ensure a coherent conversation.

Question: {question}

Context:
{context}

Chat history:
{chat_history}

Answer:
"""

    def __init__(
        self: Self,
        llm: BaseChatModel,
        vector_store: VectorStoreProtocol,
        top_k: int = 3,
    ) -> None:
        """Initializes the SimpleRAGChatBot.

        Args:
            llm: The language model used for generating responses.
            vector_store: The vector store used for similarity search.
            top_k: The number of top similar documents to retrieve.
        """
        self.top_k = top_k
        self.llm = llm
        self.vector_store = vector_store
        self.prompt = ChatPromptTemplate.from_template(self.rag_prompt)
        self.chain = self.prompt | self.llm

    def get_response(
        self: Self, question: ChatMessage, history: list[ChatMessage]
    ) -> LlmAnswer:
        """Generates a response to a given question based on chat history and similar documents.

        Args:
            question: The chat message containing the user's question.
            history: The list of previous chat messages.

        Returns:
            An LlmAnswer object containing the generated response and related documents.
        """
        logger.info(f"Searched for similar documents to '{question.text}'")
        similar_documents = self.vector_store.similarity_search(
            question.text, top_k=self.top_k
        )
        logger.debug(f"Found {len(similar_documents)} similar documents")

        response = self.chain.invoke(
            {
                "question": question.text,
                "context": "\n".join(
                    [f"[{doc['id']}] - {doc['content']}" for doc in similar_documents]
                ),
                "chat_history": [
                    {"role": msg.role.value, "text": msg.text} for msg in history
                ],
            }
        )

        logger.debug(f"Response: {response.content}")
        return LlmAnswer(
            response.content,
            documents={f"[{doc['id']}]": doc["content"] for doc in similar_documents},
        )
