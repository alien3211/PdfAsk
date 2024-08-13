from typing import Self

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate

from pdf_ask.backend.vector_store import VectorStoreProtocol


class Role(Enum):
    USER = "User"
    BOT = "Assistant"


@dataclass
class ChatMessage:
    role: [Role]
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

    def __init__(self, llm, vector_store: VectorStoreProtocol, top_k: int = 3) -> None:
        self.top_k = top_k
        self.llm = llm
        self.vector_store = vector_store
        self.prompt = ChatPromptTemplate.from_template(self.rag_prompt)
        self.chain = self.prompt | self.llm

    def get_response(
        self: Self, question: ChatMessage, history: list[ChatMessage]
    ) -> LlmAnswer:
        similar_documents = self.vector_store.similarity_search(
            question.text, top_k=self.top_k
        )

        response = self.chain.invoke(
            {
                "question": question,
                "context": "\n".join(
                    [
                        f"[{document["id"]}] - {document["content"]}"
                        for document in similar_documents
                    ]
                ),
                "chat_history": [
                    {"role": message.role, "text": message.text} for message in history
                ],
            }
        )
        return LlmAnswer(
            response.content,
            documents={
                f"[{document["id"]}]": document["content"]
                for document in similar_documents
            },
        )
