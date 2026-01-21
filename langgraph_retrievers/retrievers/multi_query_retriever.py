"""
Sources:

References:

Synopsis:
    基于langgraph自实现的multi_query_retriever。

Notes:

"""

from __future__ import annotations
from loguru import logger

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langchain_core.vectorstores import VectorStore
    from langchain_core.documents import Document
    from langchain_core.language_models import BaseChatModel
    from pydantic import BaseModel


class MultiQueryRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        structured_llm: BaseChatModel,
    ):
        self._vector_store = vector_store
        self._structured_llm = structured_llm

    async def process_state(
        self,
        state,
        config: RunnableConfig,
    ) -> dict:
        raise NotImplementedError

    def search_document(
        self,
        query: str,
    ) -> list[Document]:
        ...

    async def rewrite_query(
        self,
        original_query: str,
    ) -> list[str]:
        ...

