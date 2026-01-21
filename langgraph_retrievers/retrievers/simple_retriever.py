"""
Sources:

References:

Synopsis:

Notes:

"""

from __future__ import annotations
from loguru import logger

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langchain_core.vectorstores import VectorStore
    from langchain_core.documents import Document


class SimpleRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
    ):
        self._vector_store = vector_store

    async def process_state(
        self,
        state,
        config: RunnableConfig,
    ) -> dict:
        raise NotImplementedError

    def search_documents(
        self,
        query: str,
    ) -> list[Document]:
        ...

