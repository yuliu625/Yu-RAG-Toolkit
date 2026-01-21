"""
Sources:

References:

Synopsis:
    基于langgraph实现的基础retriever。

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
        search_configs: dict,
    ):
        self._vector_store = vector_store
        self._search_configs = search_configs

    # ==== 暴露方法。 ====
    async def process_state(
        self,
        state,
        config: RunnableConfig,
    ) -> dict:
        raise NotImplementedError

    # ==== 主要方法。 ====
    async def search_documents_by_similarity(
        self,
        query: str,
        search_configs: dict,
    ) -> list[Document]:
        result_documents = await self._vector_store.asimilarity_search(
            query=query,
            **search_configs,
        )
        logger.debug(f"Found {len(result_documents)} documents")
        logger.trace(f"Result Documents: \n{result_documents}\n")
        return result_documents

    # ==== 主要方法。 ====
    async def search_documents_by_mmr(
        self,
        query: str,
        search_configs: dict,
    ) -> list[Document]:
        result_documents = await self._vector_store.amax_marginal_relevance_search(
            query=query,
            **search_configs,
        )
        logger.debug(f"Found {len(result_documents)} documents")
        logger.trace(f"Result Documents: \n{result_documents}\n")
        return result_documents

