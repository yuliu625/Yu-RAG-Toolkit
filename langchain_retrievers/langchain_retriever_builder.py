"""
Sources:

References:
    https://docs.langchain.com/oss/python/integrations/retrievers

Synopsis:

Notes:
    以下方法仅供参考。对于复杂的RAG，可以基于langgraph进行构建。
"""

from __future__ import annotations
from loguru import logger

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStore, VectorStoreRetriever


class LangchainRetrieverBuilder:
    """
    构建langchain-v0.3以及langgraph之前推荐的retriever。

    该构造器为兼容性保留，复杂RAG应使用langgraph构建。
    """
    @staticmethod
    def build_langchain_retriever(
        vector_store: VectorStore,
        search_type: Literal[
            'similarity', 'mmr',
        ],
        return_document_number: int,  # k
        metadata_filter: dict,  # filter
        fetch_k: int,
    ) -> VectorStoreRetriever:
        search_kwargs = dict(
            k=return_document_number,
            fetch_k=fetch_k,
            filter=metadata_filter,
        )
        retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )
        return retriever

