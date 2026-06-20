"""
Sources:
    https://github.com/yuliu625/Yu-RAG-Toolkit/blob/main/src/langchain_retrievers/langchain_retriever_builder.py

References:
    https://docs.langchain.com/oss/python/integrations/retrievers

Synopsis:
    构造 langchain classic下的 retriever 的方法。

Notes:
    基于 as_retriever 方法，构造多为通过可序列参数进行声明。

    langchain_classic.retrievers 存在的问题是:
        - as_retriever 方法文档不清晰。
        - 具体实现的方法多样， BaseRetriever 实现困难。

    以下方法仅供参考。对于复杂的 RAG ，可以基于 langgraph 进行构建。
"""

from __future__ import annotations
from loguru import logger

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStore, VectorStoreRetriever


class LangchainRetrieverBuilder:
    """
    构建 langchain-v0.3 以及 langgraph 之前推荐的 retriever 。

    该构造器为兼容性保留，复杂 RAG 应使用 langgraph 构建。
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

