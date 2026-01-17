"""
Sources:

References:
    https://docs.langchain.com/oss/python/integrations/retrievers

Synopsis:

Notes:

"""

from __future__ import annotations
from loguru import logger

from langchain_classic.retrievers import (
    # 查询增强。
    MultiQueryRetriever,
    SelfQueryRetriever,
    # 文档长度和语义完整性。
    ParentDocumentRetriever,
    MultiVectorRetriever,
    # 后处理。
    ContextualCompressionRetriever,
    EnsembleRetriever,
)

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
    from langchain_core.retrievers import BaseRetriever


class LangchainRetrieverFactory:
    @staticmethod
    def create_langchain_retriever(

    ) -> VectorStoreRetriever:
        raise NotImplementedError
        return retriever

    """
    查询增强。
    """

    @staticmethod
    def create_multi_query_retriever(
        original_retriever: VectorStoreRetriever,
    ) -> BaseRetriever:
        retriever = MultiQueryRetriever.from_llm(
            retriever=original_retriever,

        )

    @staticmethod
    def create_self_document_retriever(

    ) -> BaseRetriever:
        raise NotImplementedError

    """
    文档长度和语义完整性。
    """

    @staticmethod
    def create_parent_document_retriever(

    ) -> BaseRetriever:
        raise NotImplementedError

    @staticmethod
    def create_multi_vector_retriever(

    ) -> BaseRetriever:
        raise NotImplementedError

    """
    后处理。
    """

    @staticmethod
    def create_context_compression_retriever(

    ) -> BaseRetriever:
        raise NotImplementedError

    @staticmethod
    def create_ensemble_retriever(

    ) -> BaseRetriever:
        raise NotImplementedError

