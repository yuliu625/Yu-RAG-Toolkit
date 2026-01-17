"""
Sources:
    https://github.com/yuliu625/Yu-RAG-Toolkit/langchain_retrievers/langchain_retriever_factory.py

References:
    https://docs.langchain.com/oss/python/integrations/retrievers

Synopsis:
    基于langchain_classic的预构建retriever的构造方法。

Notes:
    这里的方法适用于简单RAG的快速实现。应该仅在原型开发时暂用。

    遗弃与兼容性:
        - 当前各种retriever为兼容langchain方法而保留，但不应在:
            - 复杂系统;
            - 现代应用;
        中使用。
        - 我已经在逐步放弃来自langchain_classic相关的依赖。
        - 复杂的RAG系统，请查看基于langgraph的构建。
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
    from langchain_core.language_models import BaseChatModel
    from langchain_core.prompts import BasePromptTemplate


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
        llm: BaseChatModel,
        instruction_prompt: BasePromptTemplate,
    ) -> BaseRetriever:
        retriever = MultiQueryRetriever.from_llm(
            retriever=original_retriever,
            llm=llm,
            prompt=instruction_prompt,
        )
        return retriever

    @staticmethod
    def create_self_document_retriever(

    ) -> BaseRetriever:
        # retriever = SelfQueryRetriever.from_llm()
        raise NotImplementedError

    """
    文档长度和语义完整性。
    """

    @staticmethod
    def create_parent_document_retriever(

    ) -> BaseRetriever:
        # retriever = ParentDocumentRetriever()
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

