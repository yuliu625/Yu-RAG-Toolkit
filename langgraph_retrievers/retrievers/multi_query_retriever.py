"""
Sources:

References:

Synopsis:
    基于langgraph自实现的multi_query_retriever。

Notes:
    注意:
        deduplicate_documents有3种实现方法，分别为:
            - langchain_classic::multi_query的实现:
                ```python
                def _unique_documents(documents: Sequence[Document]) -> list[Document]:
                    return [doc for i, doc in enumerate(documents) if doc not in documents[:i]]
                ```
                该方法依赖Document的__eq__方法，并且时间复杂度过高。
            - 基于id去重。
            - 基于内容的hash去重。
"""

from __future__ import annotations
from loguru import logger

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from typing import TYPE_CHECKING, cast
if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langchain_core.vectorstores import VectorStore
    from langchain_core.documents import Document
    from langchain_core.language_models import BaseChatModel


class _RewrittenQueries(BaseModel):
    """
    用于存储重写后的查询语句的数据类。
    """
    queries: list[str] = Field(
        description="针对原始问题生成的不同角度的检索查询语句列表。",
        min_length=3, max_length=5,  # 对于输出的数量进行限制。
    )


_structured_llm_system_message = SystemMessage(
    content="你是一个文档检索优化专家。用户的原始问题可能表述不清或过于简洁。"
    "请生成 3 个不同版本的搜索查询，以帮助从向量数据库中召回最相关的文档。"
    "尝试使用不同的措辞、技术术语或侧重点。",
)


class MultiQueryRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        structured_llm: BaseChatModel,
        structured_llm_system_message: SystemMessage,
    ):
        self._vector_store = vector_store
        self._structured_llm = structured_llm
        self._structured_llm_system_message = structured_llm_system_message

    async def process_state(
        self,
        state,
        config: RunnableConfig,
    ) -> dict:
        raise NotImplementedError

    async def parallel_search_documents_by_similarity(
        self,
        query: str,
        search_configs: dict,
    ) -> list[Document]:
        logger.debug(f"Query: {query}")
        rewritten_queries = await self.rewrite_query(
            original_query=query,
        )
        original_result_documents = []
        for rewritten_query in rewritten_queries:
            result_documents = await self.search_documents_by_similarity(
                query=rewritten_query,
                search_configs=search_configs,
            )
            original_result_documents.extend(result_documents)
        unique_result_documents = self.deduplicate_documents(
            documents=original_result_documents,
        )
        return unique_result_documents

    async def parallel_search_documents_by_mmr(
        self,
        query: str,
        search_configs: dict,
    ) -> list[Document]:
        logger.debug(f"Query: {query}")
        rewritten_queries = await self.rewrite_query(
            original_query=query,
        )
        original_result_documents = []
        for rewritten_query in rewritten_queries:
            result_documents = await self.search_documents_by_mmr(
                query=rewritten_query,
                search_configs=search_configs,
            )
            original_result_documents.extend(result_documents)
        unique_result_documents = self.deduplicate_documents(
            documents=original_result_documents,
        )
        return unique_result_documents

    # ==== 主要方法。 ====
    async def search_documents_by_similarity(
        self,
        query: str,
        search_configs: dict,
    ) -> list[Document]:
        logger.debug(f"Query: {query}")
        logger.debug(f"Search Configs: \n{search_configs}")
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
        logger.debug(f"Query: {query}")
        logger.debug(f"Search Configs: \n{search_configs}")
        result_documents = await self._vector_store.amax_marginal_relevance_search(
            query=query,
            **search_configs,
        )
        logger.debug(f"Found {len(result_documents)} documents")
        logger.trace(f"Result Documents: \n{result_documents}\n")
        return result_documents

    # ==== 重要方法。 ====
    async def rewrite_query(
        self,
        original_query: str,
    ) -> list[str]:
        structured_output = await self._structured_llm.ainvoke(
            input=[
                self._structured_llm_system_message,
                HumanMessage(content=original_query),
            ],
        )
        structured_output = cast(BaseModel, structured_output)
        queries = structured_output.queries
        assert isinstance(queries, list)
        assert isinstance(queries[0], str)
        return queries

    # ==== 工具方法。 ====
    def deduplicate_documents(
        self,
        documents: list[Document],
    ) -> list[Document]:
        seen_hashes = set()
        unique_documents = []
        for document in documents:
            content_hash = hash(document.page_content)
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_documents.append(document)
        return unique_documents

    # ==== 工具方法。具体的structured_llm可在外部构造。 ====
    def _build_structured_llm(
        self,
        llm: BaseChatModel,
        schema_pydantic_base_model: type[BaseModel],
        # system_message: SystemMessage,
        max_retries: int = 3,
    ) -> BaseChatModel:
        """
        构造structured_llm的方法。

        这个实现默认:
            - 不对于system-message进行限制，不将system-message与llm提前绑定。
                - 但相关控制需要使用structured_llm的方法实现。

        Args:
            llm (BaseChatModel): 基础的用于推理的基座模型。需要具有结构化提取功能。
            schema_pydantic_base_model (BaseModel): 基于pydantic定义的schema。
            max_retries (int): 最大重试次数。基于runnable本身的实现。

        Returns:
            BaseChatModel: 被限制为仅会进行结构化输出的structured_llm。
        """
        structured_llm = llm.with_structured_output(
            schema=schema_pydantic_base_model,
        ).with_retry(
            stop_after_attempt=max_retries,
        )
        structured_llm = cast('BaseChatModel', structured_llm)
        return structured_llm

