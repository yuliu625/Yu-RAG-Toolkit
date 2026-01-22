"""
对于multi-query retriever的测试。
"""

from __future__ import annotations
import pytest
from loguru import logger

from langgraph_retrievers.retrievers.multi_query_retriever import (
    MultiQueryRetriever,
    _RewrittenQueries,
    _structured_llm_system_message,
)
from langchain_storing.chroma_vector_store_builder import ChromaVectorStoreBuilder
from langchain_embedding.embedding_model_factory import EmbeddingModelFactory

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStore
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import SystemMessage


def build_test_vector_store():
    vector_store = ChromaVectorStoreBuilder.load_vector_store(
        persist_directory=r"D:\dataset\smart\tests\t_vector_store",
        embedding_function=EmbeddingModelFactory.create_ollama_embedding_model('nomic-embed-text', None, None, None, None, None,),
    )
    return vector_store


def build_test_structured_llm():
    llm = ChatOllama(
        model='qwen2.5:1.5b',
        temperature=0.0,
    )
    structured_llm = llm.with_structured_output(
        schema=_RewrittenQueries,
    ).with_retry(
        stop_after_attempt=3,
    )
    return structured_llm


class TestMultiQueryRetriever:
    @pytest.mark.parametrize(
        "vector_store, search_configs, structured_llm, structured_llm_system_message", [
        (build_test_vector_store(), {}, build_test_structured_llm(), _structured_llm_system_message),
    ])
    @pytest.mark.asyncio
    async def test_multi_query_retriever_rewrite_query(
        self,
        vector_store: VectorStore,
        search_configs: dict,
        structured_llm: BaseChatModel,
        structured_llm_system_message: SystemMessage,
    ):
        multi_query_retriever = MultiQueryRetriever(
            vector_store=vector_store,
            search_configs=search_configs,
            structured_llm=structured_llm,
            structured_llm_system_message=structured_llm_system_message,
        )
        rewritten_queries = await multi_query_retriever.rewrite_query(
            original_query="量子计算是什么？",
        )
        logger.info(f"\nRewritten Queries: \n{rewritten_queries}\n")

    @pytest.mark.parametrize(
        "vector_store, search_configs, structured_llm, structured_llm_system_message", [
        (build_test_vector_store(), {}, build_test_structured_llm(), _structured_llm_system_message),
    ])
    @pytest.mark.asyncio
    async def test_multi_query_retriever_parallel_search(
        self,
        vector_store: VectorStore,
        search_configs: dict,
        structured_llm: BaseChatModel,
        structured_llm_system_message: SystemMessage,
    ):
        multi_query_retriever = MultiQueryRetriever(
            vector_store=vector_store,
            search_configs=search_configs,
            structured_llm=structured_llm,
            structured_llm_system_message=structured_llm_system_message,
        )
        result_documents = await multi_query_retriever.parallel_search_documents_by_mmr(
            query="公司当前的风险因素在哪里？",
            search_configs={},
        )
        logger.info(f"\nDocument Number: {len(result_documents)}")
        logger.info(f"\nResult Documents: \n{result_documents}\n")

