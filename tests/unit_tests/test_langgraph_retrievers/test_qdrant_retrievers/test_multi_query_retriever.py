"""
Tests for multi-query retriever.
"""

from __future__ import annotations
import pytest
from langchain_core.language_models import BaseChatModel
from loguru import logger

from langchain_storing.qdrant_client_builder import QdrantClientBuilder
from langgraph_retrievers.qdrant_retrievers.multi_query_retriever import (
    QdrantMultiQueryRetriever,
    _RewrittenQueries,
    _structured_llm_system_message,
)
from langchain_embedding.qdrant_embedding_models.bge_m3_embedding_model import BGEM3EmbeddingModel

from qdrant_client import QdrantClient
from langchain_ollama import ChatOllama
from copy import deepcopy

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_embedding.qdrant_embedding_interfaces.qdrant_text_embedding_model_interface import (
        QdrantTextEmbeddingModelInterface,
    )


@pytest.fixture(name='client')
def build_qdrant_client() -> QdrantClient:
    client = QdrantClientBuilder.load_client_from_disk(
        path=r"D:\dataset\smart\data_pipeline_cache\vector_store\000004",
        collection_name='default',
    )
    return client


@pytest.fixture(name='qdrant_embedding_model')
def build_qdrant_embedding_model() -> QdrantTextEmbeddingModelInterface:
    embedding_model = BGEM3EmbeddingModel(
        model_name_or_path=r"D:\model\BAAI\bge-m3",
        batch_size=1,
    )
    return embedding_model


@pytest.fixture(name='structured_llm')
def build_test_structured_llm():
    llm = ChatOllama(
        model='qwen2.5:0.5b',
        temperature=0.0,
    )
    structured_llm = llm.with_structured_output(
        schema=_RewrittenQueries,
    ).with_retry(
        stop_after_attempt=3,
    )
    return structured_llm


@pytest.fixture(name='search_configs')
def build_test_search_configs() -> dict:
    search_configs = dict(
        collection_name='default',
        limit=3,
        dense_limit=3,
        sparse_limit=3,
        multi_vector_limit=3,
    )
    return search_configs


@pytest.fixture(name='multi_query_retriever')
def build_qdrant_multi_query_retriever(
    client: QdrantClient,
    qdrant_embedding_model: QdrantTextEmbeddingModelInterface,
    search_configs: dict,
    structured_llm: BaseChatModel,
) -> QdrantMultiQueryRetriever:
    multi_query_retriever = QdrantMultiQueryRetriever(
        client=client,
        qdrant_embedding_model=qdrant_embedding_model,
        search_configs=search_configs,
        structured_llm=structured_llm,
        structured_llm_system_message=_structured_llm_system_message,
    )
    return multi_query_retriever


@pytest.mark.asyncio
async def test_search_documents_via_dense_retrieval(
    multi_query_retriever: QdrantMultiQueryRetriever,
    search_configs: dict,
) -> None:
    documents = await multi_query_retriever.parallel_search_documents(
        query="haha",
        search_configs=search_configs,
        search_method='dense',
    )
    logger.info(f"Documents Length: {len(documents)}")
    logger.info(f"Documents: {documents}")


@pytest.mark.asyncio
async def test_search_documents_via_sparse_retrieval(
    multi_query_retriever: QdrantMultiQueryRetriever,
    search_configs: dict,
) -> None:
    documents = await multi_query_retriever.parallel_search_documents(
        query="haha",
        search_configs=search_configs,
        search_method='sparse',
    )
    logger.info(f"Documents Length: {len(documents)}")
    logger.info(f"Documents: {documents}")


@pytest.mark.asyncio
async def test_search_documents_via_multi_vector(
    multi_query_retriever: QdrantMultiQueryRetriever,
    search_configs: dict,
) -> None:
    documents = await multi_query_retriever.parallel_search_documents(
        query="haha",
        search_configs=search_configs,
        search_method='multi_vector',
    )
    logger.info(f"Documents Length: {len(documents)}")
    logger.info(f"Documents: {documents}")


@pytest.mark.asyncio
async def test_hybrid_search_documents(
    multi_query_retriever: QdrantMultiQueryRetriever,
    search_configs: dict,
) -> None:
    documents = await multi_query_retriever.parallel_search_documents(
        query="haha",
        search_configs=search_configs,
        search_method='hybrid',
    )
    logger.info(f"Documents Length: {len(documents)}")
    logger.info(f"Documents: {documents}")


@pytest.mark.asyncio
async def test_all_search_documents(
    multi_query_retriever: QdrantMultiQueryRetriever,
    search_configs: dict,
) -> None:
    documents = await multi_query_retriever.parallel_search_documents(
        query="haha",
        search_configs=search_configs,
        search_method='all',
    )
    logger.info(f"Documents Length: {len(documents)}")
    logger.info(f"Documents: {documents}")

