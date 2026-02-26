"""
Tests for simple retriever.
"""

from __future__ import annotations
import pytest
from loguru import logger

from langchain_storing.qdrant_client_builder import QdrantClientBuilder
from langgraph_retrievers.qdrant_retrievers.simple_retriever import QdrantSimpleRetriever
from langchain_embedding.qdrant_embedding_models.bge_m3_embedding_model import BGEM3EmbeddingModel

from qdrant_client import QdrantClient

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_embedding.qdrant_embedding_interfaces.qdrant_text_embedding_model_interface import (
        QdrantTextEmbeddingModelInterface,
    )


def _get_client() -> QdrantClient:
    client = QdrantClientBuilder.load_client_from_disk(
        path=r"D:\dataset\smart\data_pipeline_cache\vector_store\000004",
        collection_name='default',
    )
    return client


def _get_qdrant_embedding_model() -> BGEM3EmbeddingModel:
    embedding_model = BGEM3EmbeddingModel(
        model_name_or_path=r"D:\model\BAAI\bge-m3",
        batch_size=1,
    )
    return embedding_model


def _get_search_config() -> dict:
    search_config = dict(
        collection_name='default',
        limit=3,
        dense_limit=3,
        sparse_limit=3,
        multi_vector_limit=3,
    )
    return search_config


def _get_qdrant_simple_retriever() -> QdrantSimpleRetriever:
    qdrant_simple_retriever = QdrantSimpleRetriever(
        client=_get_client(),
        qdrant_embedding_model=_get_qdrant_embedding_model(),
        search_config=_get_search_config(),
    )
    return qdrant_simple_retriever


class TestQdrantSimpleRetriever:
    @pytest.mark.parametrize(
        "simple_retriever", [
        _get_qdrant_simple_retriever(),
    ])
    def test_search_documents_via_dense_retrieval(
        self,
        simple_retriever: QdrantSimpleRetriever,
    ) -> None:
        documents = simple_retriever.search_documents_via_dense_retrieval(
            query="haha",
            search_configs=_get_search_config(),
        )
        logger.info(f"Documents Length: {len(documents)}")
        logger.info(f"Documents: {documents}")

    @pytest.mark.parametrize(
        "simple_retriever", [
        _get_qdrant_simple_retriever(),
    ])
    def test_search_documents_via_sparse_retrieval(
        self,
        simple_retriever: QdrantSimpleRetriever,
    ) -> None:
        documents = simple_retriever.search_documents_via_sparse_retrieval(
            query="haha",
            search_configs=_get_search_config(),
        )
        logger.info(f"Documents Length: {len(documents)}")
        logger.info(f"Documents: {documents}")

    @pytest.mark.parametrize(
        "simple_retriever", [
        _get_qdrant_simple_retriever(),
    ])
    def test_search_documents_via_multi_vector(
        self,
        simple_retriever: QdrantSimpleRetriever,
    ) -> None:
        documents = simple_retriever.search_documents_via_multi_vector(
            query="haha",
            search_configs=_get_search_config(),
        )
        logger.info(f"Documents Length: {len(documents)}")
        logger.info(f"Documents: {documents}")

    @pytest.mark.parametrize(
        "simple_retriever", [
        _get_qdrant_simple_retriever(),
    ])
    def test_hybrid_search_documents(
        self,
        simple_retriever: QdrantSimpleRetriever,
    ) -> None:
        documents = simple_retriever.hybrid_search_documents(
            query="haha",
            search_configs=_get_search_config(),
        )
        logger.info(f"Documents Length: {len(documents)}")
        logger.info(f"Documents: {documents}")

    @pytest.mark.parametrize(
        "simple_retriever", [
        _get_qdrant_simple_retriever(),
    ])
    def test_all_search_documents(
        self,
        simple_retriever: QdrantSimpleRetriever,
    ) -> None:
        documents = simple_retriever.all_search_documents(
            query="haha",
            search_configs=_get_search_config(),
        )
        logger.info(f"Documents Length: {len(documents)}")
        logger.info(f"Documents: {documents}")

