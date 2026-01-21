"""
测试基础retriever。
"""

from __future__ import annotations
import pytest
from loguru import logger

from langgraph_retrievers.retrievers.simple_retriever import SimpleRetriever
from langchain_storing.chroma_vector_store_builder import ChromaVectorStoreBuilder
from langchain_embedding.embedding_model_factory import EmbeddingModelFactory

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStore


def build_test_vector_store():
    vector_store = ChromaVectorStoreBuilder.load_vector_store(
        persist_directory=r"D:\dataset\smart\tests\t_vector_store",
        embedding_function=EmbeddingModelFactory.create_ollama_embedding_model('nomic-embed-text', None, None, None, None, None,),
    )
    return vector_store


class TestSimpleRetriever:
    @pytest.mark.parametrize(
        "vector_store", [
        (build_test_vector_store()),
    ])
    @pytest.mark.asyncio
    async def test_simple_retriever(
        self,
        vector_store: VectorStore,
    ):
        simple_retriever = SimpleRetriever(
            vector_store=vector_store,
            search_configs={},
        )
        result_documents_1 = await simple_retriever.search_documents_by_similarity(
            query="haha",
            search_configs={},
        )
        logger.debug(f"Result Documents 1: \n{result_documents_1}\n")
        result_documents_2 = await simple_retriever.search_documents_by_mmr(
            query="haha",
            search_configs={},
        )
        logger.debug(f"Result Documents 2: \n{result_documents_2}\n")

