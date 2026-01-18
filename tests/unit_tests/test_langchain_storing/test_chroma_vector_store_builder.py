"""
测试构建Chroma vector_store的方法
"""

from __future__ import annotations
import pytest
from loguru import logger

from langchain_storing.chroma_vector_store_builder import ChromaVectorStoreBuilder
from langchain_embedding.embedding_model_factory import EmbeddingModelFactory

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.documents import Document


class TestChromaVectorStoreBuilder:
    @pytest.mark.parametrize(
        "persist_directory, embedding_function", [
        (r"D:\dataset\smart\tests\t_vector_store", EmbeddingModelFactory.create_ollama_embedding_model('nomic-embed-text', None, None, None, None, None,)),
    ])
    def test_load_vector_store(
        self,
        persist_directory: str,
        embedding_function: Embeddings,
    ):
        vector_store = ChromaVectorStoreBuilder.load_vector_store(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
        )

    @pytest.mark.parametrize(
    )
    def test_build_new_vector_store_via_default_method(
        self,
        persist_directory: str,
        embedding_function: Embeddings,
        documents: list[Document],
    ):
        ...

    @pytest.mark.parametrize()
    def test_build_new_vector_store_via_specific_method(
        self,
        persist_directory: str,
        embedding_function: Embeddings,
        documents: list[Document],
        metadatas: list[dict],
    ):
        ...

    @pytest.mark.parametrize()
    def test_build_new_vector_store_via_texts(
        self,
        persist_directory: str,
        embedding_function: Embeddings,
        texts: list[str],
        metadatas: list[dict],
    ):
        ...

    @pytest.mark.parametrize()
    def test_build_new_vector_store_via_image_uri(
        self,
        persist_directory: str,
        embedding_function: Embeddings,
        image_uris: list[str],
        metadatas: list[dict],
    ):
        ...

