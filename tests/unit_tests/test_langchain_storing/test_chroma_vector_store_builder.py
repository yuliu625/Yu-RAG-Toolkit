"""
测试构建Chroma vector_store的方法
"""

from __future__ import annotations
import pytest
from loguru import logger

from langchain_storing.chroma_vector_store_builder import ChromaVectorStoreBuilder
from langchain_embedding.embedding_model_factory import EmbeddingModelFactory
from langchain_loading.load_text import TextLoadingMethods
from langchain_splitting.split_markdown import MarkdownSplittingMethods

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.documents import Document


# HACK: 待整合进fixtures。
def make_documents():
    # HARDCODED
    document = TextLoadingMethods.load_text(
        text_path=r"D:\dataset\smart\tests\docling_1\000004.md",
        encoding='utf-8',
        is_autodetect_encoding=False,
    )
    # HARDCODED
    documents = MarkdownSplittingMethods.split_markdown_by_header(
        document=document,
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ],
        is_strip_headers=False,
    )
    return documents


class TestChromaVectorStoreBuilder:
    @pytest.mark.parametrize(
        "persist_directory, embedding_function", [
        (r"D:\dataset\smart\tests\t_vector_store",
         EmbeddingModelFactory.create_ollama_embedding_model('nomic-embed-text', None, None, None, None, None,)),
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
        # vector_store.search()
        # vector_store.similarity_search()
        # vector_store.similarity_search_with_relevance_scores()
        # vector_store.similarity_search_with_score()

    @pytest.mark.parametrize(
        "persist_directory, embedding_function, documents", [
        (r"D:\dataset\smart\tests\t_vector_store",
         EmbeddingModelFactory.create_ollama_embedding_model('nomic-embed-text', None, None, None, None, None,),
         make_documents(),),
    ])
    def test_build_new_vector_store_via_default_method(
        self,
        persist_directory: str,
        embedding_function: Embeddings,
        documents: list[Document],
    ):
        logger.trace(f"\nDocuments: \n{documents}\n")
        logger.debug(f"\nType of documents: \n{type(documents)}\n")
        logger.debug(f"\n Type of document: {type(documents[0])}\n")
        vector_store = ChromaVectorStoreBuilder.build_new_vector_store_via_default_method(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            documents=documents,
        )

    # @pytest.mark.parametrize()
    def test_build_new_vector_store_via_specific_method(
        self,
        persist_directory: str,
        embedding_function: Embeddings,
        documents: list[Document],
        metadatas: list[dict],
    ):
        ...

    # @pytest.mark.parametrize()
    def test_build_new_vector_store_via_texts(
        self,
        persist_directory: str,
        embedding_function: Embeddings,
        texts: list[str],
        metadatas: list[dict],
    ):
        ...

    # @pytest.mark.parametrize()
    def test_build_new_vector_store_via_image_uri(
        self,
        persist_directory: str,
        embedding_function: Embeddings,
        image_uris: list[str],
        metadatas: list[dict],
    ):
        ...

