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
def make_test_documents():
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


# HACK: 待整合进fixtures。
def make_test_embedding_model():
    # HARDCODED
    embedding_model = EmbeddingModelFactory.create_ollama_embedding_model(
        model_name="nomic-embed-text",
        num_ctx=None,
        repeat_penalty=None,
        temperature=None,
        stop_tokens=None,
        top_k=None,
        top_p=None,
    )
    return embedding_model


class TestChromaVectorStoreBuilder:
    @pytest.mark.parametrize(
        "persist_directory, embedding_function", [
        (r"D:\dataset\smart\tests\t_vector_store",
         make_test_embedding_model(),),
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
        logger.success(f"\nVector store loaded: \n{vector_store}\n")

    @pytest.mark.parametrize(
        "persist_directory, embedding_function", [
        (r"D:\dataset\smart\tests\t_vector_store",
         make_test_embedding_model(),),
    ])
    def test_vector_store_methods(
        self,
        persist_directory: str,
        embedding_function: Embeddings,
    ):
        vector_store = ChromaVectorStoreBuilder.load_vector_store(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
        )
        # search
        results_1 = vector_store.search(
            query="haha",
            search_type='similarity',
        )
        logger.debug(f"\nResults 1: \n{results_1}\n")
        results_2 = vector_store.search(
            query="haha",
            search_type='mmr',
        )
        logger.debug(f"\nResults 2: \n{results_2}\n")
        # similarity_search
        results_3 = vector_store.similarity_search(
            query="haha",
            k=3,
        )
        logger.debug(f"\nResults 3: \n{results_3}\n")
        # similarity_search_with_relevance_scores
        results_4 = vector_store.similarity_search_with_relevance_scores(
            query="haha",
            k=3,
        )
        logger.debug(f"\nResults 4: {results_4}\n")
        # mmr
        results_5 = vector_store.max_marginal_relevance_search(
            query="haha",
            k=3,
        )
        logger.debug(f"\nResults 5: {results_5}\n")
        # vector_store.max_marginal_relevance_search_by_vector()
        # similarity_search_with_score
        ## bad methods. not include any args
        results_6 = vector_store.similarity_search_with_score(
            query="haha",
            k=3,
        )
        logger.debug(f"\nResults 6: {results_6}\n")

    @pytest.mark.parametrize(
        "persist_directory, embedding_function, documents", [
        (r"D:\dataset\smart\tests\t_vector_store",
         make_test_embedding_model(),
         make_test_documents(),),
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
        logger.info(f"\nVector store built: \n{vector_store}\n")

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

