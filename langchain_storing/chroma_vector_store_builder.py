"""
Sources:
    https://github.com/yuliu625/Yu-RAG-Toolkit/langchain_storing/chroma_vector_store_builder.py

References:
    https://docs.langchain.com/oss/python/integrations/vectorstores

Synopsis:
    基于chroma实现的多模态vector_store的构建ChromaVectorStore方法。

Notes:

"""

from __future__ import annotations
from loguru import logger

from langchain_chroma import Chroma
from pathlib import Path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStore
    from langchain_core.embeddings import Embeddings
    from langchain_core.documents import Document


class ChromaVectorStoreBuilder:
    # ====主要方法。====
    @staticmethod
    def load_vector_store(
        persist_directory: str | Path,
        embedding_function: Embeddings,
    ) -> VectorStore:
        """
        直接从本地加载vector_store，不做任何处理。

        Args:
            persist_directory:
            embedding_function:

        Returns:
            VectorStore:
        """
        # 路径处理。
        persist_directory = Path(persist_directory)
        if persist_directory.exists():
            # 可能会出现问题的情况。已经构建过vector store。
            logger.warning(f"Persist directory {persist_directory} already exists.")
        else:
            # 新构建的vector store。
            logger.info(f"New Vector Store in {persist_directory}.")
        vector_store = Chroma(
            persist_directory=str(persist_directory),
            embedding_function=embedding_function,
        )
        return vector_store

    # ====重要方法。====
    @staticmethod
    def build_new_vector_store_via_default_method(
        persist_directory: str | Path,
        embedding_function: Embeddings,
        documents: list[Document],
    ) -> VectorStore:
        # 路径处理。
        persist_directory = Path(persist_directory)
        if persist_directory.exists():
            # 可能会出现问题的情况。已经构建过vector store。
            logger.warning(f"Persist directory {persist_directory} already exists.")
        else:
            # 新构建的vector store。
            logger.info(f"New Vector Store in {persist_directory}.")
        vector_store = Chroma.from_documents(
            persist_directory=str(persist_directory),
            embedding=embedding_function,
            documents=documents,
        )
        return vector_store

    # ====示例方法。====
    @staticmethod
    def build_new_vector_store_via_specific_method(
        persist_directory: str | Path,
        embedding_function: Embeddings,
        documents: list[Document],
        metadatas: list[dict],
    ) -> VectorStore:
        # 路径处理。
        persist_directory = Path(persist_directory)
        if persist_directory.exists():
            # 可能会出现问题的情况。已经构建过vector store。
            logger.warning(f"Persist directory {persist_directory} already exists.")
        else:
            # 新构建的vector store。
            logger.info(f"New Vector Store in {persist_directory}.")
        vector_store = Chroma(
            persist_directory=str(persist_directory),
            embedding_function=embedding_function,
        )
        # 示例方法。
        ## 具体的，可以以add_texts和add_images实现。
        vector_store.add_documents(
            documents=documents,
            metadatas=metadatas,
        )
        return vector_store

    # ====示例方法。====
    @staticmethod
    def build_new_vector_store_via_texts(
        persist_directory: str | Path,
        embedding_function: Embeddings,
        texts: list[str],
        metadatas: list[dict],
    ) -> VectorStore:
        # 路径处理。
        persist_directory = Path(persist_directory)
        if persist_directory.exists():
            # 可能会出现问题的情况。已经构建过vector store。
            logger.warning(f"Persist directory {persist_directory} already exists.")
        else:
            # 新构建的vector store。
            logger.info(f"New Vector Store in {persist_directory}.")
        vector_store = Chroma(
            persist_directory=str(persist_directory),
            embedding_function=embedding_function,
        )
        # 示例方法。
        ## 具体的，可以以add_texts和add_images实现。
        vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
        )
        return vector_store

    # ====示例方法。====
    @staticmethod
    def build_new_vector_store_via_image_uri(
        persist_directory: str | Path,
        embedding_function: Embeddings,
        image_uris: list[str],
        metadatas: list[dict],
    ) -> VectorStore:
        # 路径处理。
        persist_directory = Path(persist_directory)
        if persist_directory.exists():
            # 可能会出现问题的情况。已经构建过vector store。
            logger.warning(f"Persist directory {persist_directory} already exists.")
        else:
            # 新构建的vector store。
            logger.info(f"New Vector Store in {persist_directory}.")
        vector_store = Chroma(
            persist_directory=str(persist_directory),
            embedding_function=embedding_function,
        )
        # 示例方法。
        ## 具体的，可以以add_texts和add_images实现。
        vector_store.add_images(
            uris=image_uris,
            metadatas=metadatas,
        )
        return vector_store

