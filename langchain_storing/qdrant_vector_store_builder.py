"""
Sources:
    https://github.com/yuliu625/Yu-RAG-Toolkit/langchain_storing/qdrant_vector_store_builder.py

References:
    https://qdrant.tech/

Synopsis:
    基于 qdrant 的 vector_store 管理与控制。

Notes:
    Chroma 上位替代，生态对于 embedding 研究支持更好。

    Qdrant 加载方法:
        - langchain: QdrantVectorStore.from_existing_collection 。
        - client: 先构建 client ，然后传递给 QdrantVectorStore 。
"""

from __future__ import annotations
from loguru import logger

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from pathlib import Path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStore
    from langchain_core.embeddings import Embeddings
    from langchain_core.documents import Document


class QdrantVectorStoreBuilder:
    @staticmethod
    def load_vector_store_from_disk(
        path: str | Path,
        collection_name: str,
        embedding_model: Embeddings,
    ) -> QdrantVectorStore:
        # 路径处理
        path = Path(path)
        # 冗余检查。
        if path.exists():
            # 可能会出现问题的情况。已经构建过vector store。
            logger.warning(f"Persist directory {path} already exists.")
        else:
            # 新构建的vector store。
            logger.info(f"New Vector Store in {path}.")
        # 以下方法会自动检查是否存在，如果不存在会报错。
        ## 以下为必要参数，其他设置在创建时被设置。
        vector_store = QdrantVectorStore.from_existing_collection(
            path=str(path),
            collection_name=collection_name,  # 约定: 默认为 default_name 。
            embedding=embedding_model,
        )
        return vector_store

    @staticmethod
    def load_vector_store_from_url(
        url: str,
        collection_name: str,
        vectors_config: VectorParams,
        embedding_model: Embeddings,
    ) -> QdrantVectorStore:
        client = QdrantClient(
            url=url,
        )
        raise NotImplementedError

    @staticmethod
    def load_vector_store_from_memory(
        collection_name: str,
        vectors_config: VectorParams,
        embedding_model: Embeddings,
    ) -> QdrantVectorStore:
        client = QdrantClient(':memory:')
        raise NotImplementedError

    @staticmethod
    def build_new_vector_store_via_documents(
        path: str | Path,
        collection_name: str,
        embedding_model: Embeddings,
        vectors_config: VectorParams,
        documents: list[Document],
    ) -> QdrantVectorStore:
        # 路径处理
        path = Path(path)
        # 冗余检查。
        if path.exists():
            # 可能会出现问题的情况。已经构建过vector store。
            logger.warning(f"Persist directory {path} already exists.")
        else:
            # 新构建的vector store。
            logger.info(f"New Vector Store in {path}.")
        client = QdrantClient(
            path=str(path),
        )
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
        )
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding_model,
        )
        # 示例方法。
        ## 具体的，可以以add_texts实现。
        vector_store.add_documents(
            documents=documents,
        )
        return vector_store

    @staticmethod
    def build_new_vector_store_via_texts(
        path: str | Path,
        collection_name: str,
        embedding_model: Embeddings,
        vectors_config: VectorParams,
        texts: list[str],
        metadatas: list[dict],
    ) -> QdrantVectorStore:
        # 路径处理
        path = Path(path)
        # 冗余检查。
        if path.exists():
            # 可能会出现问题的情况。已经构建过vector store。
            logger.warning(f"Persist directory {path} already exists.")
        else:
            # 新构建的vector store。
            logger.info(f"New Vector Store in {path}.")
        client = QdrantClient(
            path=str(path),
        )
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
        )
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding_model,
        )
        # 示例方法。
        ## 具体的，可以以add_texts实现。
        vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
        )
        return vector_store

