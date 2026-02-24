"""
Sources:
    https://github.com/yuliu625/Yu-RAG-Toolkit/langchain_storing/qdrant_client_builder.py

References:
    https://qdrant.tech/

Synopsis:
    直接对于 qdrant client 的创建方法。

Notes:
    langchain 中对于 qdrant 操作有限制，直接对 client 的操作方法。

    注意:
        - 创建 collection : 以下方法实际为创建 collection :
            - 自动构建: 如果不存在基础的 vector store ，会自动创建。
            - 存在检查: 如果重复构建，qdrant 内置报错。

    约定:
        - default name: 科研非应用场景，一表一库， collection name 约定为 default 。
"""

from __future__ import annotations
from loguru import logger

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    MultiVectorConfig,
)

from pathlib import Path

from typing import TYPE_CHECKING, Mapping
# if TYPE_CHECKING:


class QdrantClientBuilder:
    """
    对 qdrant client 的操作方法。

    包括:
        - create: 构建空的 collection 。
        - load: 加载已经构建的 collection 。
    """
    @staticmethod
    def create_empty_client_from_disk(
        path: str | Path,
        collection_name: str,
        vectors_config: Mapping[str, VectorParams],
        sparse_vectors_config: Mapping[str, SparseVectorParams],
    ) -> QdrantClient:
        qdrant_client = QdrantClient(
            path=str(path),
        )
        qdrant_client.create_collection(
            collection_name=collection_name,  # 约定: 默认为 default 。
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
        return qdrant_client

    @staticmethod
    def create_empty_client_from_url(
        url: str,
        collection_name: str,
        vectors_config: Mapping[str, VectorParams],
        sparse_vectors_config: Mapping[str, SparseVectorParams],
    ) -> QdrantClient:
        qdrant_client = QdrantClient(
            url=url,
            # port=port,  # url 为字符串，可以直接指定 port 。
        )
        qdrant_client.create_collection(
            collection_name=collection_name,  # 约定: 默认为 default 。
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
        return qdrant_client

    @staticmethod
    def create_empty_client_from_memory(
        collection_name: str,
        vectors_config: Mapping[str, VectorParams],
        sparse_vectors_config: Mapping[str, SparseVectorParams],
    ) -> QdrantClient:
        qdrant_client = QdrantClient(
            location=':memory:',
        )
        qdrant_client.create_collection(
            collection_name=collection_name,  # 约定: 默认为 default 。
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
        return qdrant_client

    @staticmethod
    def load_client_from_disk(
        path: str | Path,
        collection_name: str,
        # vectors_config: Mapping[str, VectorParams],
        # sparse_vectors_config: Mapping[str, SparseVectorParams],
    ) -> QdrantClient:
        qdrant_client = QdrantClient(
            path=str(path),
            collection_name=collection_name,
            # vectors_config=vectors_config,
            # sparse_vectors_config=sparse_vectors_config,
        )
        return qdrant_client

    @staticmethod
    def load_client_from_url(
        url: str,
        collection_name: str,
        # vectors_config: Mapping[str, VectorParams],
        # sparse_vectors_config: Mapping[str, SparseVectorParams],
    ) -> QdrantClient:
        qdrant_client = QdrantClient(
            url=url,
            collection_name=collection_name,
            # vectors_config=vectors_config,
            # sparse_vectors_config=sparse_vectors_config,
        )
        return qdrant_client

    @staticmethod
    def load_client_from_memory(
        collection_name: str,
        # vectors_config: Mapping[str, VectorParams],
        # sparse_vectors_config: Mapping[str, SparseVectorParams],
    ) -> QdrantClient:
        raise NotImplementedError(
            "在内存中只会构建后就释放，不会有加载的方法。"
        )

