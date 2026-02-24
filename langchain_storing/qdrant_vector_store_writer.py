"""
Sources:
    https://github.com/yuliu625/Yu-RAG-Toolkit/langchain_storing/qdrant_vector_store_writer.py

References:
    https://qdrant.tech/

Synopsis:
    对于 langchain_qdrant 的强化添加方法。

Notes:
    因为 langchain_qdrant 的添加 points 方法有限，我构造了一个强化的添加方法。

    约定:
        - 全编码: 充分利用 qdrant 的优势，当前方法默认将存储 dense, sparse, multi_vector 全部3种格式。

    优化方法:
        - QdrantClient::upload_points & QdrantClient::upload_collection: 我使用了对于 python SDK 的优化方法。
"""

from __future__ import annotations
from loguru import logger

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    PointStruct,
    Distance,
    VectorParams,
    SparseVectorParams,
    MultiVectorConfig,
)

from typing import TYPE_CHECKING, Mapping, Sequence
if TYPE_CHECKING:
    from langchain_core.documents import Document


class QdrantVectorStoreWriter:
    @staticmethod
    def add_documents(
        client: QdrantClient,
        collection_name: str,
        documents: Sequence[Document],
    ) -> QdrantClient:
        raise NotImplementedError

    @staticmethod
    def add_points(
        client: QdrantClient,
        collection_name: str,
        points: Sequence[PointStruct],
    ) -> QdrantClient:
        client.upload_points(
            collection_name=collection_name,
            points=points,
        )
        return client

    @staticmethod
    def add_collection(
        client: QdrantClient,
        collection_name: str,
        # point_ids: Sequence[int],
        vectors: Sequence[dict],
        payload: Sequence[dict],
    ) -> QdrantClient:
        client.upload_collection(
            collection_name=collection_name,
            # HACK: 不传入ids，qdrant 自行生成随机 UUID 。
            # ids=ids,
            vectors=vectors,
            payload=payload,
        )
        return client

