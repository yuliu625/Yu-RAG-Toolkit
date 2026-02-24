"""
Sources:
    https://github.com/yuliu625/Yu-RAG-Toolkit/langchain_storing/qdrant_point_builder.py

References:
    https://qdrant.tech/

Synopsis:
    构建 qdrant client 可插入 points 的方法。

Notes:
    因为 langchain_qdrant 的添加 points 方法有限，我构造了一个强化的添加方法。
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
    MultiVectorComparator,
    SparseVector,
)

from typing import TYPE_CHECKING, Mapping, Sequence
if TYPE_CHECKING:
    from langchain_core.documents import Document


class QdrantPointBuilder:
    @staticmethod
    def build_point(
        document: Document,
        point_id: int,
        dense_vector: list[float],
        sparse_vector: SparseVector,
        multi_vector: list[list[float]],
    ) -> PointStruct:
        point = PointStruct(
            id=point_id,
            vector=dict(
                dense=dense_vector,
                sparse=sparse_vector,
                multi_vector=multi_vector,
            ),
            # HACK: 将 document object 序列化。
            payload=dict(
                content=document.page_content,
                metadata=document.metadata,
            ),
        )
        return point

    @staticmethod
    def build_points(
        documents: Sequence[Document],
        # point_ids: Sequence[int],
        dense_vectors: list[list[float]],
        sparse_vectors: list[SparseVector],
        multi_vectors: list[list[list[float]]],
    ) -> list[PointStruct]:
        assert len(documents) == len(dense_vectors)
        assert len(documents) == len(sparse_vectors)
        assert len(documents) == len(multi_vectors)
        points = []
        # HACK: ids以默认自增实现。
        for _i, document in enumerate(documents):
            point = QdrantPointBuilder.build_point(
                document=document,
                point_id=_i,
                dense_vector=dense_vectors[_i],
                sparse_vector=sparse_vectors[_i],
                multi_vector=multi_vectors[_i],
            )
            points.append(point)
        return points

    @staticmethod
    def build_collection(
        documents: Sequence[Document],
        vectors: Sequence[dict],
    ):
        raise NotImplementedError(
            "如果能直接构建好 vectors ，那么直接使用 upload_collection 方法即可，不需要构建 points 。"
        )

    @staticmethod
    def build_payload(
        documents: Sequence[Document],
    ) -> list[dict]:
        result: list[dict] = [
            dict(
                content=document.page_content,
                metadata=document.metadata,
            )
            for document in documents
        ]
        return result

