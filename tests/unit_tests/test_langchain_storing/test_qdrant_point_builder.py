"""
Tests for qdrant point builder.
"""

from __future__ import annotations
import pytest
from loguru import logger

from langchain_storing.qdrant_point_builder import (
    QdrantPointBuilder,
)

from qdrant_client.http.models import (
    PointStruct,
    Distance,
    VectorParams,
    SparseVectorParams,
    MultiVectorConfig,
    MultiVectorComparator,
    SparseVector,
)
from langchain_core.documents import Document

from typing import TYPE_CHECKING, Sequence
# if TYPE_CHECKING:


def _get_documents() -> Sequence[Document]:
    documents = [
        Document(
            page_content='document 1',
            metadata=dict(source='source 1'),
        ),
        Document(
            page_content='document 2',
            metadata=dict(source='source 2'),
        )
    ]
    return documents


def _get_point():
    document = Document(
        page_content='document 1',
        metadata=dict(source='source 1'),
    )
    point_id = 1
    dense_vector = [1.0, 2.0]
    sparse_vector = SparseVector(
        indices=[233],
        values=[0.32],
    )
    multi_vector = [
        [0.5, 0.7],
        [0.8, 0.3],
    ]
    return document, point_id, dense_vector, sparse_vector, multi_vector


def _get_points():
    documents = [
        Document(
            page_content='document 1',
            metadata=dict(source='source 1'),
        ),
        Document(
            page_content='document 2',
            metadata=dict(source='source 2'),
        )
    ]
    point_ids = [1, 2]
    dense_vectors = [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
    sparse_vectors = [
        SparseVector(
            indices=[233],
            values=[0.32],
        ),
        SparseVector(
            indices=[233],
            values=[0.32],
        )
    ]
    multi_vectors = [
        [
            [0.5, 0.7],
            [0.8, 0.3],
        ],
        [
            [0.8, 0.3],
            [0.2, 0.1],
        ]
    ]
    return documents, point_ids, dense_vectors, sparse_vectors, multi_vectors



class TestQdrantPointBuilder:
    @pytest.mark.parametrize(
        'documents',
        [_get_documents()],
    )
    def test_build_payload(
        self,
        documents: Sequence[Document],
    ):
        payload = QdrantPointBuilder.build_payload(
            documents=documents,
        )
        logger.info(f"\nPayload: \n{payload}")

    @pytest.mark.parametrize(
        'document, point_id, dense_vector, sparse_vector, multi_vector',
        [_get_point(),],
    )
    def test_build_point(
        self,
        document: Document,
        point_id: int,
        dense_vector: list[float],
        sparse_vector: SparseVector,
        multi_vector: list[list[float]],
    ):
        point = QdrantPointBuilder.build_point(
            document=document,
            point_id=point_id,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            multi_vector=multi_vector,
        )
        logger.info(f"\nPoint: \n{point}")

    @pytest.mark.parametrize(
        'documents, point_ids, dense_vectors, sparse_vectors, multi_vectors',[
            _get_points(),
    ],)
    def test_build_points(
        self,
        documents: Sequence[Document],
        point_ids: Sequence[int],
        dense_vectors: list[list[float]],
        sparse_vectors: list[SparseVector],
        multi_vectors: list[list[list[float]]],
    ):
        points = QdrantPointBuilder.build_points(
            documents=documents,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
            multi_vectors=multi_vectors,
        )
        logger.info(f"\nPoints: \n{points}")

