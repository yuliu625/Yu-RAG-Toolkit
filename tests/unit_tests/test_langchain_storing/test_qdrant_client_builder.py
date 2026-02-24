"""
Tests for qdrant client builder.
"""

from __future__ import annotations
import pytest
from loguru import logger

from langchain_storing.qdrant_client_builder import (
    QdrantClientBuilder,
)
from qdrant_client.http.models import (
    PointStruct,
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    MultiVectorConfig,
    MultiVectorComparator,
)

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class TestQdrantClientBuilder:
    def test_create_empty_client_from_disk(
        self,
    ):
        client = QdrantClientBuilder.create_empty_client_from_disk(
            path="./t_qdrant",
            collection_name='default',
            vectors_config=dict(
                dense=VectorParams(
                    size=1024,
                    distance=Distance.COSINE,
                ),
                multi_vector=VectorParams(
                    size=1024,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM,
                    )
                )
            ),
            sparse_vectors_config=dict(
                sparse=SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False,
                    )
                )
            )
        )
        logger.info(f"\nClient: \n{client}")

    def test_load_client_from_disk(
        self,
    ):
        client = QdrantClientBuilder.load_client_from_disk(
            path="./t_qdrant",
            collection_name='default',
        )
        logger.info(f"\nLoaded Client: \n{client}")
        logger.info(f"\nCollection Config: \n{client.get_collection(collection_name='default').config}")

