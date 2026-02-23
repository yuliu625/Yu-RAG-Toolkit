"""
Sources:
    https://github.com/yuliu625/Yu-RAG-Toolkit/langchain_storing/qdrant_vector_store_writer.py

References:
    https://qdrant.tech/

Synopsis:
    对于 langchain_qdrant 的强化添加方法。

Notes:
    因为 langchain_qdrant 的添加 points 方法有限，我构造了一个强化的添加方法。
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


class QdrantVectorStoreWriter:
    ...

