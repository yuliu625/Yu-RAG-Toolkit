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
)

from typing import TYPE_CHECKING, Mapping, Sequence
if TYPE_CHECKING:
    from langchain_core.documents import Document


class QdrantPointBuilder:
    ...

