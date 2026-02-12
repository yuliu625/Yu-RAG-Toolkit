"""
Sources:
    https://github.com/yuliu625/Yu-RAG-Toolkit/langchain_storing/qdrant_vector_store_builder.py

References:
    https://qdrant.tech/

Synopsis:
    基于 qdrant 的 vector_store 管理与控制。

Notes:
    Chroma 上位替代，生态对于 embedding 研究支持更好。
"""

from __future__ import annotations
from loguru import logger

from langchain_qdrant import QdrantVectorStore
from pathlib import Path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStore
    from langchain_core.embeddings import Embeddings
    from langchain_core.documents import Document


class QdrantVectorStoreBuilder:
    @staticmethod
    def load_vector_store(
        path: str | Path,
    ) -> QdrantVectorStore:
        ...

