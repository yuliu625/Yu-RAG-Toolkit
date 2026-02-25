"""
Sources:

References:

Synopsis:
    对于文本在各种编码方法下的标准。

Notes:
    通用性: 所有方法不强制实现。
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from loguru import logger

from typing import TYPE_CHECKING, Sequence
# if TYPE_CHECKING:


class QdrantTextEmbeddingModelInterface:
    def encode_dense_query(
        self,
        query: str,
    ) -> list[float]:
        raise NotImplementedError

    def encode_sparse_query(
        self,
        query: str,
    ) -> dict[int, float]:
        raise NotImplementedError

    def encode_multi_vector_query(
        self,
        query: str,
    ) -> list[list[float]]:
        raise NotImplementedError

    def encode_dense_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        raise NotImplementedError

    def encode_sparse_texts(
        self,
        texts: Sequence[str],
    ) -> list[dict[int, float]]:
        raise NotImplementedError

    def encode_multi_vector_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[list[float]]]:
        raise NotImplementedError

