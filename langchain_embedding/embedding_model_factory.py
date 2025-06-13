"""
基于langchain的embedding-model-factory。
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from langchain_huggingface import HuggingFaceEmbeddings
import os

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class EmbeddingModelFactory:
    """
    基础的embedding-model-factory。
    """
    def __init__(self):
        ...

    def get_huggingface_embedding_model(
        self,
    ):
        ...

    def get_multi_modal_embedding_model(
        self,
    ):
        ...


class CachedEmbeddingModelFactory:
    """
    带有缓存的embedding-model-factory。

    实现这个类的原因在于，很多时候embedding-model是在本地运行，因此需要使用同一个embedding-model以节省资源。
    """
    ...

