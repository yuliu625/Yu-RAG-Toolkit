"""
Sources:

References:
    https://docs.langchain.com/oss/python/integrations/text_embedding

Synopsis:
    基于langchain的embedding-model-factory。

Notes:

"""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings
import os

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


class EmbeddingModelFactory:
    """
    基础的embedding-model-factory。
    """
    @staticmethod
    def create_huggingface_embedding_model(

    ) -> Embeddings:
        raise NotImplementedError

    @staticmethod
    def create_multi_modal_embedding_model(

    ) -> Embeddings:
        raise NotImplementedError


class CachedEmbeddingModelFactory:
    """
    带有缓存的embedding-model-factory。

    实现这个类的原因在于，很多时候embedding-model是在本地运行，因此需要使用同一个embedding-model以节省资源。
    """
    ...

