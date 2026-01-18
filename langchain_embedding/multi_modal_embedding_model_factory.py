"""
Sources:

References:

Synopsis:
    多模态的embedding_model的工厂。

Notes:

"""

from __future__ import annotations
from loguru import logger

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


class MultiModalEmbeddingModelFactory:
    @staticmethod
    def create_multi_modal_embedding_model(

    ) -> Embeddings:
        raise NotImplementedError

