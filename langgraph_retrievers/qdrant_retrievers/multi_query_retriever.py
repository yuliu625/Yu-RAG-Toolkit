"""
Sources:

References:

Synopsis:

Notes:

"""

from __future__ import annotations
from loguru import logger

from qdrant_client import QdrantClient

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_embedding.qdrant_embedding_interfaces.qdrant_text_embedding_model_interface import (
        QdrantTextEmbeddingModelInterface,
    )
    from langchain_core.runnables import RunnableConfig
    from langchain_core.documents import Document


class QdrantMultiQueryRetriever:
    def __init__(
        self,
        client: QdrantClient,
        qdrant_embedding_model: QdrantTextEmbeddingModelInterface,
        search_config: dict,
    ):
        self._client = client
        self._qdrant_embedding_model = qdrant_embedding_model
        self._search_config = search_config

    # ==== 暴露方法。 ====
    async def process_state(
        self,
        state,
        config: RunnableConfig,
    ) -> dict:
        raise NotImplementedError

    def search_documents_via_dense_retrieval(
        self,
    ):
        ...

    def search_documents_via_sparse_retrieval(
        self,
    ):
        ...

    def search_documents_via_late_interactions(
        self,
    ):
        ...

    def hybrid_search_documents(
        self,
    ):
        ...

