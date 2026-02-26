"""
Sources:

References:

Synopsis:

Notes:

"""

from __future__ import annotations
from loguru import logger

from langchain_document_processing.document_transformer import DocumentTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Prefetch,
    Fusion,
    FusionQuery,
    SparseVector,
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_embedding.qdrant_embedding_interfaces.qdrant_text_embedding_model_interface import (
        QdrantTextEmbeddingModelInterface,
    )
    from langchain_core.runnables import RunnableConfig
    from langchain_core.documents import Document


class QdrantSimpleRetriever:
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
        query: str,
        search_configs: dict,
    ) -> list[Document]:
        dense_query = self._qdrant_embedding_model.encode_dense_query(
            query=query,
        )
        response = self._client.query_points(
            collection_name=search_configs['collection_name'],
            query=dense_query,
            using='dense',
            limit=search_configs['dense_limit'],
        )
        documents = DocumentTransformer.transform_qdrant_query_response_to_langchain_documents(
            response=response,
        )
        return documents

    def search_documents_via_sparse_retrieval(
        self,
        query: str,
        search_configs: dict,
    ) -> list[Document]:
        sparse_query = self._qdrant_embedding_model.encode_sparse_query(
            query=query,
        )
        response = self._client.query_points(
            collection_name=search_configs['collection_name'],
            query=SparseVector(
                indices=sparse_query.keys(),
                values=sparse_query.values(),
            ),
            using='sparse',
            limit=search_configs['sparse_limit'],
        )
        documents = DocumentTransformer.transform_qdrant_query_response_to_langchain_documents(
            response=response,
        )
        return documents

    def search_documents_via_multi_vector(
        self,
        query: str,
        search_configs: dict,
    ) -> list[Document]:
        multi_vector_query = self._qdrant_embedding_model.encode_multi_vector_query(
            query=query,
        )
        response = self._client.query_points(
            collection_name=search_configs['collection_name'],
            query=multi_vector_query,
            using='multi_vector',
            limit=search_configs['multi_vector_limit'],
        )
        documents = DocumentTransformer.transform_qdrant_query_response_to_langchain_documents(
            response=response,
        )
        return documents

    def hybrid_search_documents(
        self,
        query: str,
        search_configs: dict,
    ) -> list[Document]:
        dense_query = self._qdrant_embedding_model.encode_dense_query(
            query=query,
        )
        sparse_query = self._qdrant_embedding_model.encode_sparse_query(
            query=query,
        )
        response = self._client.query_points(
            collection_name=search_configs['collection_name'],
            prefetch=[
                # dense
                Prefetch(
                    query=dense_query,
                    using='dense',
                    limit=search_configs['dense_limit'],
                ),
                # sparse
                Prefetch(
                    query=SparseVector(
                        indices=sparse_query.keys(),
                        values=sparse_query.values(),
                    ),
                    using='sparse',
                    limit=search_configs['sparse_limit'],
                )
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=search_configs['limit'],
        )
        documents = DocumentTransformer.transform_qdrant_query_response_to_langchain_documents(
            response=response,
        )
        return documents

    def all_search_documents(
        self,
        query: str,
        search_configs: dict,
    ) -> list[Document]:
        dense_query = self._qdrant_embedding_model.encode_dense_query(
            query=query,
        )
        sparse_query = self._qdrant_embedding_model.encode_sparse_query(
            query=query,
        )
        multi_vector_query = self._qdrant_embedding_model.encode_multi_vector_query(
            query=query,
        )
        response = self._client.query_points(
            collection_name=search_configs['collection_name'],
            prefetch=[
                # dense
                Prefetch(
                    query=dense_query,
                    using='dense',
                    limit=search_configs['dense_limit'],
                ),
                # sparse
                Prefetch(
                    query=SparseVector(
                        indices=sparse_query.keys(),
                        values=sparse_query.values(),
                    ),
                    using='sparse',
                    limit=search_configs['sparse_limit'],
                ),
                # late interaction
                Prefetch(
                    query=multi_vector_query,
                    using='multi_vector',
                    limit=search_configs['multi_vector_limit'],
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=search_configs['limit'],
        )
        documents = DocumentTransformer.transform_qdrant_query_response_to_langchain_documents(
            response=response,
        )
        return documents

