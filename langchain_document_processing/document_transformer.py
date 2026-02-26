"""
Sources:

References:

Synopsis:
    不同 vector store 下，对于 Document 对象的转换方法。

Notes:
    langchain 体系下的 Document 具有局限性，在直接操作 vector store 的情况下，需要转换 documents 以实现和其他工具的兼容。
"""

from __future__ import annotations
from loguru import logger

from langchain_core.documents import Document
from qdrant_client.models import (
    QueryResponse,
)

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class DocumentTransformer:
    @staticmethod
    def transform_qdrant_query_response_to_langchain_documents(
        response: QueryResponse,
    ) -> list[Document]:
        documents = []
        for point in response.points:
            # metadata = point.payload or {}
            metadata = point.payload['metadata']
            # make page_content
            page_content = point.payload['content']
            # make langchain::Document
            document = Document(
                page_content=page_content,
                metadata=metadata,
            )
            documents.append(document)
        return documents

