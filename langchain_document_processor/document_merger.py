"""
组合documents的方法。
"""

from __future__ import annotations

from agnostic_utils.content_annotator import ContentAnnotator
from agnostic_utils.content_block_processor import ContentBlockProcessor

from langchain_core.messages import HumanMessage

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.documents import Document


class DocumentMerger:
    @staticmethod
    def get_human_message(
        query: str,
        text_documents: list[Document] | None = None,
        image_documents: list[Document] | None = None,
    ) -> HumanMessage:
        ...

    @staticmethod
    def merge_multimodal_documents(
        text_documents: list[Document] | None = None,
        image_documents: list[Document] | None = None,
    ) -> list[Document]:
        ...

    @staticmethod
    def merge_text_documents(
        text_documents: list[Document],
    ) -> str:
        ...

    @staticmethod
    def merge_image_documents(
        image_documents: list[Document],
    ) -> list[dict]:
        ...


