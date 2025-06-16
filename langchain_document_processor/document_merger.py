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
    # ====主要方法。====
    @staticmethod
    def get_human_message(
        query: str,
        text_documents: list[Document] | None = None,
        image_documents: list[Document] | None = None,
    ) -> HumanMessage:
        if text_documents is not None and image_documents is None:
            # 仅文本文档的情况。
            ...
        elif text_documents is None and image_documents is not None:
            # 仅图片文档的情况。
            ...
        elif text_documents is not None and image_documents is not None:
            # 文本和图片模态都有的情况。
            ...
        else:
            raise

    @staticmethod
    def merge_multimodal_documents(
        query: str,
        text_documents: list[Document] | None = None,
        image_documents: list[Document] | None = None,
    ) -> list[Document]:
        ...

    @staticmethod
    def merge_text_documents(
        text_documents: list[Document],
    ) -> str:
        text_content = ""
        for text_document in text_documents:
            document_text = ContentAnnotator.annotate_with_xml(
                tag='document',
                original_text=text_document.page_content
            )
            document_text += '\n'
            text_content += document_text
        return text_content

    @staticmethod
    def merge_image_documents(
        image_documents: list[Document],
    ) -> list[dict]:
        image_content_blocks = [
            ContentBlockProcessor.get_image_content_block_from_base64(base64_str=image_document.page_content)
            for image_document in image_documents
        ]
        return image_content_blocks


