"""
测试文档合并方法。
"""

from __future__ import annotations
import pytest
from loguru import logger

from langchain_document_processing.document_merger import DocumentMerger

from langchain_core.documents import Document

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.documents import Document


def make_test_text_documents():
    text_documents = [
        Document(page_content="Document content 1"),
        Document(page_content="Document content 2"),
        Document(page_content="Document content 3"),
    ]
    return text_documents


class TestDocumentMerger:
    @pytest.mark.parametrize(
        "text_documents", [
        (make_test_text_documents()),
    ])
    def test_merge_text_documents(
        self,
        text_documents: list[Document],
    ):
        merged_result = DocumentMerger.merge_text_documents(
            text_documents=text_documents,
        )
        logger.info(f"\nMerged Result: \n{merged_result}\n")

