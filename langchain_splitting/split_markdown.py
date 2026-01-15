"""
Sources:
    https://github.com/yuliu625/Yu-RAG-Toolkit/langchain_splitting/split_markdown.py

References:

Synopsis:
    分割markdown语法构建的文本。

Notes:

"""

from __future__ import annotations
from loguru import logger

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
    ExperimentalMarkdownSyntaxTextSplitter,
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.documents import Document


class MarkdownSplittingMethods:
    @staticmethod
    def split_markdown_by_header(
        document: Document,
        headers_to_split_on: list[tuple[str, str]],
    ) -> list[Document]:
        default_headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
        )
        # splitter需要的输入是str。但在这个实现中，为了pipeline的清晰流程，以Document对象输入。
        documents = splitter.split_text(
            text=document.page_content,
        )
        return documents

    @staticmethod
    def split_markdown_by_text(
        document: Document,
    ) -> list[Document]:
        raise NotImplementedError

    @staticmethod
    def split_markdown_by_syntax(
        document: Document,
    ) -> list[Document]:
        raise NotImplementedError

