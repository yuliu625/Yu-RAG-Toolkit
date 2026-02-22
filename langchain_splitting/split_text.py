"""
Sources:
    https://github.com/yuliu625/Yu-RAG-Toolkit/langchain_splitting/split_text.py

References:
    https://docs.langchain.com/oss/python/integrations/splitters

Synopsis:
    分割一般文本的方法。

Notes:
    基于 TextSplitter ， 由正则方法实现。

    约定使用场景:
        - 简单处理: 直接构造 chunks 。
        - 二次处理: 在高级特征处理后，为满足 length 限制，再次进行处理。
"""

from __future__ import annotations
from loguru import logger

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from typing import TYPE_CHECKING, Sequence
if TYPE_CHECKING:
    from langchain_core.documents import Document


class TextSplittingMethods:
    @staticmethod
    def split_documents_recursively(
        documents: Sequence[Document],
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        documents = splitter.split_documents(
            documents=documents,
        )
        return documents

    @staticmethod
    def split_text_recursively(
        document: Document,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[str]:
        logger.warning(r"未完善统一方法。")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        texts = splitter.split_text(
            text=document.page_content,
        )
        return texts

    @staticmethod
    def split_text_by_length(

    ):
        raise NotImplementedError

    @staticmethod
    def split_text_by_token(

    ):
        raise NotImplementedError

