"""
Sources:
    https://github.com/yuliu625/Yu-RAG-Toolkit/langchain_splitting/split_text.py

References:
    https://docs.langchain.com/oss/python/integrations/splitters

Synopsis:
    分割一般的text。

Notes:

"""

from __future__ import annotations
from loguru import logger

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.documents import Document


class TextSplittingMethods:
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

