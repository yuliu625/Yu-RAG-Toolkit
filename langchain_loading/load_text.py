"""
Sources:
    https://github.com/yuliu625/Yu-RAG-Toolkit/langchain_loading/load_text.py

References:
    https://docs.langchain.com/oss/python/integrations/document_loaders

Synopsis:
    加载text的方法。

Notes:

"""

from __future__ import annotations
from loguru import logger

from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
)
from pathlib import Path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.document_loaders import BaseLoader


class TextLoadingMethods:
    @staticmethod
    def load_text(
        text_path: str | Path,
        encoding: str,
        is_autodetect_encoding: bool,
    ) -> Document:
        """
        加载不限制文件类型的文本。

        Args:
            text_path:
            encoding:
            is_autodetect_encoding:

        Returns:
            Document:
        """
        loader = TextLoader(
            file_path=text_path,
            encoding=encoding,
            autodetect_encoding=is_autodetect_encoding,
        )
        documents = loader.load()
        logger.trace(f"Document: \n", documents[0])
        # 在默认加载下，应只有一个Document对象。
        assert len(documents) == 1
        return documents[0]

    @staticmethod
    def load_text_by_directory(
        directory_path: str | Path,
    ) -> list[Document]:
        """
        通过文件夹路径通过pattern匹配的方法获取文件并进行加载。

        该方法可选但我没有具体去实现。我不喜欢不可控的过多自动推断和处理。

        Args:
            directory_path:

        Returns:
            list[Document]:
        """
        loader = DirectoryLoader(
            path=directory_path,
        )
        documents = loader.load()
        return documents

