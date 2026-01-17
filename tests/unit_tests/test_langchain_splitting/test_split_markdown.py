"""
测试分割markdown文本的方法。
"""

from __future__ import annotations
import pytest
from loguru import logger

from langchain_loading.load_text import TextLoadingMethods
from langchain_splitting.split_markdown import MarkdownSplittingMethods

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class TestMarkdownSplittingMethods:
    @pytest.mark.parametrize()
    def test_split_markdown_by_header(
        self,
        markdown_file_path: str,
    ):
        # 该方法常见约定指定的参数。
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
        ]
        is_strip_headers: bool = False
        target_document = TextLoadingMethods.load_text(
            text_path=markdown_file_path,
            encoding='utf-8',
            is_autodetect_encoding=False,
        )
        result_documents = MarkdownSplittingMethods.split_markdown_by_header(
            document=target_document,
            headers_to_split_on=headers_to_split_on,
            is_strip_headers=is_strip_headers,
        )

