"""
测试加载文本的方法。
"""

from __future__ import annotations
import pytest
from loguru import logger

from langchain_loading.load_text import TextLoadingMethods

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class TestTextLoadingMethods:
    @pytest.mark.parametrize(
        'text_path', [
        (r"D:\dataset\smart\tests\pymupdf_1\000004.md"),
    ])
    def test_load_text(
        self,
        text_path: str,
    ):
        # 该方法常见约定指定的参数。
        encoding: str = 'utf-8'
        is_autodetect_encoding: bool = False
        document = TextLoadingMethods.load_text(
            text_path=text_path,
            encoding=encoding,
            is_autodetect_encoding=is_autodetect_encoding,
        )
        logger.info(f"\nDocument: \n{document}")

