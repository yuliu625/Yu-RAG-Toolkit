"""
测试构建Chroma vector_store的方法
"""

from __future__ import annotations
import pytest
from loguru import logger

from langchain_storing.chroma_vector_store_builder import ChromaVectorStoreBuilder

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class TestChromaVectorStoreBuilder:
    @pytest.mark.parametrize()
    def test_load_vector_store(
        self,
        persist_directory: str
    ):
        vector_store = ChromaVectorStoreBuilder.load_vector_store(
            persist_directory=persist_directory,
        )


