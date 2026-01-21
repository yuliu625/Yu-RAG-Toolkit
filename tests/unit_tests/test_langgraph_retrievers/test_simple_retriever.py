"""
测试基础retriever。
"""

from __future__ import annotations
import pytest
from loguru import logger

from langgraph_retrievers.retrievers.simple_retriever import SimpleRetriever
from langchain_storing.chroma_vector_store_builder import ChromaVectorStoreBuilder

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class TestSimpleRetriever:
    ...

