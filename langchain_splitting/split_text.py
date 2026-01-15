"""
Sources:
    https://github.com/yuliu625/Yu-RAG-Toolkit/langchain_splitting/split_text.py

References:


Synopsis:
    分割一般的text。

Notes:

"""

from __future__ import annotations
from loguru import logger

from langchain_text_splitters import RecursiveCharacterTextSplitter

from typing import TYPE_CHECKING
# if TYPE_CHECKING:
