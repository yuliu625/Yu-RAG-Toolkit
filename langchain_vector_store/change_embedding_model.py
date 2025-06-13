"""
必要文件: indexer。
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from .base_indexer import BaseIndexer

from langchain_chroma import Chroma
from pathlib import Path

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
    from langchain_core.documents import Document


class EmbeddingChanger:
    def __init__(self):
        ...

    @staticmethod
    def get_all_documents(
        vector_store: VectorStore,
    ) -> list[Document]:
        """
        获取一个vector-store中全部的document。

        原本的chroma中有相关实现，这里指的是langchain_chroma的方法。

        实现方法:
            暂定的实现，相似度查询出所有的documents。
            搜索空字符，避免embedding维度问题。查找1e7数量的documents。
            指定1e7是因为一般最多保存1e6条文档，更多文档的情况会使用其他方法。

        Args:
            vector_store: 需要进行处理的vector-store，

        Returns:
            list[Document], 原始保存进vector-store的documents。
        """
        # 方法: 以空字符串，查询远超过存储容量的记录，这样就能返回全部的记录。
        documents = vector_store.similarity_search('', k=10**6)
        return documents

