"""
Sources:

References:

Synopsis:

Notes:

"""

from __future__ import annotations
from loguru import logger

from langchain_chroma import Chroma
from pathlib import Path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStore
    from langchain_core.embeddings import Embeddings

class ChromaVectorStoreBuilder:
    @staticmethod
    def load_vector_store(
        persist_directory: str | Path,
        embedding_function: Embeddings,
    ) -> VectorStore:
        # 路径处理。
        persist_directory = Path(persist_directory)
        if persist_directory.exists():
            logger.warning(f"Persist directory {persist_directory} already exists.")
        vector_store = Chroma(
            persist_directory=str(persist_directory),
            embedding_function=embedding_function,
        )
        return vector_store

    @staticmethod
    def build_new_vector_store(
        persist_directory: str,
        embedding_function: Embeddings,
    ) -> VectorStore:
        ...


