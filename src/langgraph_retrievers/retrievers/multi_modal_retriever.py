"""
Sources:

References:

Synopsis:

Notes:

"""

from __future__ import annotations
from loguru import logger

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langchain_core.vectorstores import VectorStore


class MultiModalRetriever:
    def __init__(
        self,
        text_vector_store: VectorStore,
        image_vector_store: VectorStore,
    ):
        self._text_vector_store = text_vector_store
        self._image_vector_store = image_vector_store

    async def process_state(
        self,
        state,
        config: RunnableConfig,
    ) -> dict:
        raise NotImplementedError

