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


class MultiQueryRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
    ):
        ...

    async def process_state(
        self,
        state,
        config: RunnableConfig,
    ) -> dict:
        ...

