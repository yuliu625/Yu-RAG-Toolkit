"""
Sources:

References:
    https://docs.langchain.com/oss/python/integrations/retrievers

Synopsis:
    基于langgraph构建retriever的protocol。

Notes:
    为广泛的兼容性，基于langgraph的实现依然要求:
        - Input: str
        - Output: list[Document]

    为了灵活的实现选择，我定义了:
        - 基于interface的强制实现要求。
        - 基于protocol的类型标注形式。
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import asyncio
from loguru import logger

from typing import TYPE_CHECKING, Protocol
if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


class RetrieverInterface(ABC):
    @abstractmethod
    async def process_state(
        self,
        state,
        config: RunnableConfig,
    ) -> dict:
        """
        实现retriever的interface。

        该定义与我在通用MAS中的定义保持一致。

        构建规范:
            - 异步分离: 在这层隔离异步操作。如无必要，仅提供同步版本。
            - 操作分离: 直接获取需要更新的状态，不在这里构建过多逻辑。

        Args:
            state (MASState): Graph中定义的state。之后添加类型标注。
            config (RunnableConfig): runnable设计的config配置。可以不使用，但在复杂图中，可以提供更好的控制。

        Returns:
            dict: 表示更新字段的dict。
        """
        raise NotImplementedError("一般MAS中，所有agent的统一的注册方法。")


class RetrieverProtocol(Protocol):
    async def process_state(
        self,
        state,
        config: RunnableConfig,
    ) -> dict:
        """
        实现retriever的protocol。

        该定义与我在通用MAS中的定义保持一致。

        构建规范:
            - 异步分离: 在这层隔离异步操作。如无必要，仅提供同步版本。
            - 操作分离: 直接获取需要更新的状态，不在这里构建过多逻辑。

        Args:
            state (MASState): Graph中定义的state。之后添加类型标注。
            config (RunnableConfig): runnable设计的config配置。可以不使用，但在复杂图中，可以提供更好的控制。

        Returns:
            dict: 表示更新字段的dict。
        """
        raise NotImplementedError("一般MAS中，所有agent的统一的注册方法。")

