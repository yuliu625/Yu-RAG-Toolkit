"""
vector-store 相关方法的集成。
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from langchain_chroma import Chroma
from pathlib import Path

from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore


class IndexerInterface(ABC):
    """
    一个 indexer 的基础定义。

    约定:
        - vector-store 和 embedding-model 的绑定。
            vector-store 和 embedding-model 完全绑定，因此我定义一个 vector-store 仅使用完全对应的 embedding-model 。
            为了简化，一个 vector-store 仅构建一个 collection 。

    必须要实现的方法:
        - load_embedding_model: vector-store 和 embedding-model 是完全绑定的，第一步就需要实现 embedding-model 的加载。
        - load_vector_store: 最主要的方法，加载 vector-store ，初始化或者进行后续的处理。
    """

    @staticmethod
    @abstractmethod
    def load_embedding_model(*args, **kwargs) -> Embeddings:
        """
        加载 embedding-model 的方法。

        约定:
            - 使用工厂模式构建，传递获得 embedding-model 实例。

        Returns:
            Embeddings: langchain_core.embeddings 定义的 embedding-model 。
        """

    @staticmethod
    @abstractmethod
    def load_vector_store(*args, **kwargs) -> VectorStore:
        """
        加载 vector-store 的方法。根据具体的 vector-store-client 实现。

        约定:
            - 加载 vector-store 需要同时加载 embedding-model ，约定使用使用 load_embedding_model 加载。

        Returns:
            VectorStore: langchain_core.vectorstores 定义的 VectorStore 。
        """

    @staticmethod
    def save_vector_store(self, *args, **kwargs) -> None:
        """
        保存 vector-store 的方法。

        很多工具中，vector-store 在加载之后，所有的修改会自动同步。
        因此，这个方法并不强制实现。
        """
        raise NotImplementedError

    @staticmethod
    def from_document_store(*args, **kwargs) -> VectorStore:
        """
        将一个已有 embedding 的 vector-store ，修改为新的 embedding-model 。

        这里主要指的是冷启动，原始的 vector-store 实际上为 document-store 。
        """
        raise NotImplementedError


class BaseIndexer(IndexerInterface):
    """
    对于 IndexerInterface 的基础实现。

    常用的实现:
        - embedding-model:
            load_embedding_model 通过 factory-pattern 来实现，初始化方法以 strategy-pattern 来指定。
            从而实现灵活绑定 embedding-model-factory ，但仅实现一次方法，通过 embedding_model_config 指定不同的 embedding-model 。
        - vector_store:
            load_vector_store 一定会使用类中已经构建好的 embedding-model 。
            但是为了复用性，这里并不直接和该 embedding-model 绑定。
    """
    def __init__(
        self,
        embedding_model_config: dict,
    ):
        # 必要的，加载 embedding_model 。
        self._embedding_model = self.load_embedding_model(embedding_model_config)

    @staticmethod
    def get_all_documents(
        vector_store: VectorStore,
    ) -> list[Document]:
        """
        获取一个 vector-store 中全部的 document 。

        原本的 chroma 中有相关实现，这里指的是 langchain_chroma 的方法。

        实现方法:
            暂定的实现，相似度查询出所有的 documents 。
            搜索空字符，避免 embedding 维度问题。查找 1e7 数量的 documents 。
            指定 1e7 是因为一般最多保存 1e6 条文档，更多文档的情况会使用其他方法。

        Args:
            vector_store: 需要进行处理的 vector-store ，

        Returns:
            list[Document]: 原始保存进 vector-store 的 documents 。
        """
        # 方法: 以空字符串，查询远超过存储容量的记录，这样就能返回全部的记录。
        documents = vector_store.similarity_search('', k=10**6)
        return documents


class BaseChromaIndexer(BaseIndexer):
    """
    对于 chroma 的向量数据库相关的实现。

    基础的 chroma 有很多功能，这里仅实现 langchain_chroma 相关。

    对于 Chroma 的默认:
        - 存储: 一个 vector-store 仅一个 collection ，
    """
    def __init__(
        self,
        embedding_model_config: dict,
        add_method: Literal['text', 'image']
    ):
        super().__init__(
            embedding_model_config=embedding_model_config,
        )
        self._add_method = add_method

    def generate_from_documents(self):
        ...

    def generate_from_document_store(self):
        document_store = self.get_all_documents_from_document_store()

    # ====必要的实现====
    @staticmethod
    def load_vector_store(
        persist_directory: str,
        embedding_function: Embeddings,
    ) -> VectorStore:
        """
        加载 vector-store 的方法。

        Args:
            persist_directory: (str), chroma 本地持久化的文件夹路径。后续操作会自动同步。
            embedding_function: langchain_core.embeddings 定义的 embedding-model 。可以是 chroma 扩展方法的 embedding_function 。

        Returns:
            VectorStore: langchain_core.vectorstores 定义的 VectorStore 。这里实际为 Chroma 。
        """
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
        )
        return vector_store

    def get_all_documents_from_document_store(
        self,
        document_store_dir: str,
        document_store_embedding_function: Embeddings,
    ) -> list[Document]:
        """
        Chroma中，从 document-store 中获取全部的 documents 的方法。

        Args:
            document_store_dir:
            document_store_embedding_function:

        Returns:
            list[Document]: document-store 中获取全部的 documents 。
        """
        document_store = self.load_vector_store(
            persist_directory=document_store_dir,
            embedding_function=document_store_embedding_function,
        )
        all_documents = self.get_all_documents(vector_store=document_store)
        return all_documents


class DefaultChromaIndexer(BaseChromaIndexer):
    """
    默认的 ChromaIndexer 的实现。

    修改直接从这里修改。
    """
    def __init__(
        self,
        embedding_model_config: dict,
        add_method: Literal['text', 'image']
    ):
        super().__init__(
            embedding_model_config=embedding_model_config,
            add_method=add_method,
        )

    def batch_run(self):
        ...

    def run(
        self,
    ):
        ...

    # ====必要的实现====
    @staticmethod
    def load_embedding_model(
        embedding_model_config: dict
    ) -> Embeddings:
        """通过strategy-pattern加载embedding-model。"""

