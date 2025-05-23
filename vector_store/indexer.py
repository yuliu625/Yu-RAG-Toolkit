"""
vector-store相关方法的集成。
"""

from abc import ABC, abstractmethod

from langchain_chroma import Chroma

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from typing import Optional, Literal


class IndexerInterface(ABC):
    """
    一个indexer的基础定义。

    默认:
        - vector-store和embedding-model的绑定。
            vector-store和embedding-model完全绑定，因此我定义一个vector-store仅使用完全对应的embedding-model。
            为了简化，一个vector-store仅构建一个collection。

    必须要实现的方法:
        - load_embedding_model: vector-store和embedding-model是完全绑定的，第一步就需要实现embedding-model的加载。
        - load_vector_store: 最主要的方法，加载vector-store，初始化或者进行后续的处理。
    """
    @abstractmethod
    def load_embedding_model(self, *args, **kwargs) -> Embeddings:
        """
        加载embedding-model的方法。

        建议使用工厂模式构建，这里直接传递给对应的embedding-model。

        Returns:
            (Embeddings), langchain_core.embeddings定义的embedding-model。
        """

    @abstractmethod
    def load_vector_store(self, *args, **kwargs) -> VectorStore:
        """
        加载vector-store的方法。

        因具体的vector-store-client而异。

        Returns:
            (VectorStore), langchain_core.vectorstores定义的VectorStore。
        """

    def save_vector_store(self, *args, **kwargs) -> None:
        """
        保存vector-store的方法。

        很多工具中，vector-store在加载之后，所有的修改会自动同步。
        因此，这个方法并不强制实现。
        """

    def from_document_store(self, *args, **kwargs) -> VectorStore:
        """
        将一个已有embedding的vector-store，修改为新的embedding-model。

        这里主要指的是冷启动，原始的vector-store实际上为document-store。
        """


class BaseIndexer(IndexerInterface):
    """
    对于IndexerInterface的基础实现。

    常用的实现:
        - embedding-model:
            load_embedding_model通过factory-pattern来实现，初始化方法以strategy-pattern来指定。
            从而实现灵活绑定embedding-model-factory，但仅实现一次方法，通过embedding_model_config指定不同的embedding-model。
        - vector_store:
            load_vector_store一定会使用类中已经构建好的embedding-model。
            但是为了复用性，这里并不直接和该embedding-model绑定。
    """
    def __init__(
        self,
        embedding_model_config: dict,
    ):
        # 必要的，加载embedding_model。
        self._embedding_model = self.load_embedding_model(embedding_model_config)

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


class BaseChromaIndexer(BaseIndexer):
    """
    对于chroma的向量数据库相关的实现。

    基础的chroma有很多功能，这里仅实现langchain_chroma相关。

    对于Chroma的默认:
        - 存储: 一个vector-store仅一个collection，
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
    def load_vector_store(
        self,
        persist_directory: str,
        embedding_function: Embeddings,
    ) -> VectorStore:
        """
        加载vector-store的方法。

        Args:
            persist_directory: (str), chroma本地持久化的文件夹路径。后续操作会自动同步。
            embedding_function: langchain_core.embeddings定义的embedding-model。可以是chroma扩展方法的embedding_function。

        Returns:
            (VectorStore), langchain_core.vectorstores定义的VectorStore。这里实际为Chroma。
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
        Chroma中，从document-store中获取全部的documents的方法。

        Args:
            document_store_dir:
            document_store_embedding_function:

        Returns:
            (list[Document]), document-store中获取全部的documents。
        """
        document_store = self.load_vector_store(
            persist_directory=document_store_dir,
            embedding_function=document_store_embedding_function,
        )
        all_documents = self.get_all_documents(vector_store=document_store)
        return all_documents


class DefaultChromaIndexer(BaseChromaIndexer):
    """
    默认的ChromaIndexer的实现。

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
    def load_embedding_model(
        self,
        embedding_model_config: dict
    ) -> Embeddings:
        """通过strategy-pattern加载embedding-model。"""

