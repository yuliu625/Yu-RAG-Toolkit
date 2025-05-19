"""

"""

from abc import ABC, abstractmethod

from langchain_chroma import Chroma

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class IndexerInterface(ABC):
    @abstractmethod
    def load_vector_store(self, *args, **kwargs) -> VectorStore:
        """加载vector-store的方法。"""

    @abstractmethod
    def save_vector_store(self, *args, **kwargs) -> None:
        """保存vector-store的方法"""

    @abstractmethod
    def load_embedding_model(self, *args, **kwargs) -> Embeddings:
        """加载embedding-model的方法。"""

    @abstractmethod
    def change_embedding_model(self, *args, **kwargs) -> VectorStore:
        """将一个已有embedding的vector-store，修改为新的embedding-model。"""


class BaseIndexer(IndexerInterface):
    def __init__(self):
        self._vector_store = self.load_vector_store()
        self._embedding_model = self.load_embedding_model()


class ChromaIndexer(BaseIndexer):
    ...

