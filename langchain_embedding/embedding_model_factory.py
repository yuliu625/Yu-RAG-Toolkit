"""
基于langchain的embedding-model-factory。
"""

from langchain_core.embeddings import Embeddings

from langchain_huggingface import HuggingFaceEmbeddings
import os


class EmbeddingModelFactory:
    def __init__(self):
        ...

    def get_huggingface_embedding_model(
        self,
    ):
        ...

    def get_multi_modal_embedding_model(
        self,
    ):
        ...


class CachedEmbeddingModelFactory:
    ...

