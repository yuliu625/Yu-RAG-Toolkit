"""
Sources:

References:
    https://docs.langchain.com/oss/python/integrations/text_embedding

Synopsis:
    基于langchain的embedding-model-factory。

Notes:

"""

from __future__ import annotations
from loguru import logger

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


class EmbeddingModelFactory:
    """
    基础的embedding-model-factory。

    在目前的langchain体系中，embedding-model是仅基于文本的单模态模型。
    """
    @staticmethod
    def create_ollama_embedding_model(
        model_name: str,
        repeat_penalty: float | None,
        temperature: float | None,
        stop_tokens: list[str] | None,
        top_k: int | None,
        top_p: float | None,
    ) -> Embeddings:
        """
        基于Ollama中的OllamaEmbeddings的embedding-model。

        ollama已经有很好的实现和推理，这里仅指出重要args。

        Args:
            model_name:
            repeat_penalty:
            temperature:
            stop_tokens:
            top_k:
            top_p:

        Returns:
            Embeddings:
        """
        embedding_model = OllamaEmbeddings(
            model=model_name,
            repeat_penalty=repeat_penalty,
            temperature=temperature,
            stop=stop_tokens,
            top_k=top_k,
            top_p=top_p,
        )
        return embedding_model

    @staticmethod
    def create_huggingface_embedding_model(
        model_name_or_path: str,
        model_kwargs: dict,
        encode_kwargs: dict,
        query_encode_kwargs: dict,
        is_multi_process: bool,
        cache_folder: str | None,
        is_show_progress: bool,
    ) -> Embeddings:
        """
        构造基于HuggingFaceEmbeddings的embedding-model。

        Notes:
            - HuggingFaceEmbeddings支持多推理后端，但约定最好使用sentence-transformers。
            - 社区实现基于BaseModel限制，因此我在该方法中显式指定了所有参数。
                如有需要，查看HuggingFaceEmbeddings源代码。

        Args:
            model_name_or_path:
            model_kwargs:
            encode_kwargs:
            query_encode_kwargs:
            is_multi_process (bool): 是否将encode方法在多张GPU上运行。
            cache_folder:
            is_show_progress (bool): 是否显示展示编码进度的progress bar。

        Returns:
            Embeddings:
        """
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name_or_path,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            query_encode_kwargs=query_encode_kwargs,
            multi_process=is_multi_process,
            cache_folder=cache_folder,
            is_show_progress=is_show_progress,
        )
        return embedding_model

    @staticmethod
    def create_openai_embeddings_model(
        model_name: str,
    ) -> Embeddings:
        embedding_model = OpenAIEmbeddings(
            model=model_name,
        )
        raise NotImplementedError


class CachedEmbeddingModelFactory:
    """
    带有缓存的embedding-model-factory。

    实现这个类的原因在于，很多时候embedding-model是在本地运行，因此需要使用同一个embedding-model以节省资源。
    """
    ...

