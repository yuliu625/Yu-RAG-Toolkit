"""
测试各种embedding_model。
"""

from __future__ import annotations
import pytest
from loguru import logger

from langchain_embedding.embedding_model_factory import EmbeddingModelFactory

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


class TestEmbeddingModelFactory:
    @pytest.mark.parametrize(
        "model_name, num_ctx, repeat_penalty, temperature, stop_tokens, top_k, top_p", [
        ('nomic-embed-text', None, None, None, None, None, None),
    ])
    def test_create_ollama_embedding_model(
        self,
        model_name: str,
        num_ctx: int | None,
        repeat_penalty: float | None,
        temperature: float | None,
        stop_tokens: list[str] | None,
        top_k: int | None,
        top_p: float | None,
    ):
        embedding_model = EmbeddingModelFactory.create_ollama_embedding_model(
            model_name=model_name,
            num_ctx=num_ctx,
            repeat_penalty=repeat_penalty,
            temperature=temperature,
            stop_tokens=stop_tokens,
            top_k=top_k,
            top_p=top_p,
        )
        text_embedding = embedding_model.embed_query("Some text.")
        logger.info(f"\nText Embedding Dim: {len(text_embedding)}")
        logger.info(f"\nText embedding: \n{text_embedding}")

    @pytest.mark.parametrize(
        "model_name_or_path, model_kwargs, encode_kwargs, query_encode_kwargs, is_multi_process, cache_folder, is_show_progress", [
        (r'', {}, {}, {}, False, None, True),
    ])
    def test_create_huggingface_embedding_model(
        self,
        model_name_or_path: str,
        model_kwargs: dict,
        encode_kwargs: dict,
        query_encode_kwargs: dict,
        is_multi_process: bool,
        cache_folder: str | None,
        is_show_progress: bool,
    ):
        embedding_model = EmbeddingModelFactory.create_huggingface_embedding_model(
            model_name_or_path=model_name_or_path,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            query_encode_kwargs=query_encode_kwargs,
            is_multi_process=is_multi_process,
            cache_folder=cache_folder,
            is_show_progress=is_show_progress,
        )
        text_embedding = embedding_model.embed_query("Some text.")
        logger.info(f"\nText Embedding Dim: {len(text_embedding)}")
        logger.info(f"\nText embedding: \n{text_embedding}")

    # @pytest.mark.parametrize()
    def test_create_openai_embedding_model(
        self,
        model_name: str,
    ):
        ...

