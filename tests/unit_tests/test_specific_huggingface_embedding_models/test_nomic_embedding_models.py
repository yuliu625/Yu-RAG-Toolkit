"""
nomic ai
"""

from __future__ import annotations
import pytest
from loguru import logger

from langchain_embedding.embedding_model_factory import EmbeddingModelFactory

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class TestNomicEmbeddingModels:
    @pytest.mark.parametrize(
        "model_name_or_path, model_kwargs, encode_kwargs, query_encode_kwargs", [
        (r"D:\model\nomic-ai\nomic-embed-text-v1.5",
         dict(trust_remote_code=True,),
         dict(prompt='search_document'),
         dict(prompt='search_query'),),
    ])
    def test_nomic_embed_text_v15(
        self,
        model_name_or_path: str,
        model_kwargs: dict,
        encode_kwargs: dict,
        query_encode_kwargs: dict,
    ):
        embedding_model = EmbeddingModelFactory.create_huggingface_embedding_model(
            model_name_or_path=model_name_or_path,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            query_encode_kwargs=query_encode_kwargs,
            is_multi_process=False,
            cache_folder=None,
            is_show_progress=True,
        )
        logger.info(embedding_model)
        logger.info(f"model_max_length: {embedding_model._client.tokenizer.model_max_length}")
        logger.info(f"max_seq_length: {embedding_model._client.max_seq_length}")


