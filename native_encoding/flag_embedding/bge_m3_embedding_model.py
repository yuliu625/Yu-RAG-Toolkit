"""
Sources:
    https://github.com/yuliu625/Yu-RAG-Toolkit/native_encoding/flag_embedding/bge_m3_embedding_model.py

References:
    https://huggingface.co/BAAI/bge-m3

Synopsis:
    基于 FlagEmbedding 的 bge-m3 直接使用方法。

Notes:
    由于 bge-m3 有多种编码和检索方法，使用各种 all-in-one 需要基于 FlagEmbedding 才能使用。

    注意:
        - 以下实现实际上是一个 wrapper ，硬编码了 bge-m3 科研场景的必要参数。
        - 如果有进阶定制需求，应直接使用 BGEM3FlagModel 这个在 transformers 上的直接 wrapper 。
"""

from __future__ import annotations
from loguru import logger

from FlagEmbedding import BGEM3FlagModel

from typing import TYPE_CHECKING, Sequence
# if TYPE_CHECKING:


class BGEM3EmbeddingModel:
    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int,
    ):
        self._model = BGEM3FlagModel(
            model_name_or_path=model_name_or_path,
            # HARDCODED
            use_fp16=True,
        )
        self._batch_size = batch_size
        # HARDCODED
        ## 使用最大 context length 能力。
        self._max_length = 8192
        ## 测试使用，所有的 encode 方法的结果都返回。
        self._return_dense = True
        self._return_sparse = True
        self._return_colbert_vecs = True

    def encode_queries(
        self,
        queries: Sequence[str],
    ) -> dict:
        # encode queries in different methods
        results = self._model.encode_queries(
            queries=list(queries),
            return_dense=self._return_dense,
            return_sparse=self._return_sparse,
            return_colbert_vecs=self._return_colbert_vecs,
        )
        return dict(
            dense=results['dense_vecs'],
            sparse=results['lexical_weights'],
            multi_vector=results['colbert_vecs'],
        )

    def encode_texts(
        self,
        texts: Sequence[str],
    ) -> dict:
        # encode texts in different methods
        results = self._model.encode(
            sentences=list(texts),
            batch_size=self._batch_size,
            max_length=self._max_length,
            return_dense=self._return_dense,
            return_sparse=self._return_sparse,
            return_colbert_vecs=self._return_colbert_vecs,
        )
        return dict(
            dense=results['dense_vecs'],
            sparse=results['lexical_weights'],
            multi_vector=results['colbert_vecs'],
        )

