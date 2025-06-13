"""
BGE基础的multi-modal-embedding-model。

原始模型以及代码来源:
https://huggingface.co/BAAI/BGE-VL-large
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from transformers import AutoModel
import torch
from PIL import Image

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class BGEVL(Embeddings):
    """
    BGE基础的multi-modal-embedding-model。
    """
    def __init__(
        self,
        model_path: str,
    ):
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.set_processor(model_path)
        self.model.eval()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_text(texts)

    def embed_query(self, query: str) -> list[float]:
        return self.embed_text([query])[0]

    def embed_text(
        self,
        inputs: list[str]
    ) -> list[list[float]]:
        with torch.no_grad():
            text_features = self.model.encode(text=inputs)
        return text_features.tolist()

    def embed_image(
        self,
        uris: list[str],
    ) -> list[list[float]]:
        with torch.no_grad():
            image_features = self.model.encode(images=uris)
        return image_features.tolist()

