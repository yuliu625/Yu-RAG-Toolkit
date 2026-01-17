"""
Sources:

References:

Synopsis:

Notes:

nomic基于VLM实现的visual-document-retrieval模型。

原始模型以及代码来源:
https://huggingface.co/nomic-ai/nomic-embed-multimodal-7b
"""

from __future__ import annotations
from loguru import logger

from langchain_core.embeddings import Embeddings
from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor
import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class ColnomicEmbedMultimodalEmbeddingModel(Embeddings):
    """
    nomic基于VLM实现的visual-document-retrieval模型。

    这个模型是基于VLM的，为了存储和运行速度，后续需要改进device的实现。
    """
    def __init__(
        self,
        model_path: str,
    ):
        self.model = BiQwen2_5.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:3",
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
        self.processor = BiQwen2_5_Processor.from_pretrained(model_path)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_text(texts)

    def embed_query(self, query: str) -> list[float]:
        return self.embed_text([query])[0]

    def embed_text(
        self,
        inputs: list[str]
    ) -> list[list[float]]:
        batch_texts = self.processor.process_queries(inputs).to(self.model.device)
        with torch.no_grad():
            text_embeddings = self.model(**batch_texts)
        text_embeddings = list(torch.unbind(text_embeddings))
        text_embeddings = [text_embedding.tolist() for text_embedding in text_embeddings]
        return text_embeddings

    def embed_image(
        self,
        uris: list[str],
    ) -> list[list[float]]:
        images = [Image.open(uri) for uri in uris]
        batch_images = self.processor.process_images(images).to(self.model.device)
        with torch.no_grad():
            image_embeddings = self.model(**batch_images)
        image_embeddings = list(torch.unbind(image_embeddings))
        image_embeddings = [image_embedding.tolist() for image_embedding in image_embeddings]
        return image_embeddings

