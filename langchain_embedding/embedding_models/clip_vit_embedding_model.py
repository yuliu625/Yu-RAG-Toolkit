"""
以CLIP实现的multi-modal-embedding-model。

原始模型以及代码来源:
https://huggingface.co/openai/clip-vit-large-patch14
"""

from langchain_core.embeddings import Embeddings

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image


class CLIPVitEmbeddingModel(Embeddings):
    """
    CLIP被广泛使用，transformers中已经有足够的支持。
    """
    def __init__(
        self,
        model_path: str,
    ):
        self.model = CLIPModel.from_pretrained(model_path)
        self.processor = CLIPProcessor.from_pretrained(model_path)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_text(texts)

    def embed_query(self, query: str) -> list[float]:
        return self.embed_text([query])[0]

    def embed_text(
        self,
        inputs: list[str]
    ) -> list[list[float]]:
        inputs = self.processor(text=inputs, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features.tolist()

    def embed_image(
        self,
        uris: list[str],
    ) -> list[list[float]]:
        images = [Image.open(uri).convert("RGB") for uri in uris]
        inputs = self.processor(images=images, return_tensors="pt")
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features.tolist()

