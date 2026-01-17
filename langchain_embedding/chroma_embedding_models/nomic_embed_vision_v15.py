"""
Sources:

References:

Synopsis:

Notes:

nomic基础的multi-modal-embedding-model。

原始模型以及代码来源:
https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5
这里同样需要额外对应的text-embedding-model:
https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
"""

from __future__ import annotations
from loguru import logger

from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
import torch
import torch.nn.functional as F
from PIL import Image

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class NomicEmbedVisionV15EmbeddingModel(Embeddings):
    """
    nomic基础的multi-modal-embedding-model。

    这里需要2个模型。分别为:
        - nomic-embed-text-v1.5
        - nomic-embed-vision-v1.5
    """
    def __init__(
        self,
        vision_model_path: str,
        text_model_path: str,
    ):
        # image embedding model
        self.processor = AutoImageProcessor.from_pretrained(vision_model_path, trust_remote_code=True)
        self.vision_model = AutoModel.from_pretrained(vision_model_path, trust_remote_code=True)
        # text embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_path, trust_remote_code=True)
        self.text_model = AutoModel.from_pretrained(text_model_path, trust_remote_code=True)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_text(texts)

    def embed_query(self, query: str) -> list[float]:
        return self.embed_text([query])[0]

    def embed_text(
        self,
        inputs: list[str]
    ) -> list[list[float]]:
        def _mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        encoded_input = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.text_model(**encoded_input)
        text_embeddings = _mean_pooling(model_output, encoded_input['attention_mask'])
        text_embeddings = F.layer_norm(text_embeddings, normalized_shape=(text_embeddings.shape[1],))
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        return text_embeddings.tolist()

    def embed_image(
        self,
        uris: list[str],
    ) -> list[list[float]]:
        images = [Image.open(uri) for uri in uris]
        inputs = self.processor(images, return_tensors="pt")
        img_emb = self.vision_model(**inputs).last_hidden_state
        img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)
        return img_embeddings.tolist()

