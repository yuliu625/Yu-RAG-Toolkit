"""
一个假的embedding-model。

用于测试，以及暂时存储原始documents为document-store。
"""

from langchain_core.embeddings import Embeddings


class YuFakeEmbeddingModel(Embeddings):
    """
    这个类对应chroma需要的embedding_function。

    实际实现:
        - 所有的输入都会编码为长度为1的embedding，具体为[0.0]。
    """
    def __init__(
        self,
    ):
        pass

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_text(texts)

    def embed_query(self, query: str) -> list[float]:
        return self.embed_text([query])[0]

    def embed_text(
        self,
        inputs: list[str]
    ) -> list[list[float]]:
        return [[0.0] for _ in range(len(inputs))]

    def embed_image(
        self,
        uris: list[str],
    ) -> list[list[float]]:
        return [[0.0] for _ in range(len(uris))]

