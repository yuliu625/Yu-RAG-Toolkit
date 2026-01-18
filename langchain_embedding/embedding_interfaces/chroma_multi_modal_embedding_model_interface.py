"""
Sources:

References:

Synopsis:
    langchain_chroma源代码中为多模态需要实现的方法。

Notes:
    需要:
    ```python
    from langchain_chroma import Chroma
    ```

    chroma是我目前主要使用的本地嵌入式数据库，但是langchain中的Embeddings这个interface暂未支持多模态，而chroma已经有少量支持。
    chroma默认将图片以base64进行编码，以字符串进行存储。
"""

from __future__ import annotations
from abc import abstractmethod

from langchain_core.embeddings import Embeddings

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class ChromaMultiModalEmbeddingModel(Embeddings):
    """
    chroma可使用的方法。

    这里的实现其实可以更加复杂，但是因为chroma内将一些方法写死，这个类只是为了兼容。

    使用:
        - 需要实现Embeddings中本身要求的方法。这里默认embed_documents和embed_query使用embed_text，但完全可以改写。
        - 实例化chroma时，需要将这个embeddings的实例以kwargs传递给embedding_function。
        - 实际构建vector-store时，最好使用embed_text和embed_image。
    注意:
        - 查询存在缺陷，仅支持text。
        - 不支持混合检索，多模态需要多个vector-store。
    改进:
        - 更一般的方法。将embed_text方法以其他模态embedding-model实现，例如传入已通过base64编码的str，后续编码和获取结果以metadate实现。
    """

    # ====Abstract method of langchain_core::Embeddings.====
    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        对documents进行编码。

        默认实现认为documents存储的是字符串，调用批量编码文本的方法。

        可进行的修改:
            - 对于单独的其他模态，存储内容为以str代表的uri。加入IO操作，实现多模态编码。
        """
        return self.embed_text(texts)

    # ====Abstract method of langchain_core::Embeddings.====
    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        对query进行编码。

        常见约定query为文本。
        默认实现调用编码文本方法。
        """
        return self.embed_text([text])[0]

    # ====Abstract method of langchain_chroma::Chroma.====
    @abstractmethod
    def embed_text(
        self,
        inputs: list[str]
    ) -> list[list[float]]:
        """
        批量编码batch的文本。
        """
        return [[0.0] for _ in range(len(inputs))]

    # ====Abstract method of langchain_chroma::Chroma.====
    @abstractmethod
    def embed_image(
        self,
        uris: list[str],
    ) -> list[list[float]]:
        """
        批量编码batch的图像。

        chroma内部的实现，使用Image通过uri读取图像，会将编码结果转换为base64的字符串。

        该解决方案的特性:
            - 优点:
                - 文档存储和向量存储在一起，没有连接，查询结果直接使用base64可直接得到原始图片。
            - 缺点:
                - 死板。其实有其他可选解决方案。
        """
        return [[0.0] for _ in range(len(uris))]

