"""
Sources:

References:

Synopsis:
    langchain_chroma 源代码中为多模态需要实现的方法。

Notes:
    需要:
    ```python
    from langchain_chroma import Chroma
    ```

    chroma 是我目前主要使用的本地嵌入式数据库，但是 langchain 中的 Embeddings 这个 interface 暂未支持多模态，而 chroma 已经有少量支持。
    chroma 默认将图片以 base64 进行编码，以字符串进行存储。
"""

from __future__ import annotations
from abc import abstractmethod

from langchain_core.embeddings import Embeddings

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class ChromaMultiModalEmbeddingModel(Embeddings):
    """
    chroma 可使用的方法。

    这里的实现其实可以更加复杂，但是因为 chroma 内将一些方法写死，这个类只是为了兼容。

    使用:
        - 需要实现 Embeddings 中本身要求的方法。这里默认 embed_documents 和 embed_query 使用 embed_text ，但完全可以改写。
        - 实例化 chroma 时，需要将这个 embeddings 的实例以 kwargs 传递给 embedding_function 。
        - 实际构建 vector-store 时，最好使用 embed_text 和 embed_image 。
    注意:
        - 查询存在缺陷，仅支持 text 。
        - 不支持混合检索，多模态需要多个 vector-store 。
    改进:
        - 更一般的方法。将 embed_text 方法以其他模态 embedding-model 实现，例如传入已通过 base64 编码的 str ，后续编码和获取结果以 metadate 实现。
    """

    # ====Abstract method of langchain_core::Embeddings.====
    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        对 documents 进行编码。

        默认实现认为 documents 存储的是字符串，调用批量编码文本的方法。

        可进行的修改:
            - 对于单独的其他模态，存储内容为以 str 代表的 uri 。加入 IO 操作，实现多模态编码。
        """
        return self.embed_text(texts)

    # ====Abstract method of langchain_core::Embeddings.====
    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        对 query 进行编码。

        常见约定 query 为文本。
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
        批量编码 batch 的文本。
        """
        return [[0.0] for _ in range(len(inputs))]

    # ====Abstract method of langchain_chroma::Chroma.====
    @abstractmethod
    def embed_image(
        self,
        uris: list[str],
    ) -> list[list[float]]:
        """
        批量编码 batch 的图像。

        chroma 内部的实现，使用 Image 通过 uri 读取图像，会将编码结果转换为 base64 的字符串。

        该解决方案的特性:
            - 优点:
                - 文档存储和向量存储在一起，没有连接，查询结果直接使用 base64 可直接得到原始图片。
            - 缺点:
                - 死板。其实有其他可选解决方案。
        """
        return [[0.0] for _ in range(len(uris))]

