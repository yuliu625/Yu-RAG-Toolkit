# 🚀 RAG Toolkit
This is a toolkit specifically designed for **RAG (Retrieval-Augmented Generation)** research and engineering. The project systematically deconstructs the entire pipeline—from raw data processing to complex state-machine retrieval—aiming to provide a pluggable and extensible foundation for experimentation.


## 📌 Core Architecture
The repository is modularized according to the RAG lifecycle, ensuring that the logic of each stage is highly decoupled:

| Stage | Module Name | Core Function & Positioning |
| --- | --- | --- |
| **Data Ingestion** | `langchain_loading` | Encapsulates standardized loading methods; supports various unstructured data sources. |
| **Semantic Splitting** | `langchain_splitting` | Diverse document chunking strategies to optimize semantic integrity. |
| **Representation** | `langchain_embedding` | Integration of mainstream Embedding models and multi-modal vectorization schemes. |
| **Vector Storage** | `langchain_storing` | Vector index construction, metadata management, and efficient persistence. |
| **Perception Processing** | `langchain_document_processing` | `Document` object transformation logic, bridging retrieval and generation. |
| **Traditional Retrieval** | `langchain_retrievers` | **[Legacy]** Implementation of basic LangChain native retrievers. |
| **Advanced Retrieval** | `langgraph_retrievers` | **[Active]** Complex RAG strategies implemented via LangGraph state machines. |


## 🛠️ Module Details
### 1. Pipeline Foundations
- **`langchain_loading` & `langchain_splitting`**: Solves standardized preprocessing from multi-source data (PDF, Markdown, Web) into semantic chunks.
- **`langchain_document_processing`**: Specifically handles `Document` objects. This serves as the preprocessing layer after retrieval but before inputting into LLMs & VLMs.

### 2. Representation & Storage
- **`langchain_embedding`**:
  - Covers integration schemes for mainstream text embedding models.
  - **Highlight**: Implements a multi-modal embedding solution based on **ChromaDB**, supporting cross-modal vector retrieval experiments.
- **`langchain_storing`**: Focuses on the engineering implementation of vector databases, including index optimization, persistent storage, and metadata filtering.

### 3. Evolution of Retrieval Strategies
This section records the evolution of retrieval paradigms from static to dynamic:

- **`langchain_retrievers` (Basics)**:
  - Archives native retrieval techniques such as Multi-query and Contextual Compression.
  - *Note: This module is no longer updated and is kept for research comparison only.*
- **`langgraph_retrievers` (Advanced)**:
  - **Core Development Module**. Utilizes `LangGraph` to upgrade retrieval logic from simple Chains to complex state machines.
  - *Note: This module is under active development.*


## 🔗 Related Toolkits
This repository is part of my personal research ecosystem. You can use it in conjunction with the following:
- **[RAG-Toolkit](https://github.com/yuliu625/Yu-RAG-Toolkit)**: Core retrieval-augmented toolset.
- **[Agent-Development-Toolkit](https://github.com/yuliu625/Yu-Agent-Development-Toolkit)**: Focused on building logic for Intelligent Agents.
- **[Deep-Learning-Toolkit](https://github.com/yuliu625/Yu-Deep-Learning-Toolkit)**: A general-purpose foundation for deep learning tasks.
- **[Data-Science-Toolkit](https://github.com/yuliu625/Yu-Data-Science-Toolkit)**: Tools for data science and preprocessing.

