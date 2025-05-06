# config/vectorstore_config.py
class VectorStoreConfig:
    def __init__(self):
        self.vectorstore_type = "faiss"  # or "chroma"
        self.similarity_metric = "cosine"
        self.search_kwargs = {"k": 4}  # Number of docs to retrieve
