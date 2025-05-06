# retrieval/vectorstore.py
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from config.model_config import ModelConfig
from config.vectorstore_config import VectorStoreConfig
from typing import List
from langchain.schema import Document

class VectorStoreManager:
    def __init__(self):
        self.model_config = ModelConfig()
        self.vector_config = VectorStoreConfig()
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_config.embedding_model,
            model_kwargs={"device": self.model_config.embedding_device}
        )
    
    def create_vectorstore(self, documents: List[Document], save_path: str):
        """Create and save vectorstore from documents"""
        if self.vector_config.vectorstore_type == "faiss":
            vectorstore = FAISS.from_documents(documents, self.embeddings)
            vectorstore.save_local(save_path)
        elif self.vector_config.vectorstore_type == "chroma":
            vectorstore = Chroma.from_documents(
                documents, 
                self.embeddings, 
                persist_directory=save_path
            )
        return vectorstore
    
    def load_vectorstore(self, load_path: str):
        """Load existing vectorstore"""
        if self.vector_config.vectorstore_type == "faiss":
            return FAISS.load_local(load_path, self.embeddings)
        elif self.vector_config.vectorstore_type == "chroma":
            return Chroma(
                persist_directory=load_path,
                embedding_function=self.embeddings
            )
