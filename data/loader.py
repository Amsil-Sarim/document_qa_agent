# data/loader.py
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from typing import List
from langchain.schema import Document

class DocumentLoader:
    def __init__(self):
        self.loader_mapping = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader
        }
    
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """Load documents from various file formats"""
        documents = []
        for file_path in file_paths:
            ext = "." + file_path.split(".")[-1]
            if ext in self.loader_mapping:
                loader = self.loader_mapping[ext](file_path)
                documents.extend(loader.load())
        return documents
