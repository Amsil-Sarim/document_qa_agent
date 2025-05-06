# retrieval/retriever.py
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chat_models import ChatOpenAI
from config.model_config import ModelConfig
from config.vectorstore_config import VectorStoreConfig

class DocumentRetriever:
    def __init__(self, vectorstore):
        self.model_config = ModelConfig()
        self.vector_config = VectorStoreConfig()
        self.vectorstore = vectorstore
    
    def get_retriever(self, use_compression=False):
        """Create document retriever with optional compression"""
        base_retriever = self.vectorstore.as_retriever(
            search_kwargs=self.vector_config.search_kwargs
        )
        
        if use_compression:
            llm = ChatOpenAI(
                model_name=self.model_config.llm_name,
                temperature=self.model_config.llm_temperature
            )
            compressor = LLMChainExtractor.from_llm(llm)
            return ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
        return base_retriever
