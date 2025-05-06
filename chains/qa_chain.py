# chains/qa_chain.py
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from config.model_config import ModelConfig

class QAChain:
    def __init__(self, retriever):
        self.model_config = ModelConfig()
        self.retriever = retriever
        
        # Initialize LLM
        if "gpt" in self.model_config.llm_name:
            self.llm = ChatOpenAI(
                model_name=self.model_config.llm_name,
                temperature=self.model_config.llm_temperature,
                max_tokens=self.model_config.llm_max_tokens
            )
        else:
            self.llm = HuggingFaceHub(
                repo_id=self.model_config.llm_name,
                model_kwargs={
                    "temperature": self.model_config.llm_temperature,
                    "max_length": self.model_config.llm_max_tokens
                }
            )
    
    def get_qa_chain(self):
        """Create QA chain with sources"""
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )
