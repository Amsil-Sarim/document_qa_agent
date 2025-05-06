# config/model_config.py
class ModelConfig:
    def __init__(self):
        # Embedding model
        self.embedding_model = "sentence-transformers/all-mpnet-base-v2"
        self.embedding_device = "cuda"  # or "cpu"
        
        # LLM for QA
        self.llm_name = "gpt-3.5-turbo"  # or "meta-llama/Llama-2-7b-chat-hf"
        self.llm_temperature = 0.3
        self.llm_max_tokens = 1000
        
        # Chunking parameters
        self.chunk_size = 1000
        self.chunk_overlap = 200
