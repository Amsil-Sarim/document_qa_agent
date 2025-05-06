# agent/qa_agent.py
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from chains.qa_chain import QAChain
from config.model_config import ModelConfig

class QAAgent:
    def __init__(self, qa_chain):
        self.model_config = ModelConfig()
        self.qa_chain = qa_chain
        
        # Define tools
        self.tools = [
            Tool(
                name="Document QA System",
                func=self.qa_chain,
                description="Useful for answering questions about the loaded documents"
            )
        ]
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def get_agent(self):
        """Initialize conversational agent"""
        agent = initialize_agent(
            tools=self.tools,
            llm=self.qa_chain.llm,
            agent="conversational-react-description",
            memory=self.memory,
            verbose=True
        )
        return agent
