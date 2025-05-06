# main.py
import argparse
from config.paths import PathConfig
from data.loader import DocumentLoader
from data.processor import DocumentProcessor
from retrieval.vectorstore import VectorStoreManager
from retrieval.retriever import DocumentRetriever
from chains.qa_chain import QAChain
from agent.qa_agent import QAAgent

def parse_args():
    parser = argparse.ArgumentParser(description="Document QA Agent")
    parser.add_argument("--files", nargs="+", required=True, help="Document file paths")
    parser.add_argument("--reload", action="store_true", help="Reload vectorstore")
    parser.add_argument("--compress", action="store_true", help="Use compression retriever")
    return parser.parse_args()

def main():
    args = parse_args()
    path_config = PathConfig()
    
    # Document processing pipeline
    if args.reload:
        print("Loading and processing documents...")
        loader = DocumentLoader()
        processor = DocumentProcessor()
        
        documents = loader.load_documents(args.files)
        processed_docs = processor.process_documents(documents)
        
        vectorstore_manager = VectorStoreManager()
        vectorstore = vectorstore_manager.create_vectorstore(
            processed_docs,
            str(path_config.VECTORSTORE_DIR)
        )
    else:
        print("Loading existing vectorstore...")
        vectorstore_manager = VectorStoreManager()
        vectorstore = vectorstore_manager.load_vectorstore(
            str(path_config.VECTORSTORE_DIR)
        )
    
    # Setup retriever and QA chain
    retriever = DocumentRetriever(vectorstore).get_retriever(args.compress)
    qa_chain = QAChain(retriever).get_qa_chain()
    
    # Initialize agent
    agent = QAAgent(qa_chain).get_agent()
    
    # Start interactive session
    print("\nDocument QA Agent ready. Type 'exit' to quit.")
    while True:
        query = input("\nQuestion: ")
        if query.lower() == 'exit':
            break
        
        result = agent({"input": query})
        print(f"\nAnswer: {result['output']}")
        
        if 'source_documents' in result:
            print("\nSources:")
            for i, doc in enumerate(result['source_documents']):
                print(f"{i+1}. {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})")

if __name__ == "__main__":
    main()
