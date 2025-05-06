# Document QA Agent with LangChain

## Overview

This repository implements a question-answering agent over documents using LangChain. The system supports:

- Multiple document formats (PDF, Word, Text, Markdown)
- Two vector store options (FAISS or Chroma)
- Optional retrieval compression
- Conversational agent interface

## Features

- Modular pipeline for document processing and QA
- Support for both OpenAI models and open-source LLMs
- Configurable chunking and retrieval parameters
- Persistent vector store for efficient reloading
- Source citation for answers

## Installation

1. Clone the repository:
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Basic Usage
```bash
python main.py --files documents/sample.pdf --reload
```

