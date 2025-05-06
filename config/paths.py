# config/paths.py
from pathlib import Path

class PathConfig:
    def __init__(self):
        self.BASE_DIR = Path(__file__).parent.parent
        self.DATA_DIR = self.BASE_DIR / "data"
        self.VECTORSTORE_DIR = self.BASE_DIR / "vectorstore"
        
        # Create directories if they don't exist
        self.DATA_DIR.mkdir(exist_ok=True)
        self.VECTORSTORE_DIR.mkdir(exist_ok=True)
