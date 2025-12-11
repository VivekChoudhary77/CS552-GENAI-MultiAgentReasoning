"""FAISS-based retriever for RAG system."""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pickle
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from src.utils.config import EMBEDDING_MODEL, TOP_K_RETRIEVAL, VECTOR_STORE_DIR
from src.utils.logger import setup_logger

logger = setup_logger()

class FAISSRetriever:
    """FAISS-based retriever for semantic search."""
    
    def __init__(self, embedding_model_name: str = None):
        """Initialize the retriever with embedding model."""
        self.embedding_model_name = embedding_model_name or EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedder = SentenceTransformer(self.embedding_model_name)
        self.index = None
        self.documents = []
        self.metadata = []
        
    def build_index(self, documents: List[str], metadata: List[dict] = None):
        """Build FAISS index from documents."""
        logger.info(f"Building FAISS index for {len(documents)} documents...")
        self.documents = documents
        self.metadata = metadata or [{}] * len(documents)
        
        # Generate embeddings
        embeddings = self.embedder.encode(documents, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        logger.info(f"Index built with {self.index.ntotal} vectors of dimension {dimension}")
        
    def save_index(self, save_path: str = None):
        """Save FAISS index and documents to disk."""
        if save_path is None:
            save_path = os.path.join(VECTOR_STORE_DIR, "faiss_index")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{save_path}.index")
        
        # Save documents and metadata
        with open(f"{save_path}.pkl", "wb") as f:
            pickle.dump({"documents": self.documents, "metadata": self.metadata}, f)
        
        logger.info(f"Index saved to {save_path}")
        
    def load_index(self, load_path: str = None):
        """Load FAISS index and documents from disk."""
        if load_path is None:
            load_path = os.path.join(VECTOR_STORE_DIR, "faiss_index")
        
        if not os.path.exists(f"{load_path}.index"):
            raise FileNotFoundError(f"Index not found at {load_path}.index. Please build the index first.")
        
        # Load FAISS index
        self.index = faiss.read_index(f"{load_path}.index")
        
        # Load documents and metadata
        with open(f"{load_path}.pkl", "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.metadata = data["metadata"]
        
        logger.info(f"Index loaded with {self.index.ntotal} vectors")
        
    def retrieve(self, query: str, k: int = None) -> List[Tuple[str, dict, float]]:
        """Retrieve top-k most similar documents."""
        if self.index is None or len(self.documents) == 0:
            raise ValueError("Index not built or loaded. Call build_index() or load_index() first.")
        
        k = k or TOP_K_RETRIEVAL
        
        # Generate query embedding
        query_embedding = self.embedder.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        # Format results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                results.append((
                    self.documents[idx],
                    self.metadata[idx],
                    float(dist)
                ))
        
        logger.debug(f"Retrieved {len(results)} documents for query: {query[:50]}...")
        return results

