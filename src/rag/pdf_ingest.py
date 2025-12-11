"""PDF ingestion pipeline for converting PDFs to FAISS vectors."""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from typing import List
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.rag.retriever import FAISSRetriever
from src.utils.config import RAW_PDFS_DIR, VECTOR_STORE_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.utils.logger import setup_logger

logger = setup_logger()

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    logger.info(f"Extracting text from {pdf_path}...")
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    logger.info(f"Extracted {len(text)} characters from PDF")
    return text

def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
    """Split text into chunks for embedding."""
    chunk_size = chunk_size or CHUNK_SIZE
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks

def ingest_pdfs(pdf_directory: str = None, output_path: str = None) -> FAISSRetriever:
    """Ingest all PDFs from directory and build FAISS index."""
    pdf_directory = pdf_directory or RAW_PDFS_DIR
    output_path = output_path or os.path.join(VECTOR_STORE_DIR, "faiss_index")
    
    if not os.path.exists(pdf_directory):
        logger.warning(f"PDF directory {pdf_directory} does not exist. Creating it...")
        os.makedirs(pdf_directory, exist_ok=True)
        logger.info(f"Please add PDF files to {pdf_directory} and run again.")
        return None
    
    # Find all PDF files
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_directory}")
        return None
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Extract and chunk all PDFs
    all_chunks = []
    all_metadata = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        try:
            text = extract_text_from_pdf(pdf_path)
            chunks = chunk_text(text)
            
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": pdf_file,
                    "chunk_index": len(all_chunks) - 1
                })
        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {e}")
            continue
    
    if not all_chunks:
        logger.error("No text chunks extracted from PDFs")
        return None
    
    # Build FAISS index
    retriever = FAISSRetriever()
    retriever.build_index(all_chunks, all_metadata)
    retriever.save_index(output_path)
    
    logger.info(f"Successfully ingested {len(pdf_files)} PDFs into FAISS index")
    return retriever

if __name__ == "__main__":
    # Run ingestion when script is executed directly
    retriever = ingest_pdfs()
    if retriever:
        logger.info("PDF ingestion completed successfully!")

