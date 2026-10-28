#!/usr/bin/env python
"""
Seed Knowledge Script

This script processes documents from a directory and adds them to the vector store
with their embeddings for retrieval augmented generation (RAG).
"""

import os
import sys
import glob
import json
import uuid
import argparse
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add the project root to the path so we can import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.qdrant_service import QdrantService
from app.utils.embeddings import batch_get_embeddings, cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_documents(
    directory: str,
    file_types: List[str] = None,
    chunk_size: int = 1000,
    overlap: int = 100,
    strategy: str = "recursive",
) -> List[Dict[str, Any]]:
    """Process documents from a directory and prepare them for embedding

    Args:
        directory: Path to directory containing documents
        file_types: List of file extensions to process (default: [".txt", ".md"])

    Returns:
        List of dictionaries with text content and metadata
    """
    if file_types is None:
        file_types = [".txt", ".md"]

    documents = []

    # Create a pattern for each file type
    patterns = [os.path.join(directory, f"*{ft}") for ft in file_types]

    # Get all matching files
    file_paths = []
    for pattern in patterns:
        file_paths.extend(glob.glob(pattern))

    logger.info(f"Found {len(file_paths)} documents in {directory}")

    for file_path in file_paths:
        logger.info(f"Processing {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            # Split into chunks if needed
            if strategy == "semantic":
                chunks = semantic_split_text(content, chunk_size=chunk_size)
            else:
                chunks = split_text(content, chunk_size=chunk_size, overlap=overlap)
            logger.info(f"Split into {len(chunks)} chunks")

            # Create document entries
            for i, chunk in enumerate(chunks):
                document = {
                    "id": str(uuid.uuid4()),
                    "content": chunk,
                    "metadata": {
                        "source": os.path.basename(file_path),
                        "type": os.path.splitext(file_path)[1][1:],  # File extension without the dot
                        "chunk": i,
                        "path": file_path
                    }
                }
                documents.append(document)
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")

    return documents

def split_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks

    Args:
        text: Text to split
        chunk_size: Maximum chunk size in characters
        overlap: Overlap between chunks in characters

    Returns:
        List of text chunks
    """
    chunks = []

    if len(text) <= chunk_size:
        return [text]

    start = 0
    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunks.append(text[start:])
            break

        # Find a good break point (preferably at the end of a sentence)
        break_point = text.rfind(". ", start, end) + 1
        if break_point <= start:
            break_point = text.rfind(" ", start, end)
        if break_point <= start:
            break_point = end

        chunks.append(text[start:break_point])
        start = break_point - overlap

    return chunks

def semantic_split_text(
    text: str,
    chunk_size: int = 1000,
    similarity_threshold: float = 0.72,
) -> List[str]:
    """Split on paragraph boundaries when adjacent meanings diverge."""
    units = [unit.strip() for unit in text.split("\n\n") if unit.strip()]
    if len(units) < 2:
        return split_text(text, chunk_size=chunk_size, overlap=0)
    embeddings = batch_get_embeddings(units)
    if len(embeddings) != len(units):
        return split_text(text, chunk_size=chunk_size, overlap=0)

    chunks: List[str] = []
    current = units[0]
    for index in range(1, len(units)):
        similarity = cosine_similarity(embeddings[index - 1], embeddings[index])
        candidate = f"{current}\n\n{units[index]}"
        if len(candidate) > chunk_size or similarity < similarity_threshold:
            chunks.extend(split_text(current, chunk_size=chunk_size, overlap=0))
            current = units[index]
        else:
            current = candidate
    chunks.extend(split_text(current, chunk_size=chunk_size, overlap=0))
    return chunks

def main():
    """Main function to seed the knowledge base"""
    # Load environment variables
    load_dotenv()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Seed the knowledge base with documents")
    parser.add_argument("--dir", required=True, help="Directory containing knowledge base documents")
    parser.add_argument("--types", nargs="+", default=[".txt", ".md"],
                        help="File types to process (default: .txt .md)")
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="Maximum chunk size in characters (default: 1000)")
    parser.add_argument("--overlap", type=int, default=100,
                        help="Overlap between chunks in characters (default: 100)")
    parser.add_argument(
        "--strategy",
        choices=["recursive", "semantic"],
        default="recursive",
        help="Chunking strategy (default: recursive)",
    )

    args = parser.parse_args()

    try:
        # Process documents
        documents = process_documents(
            args.dir,
            args.types,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            strategy=args.strategy,
        )
        logger.info(f"Processed {len(documents)} document chunks")

        if not documents:
            logger.warning("No documents found to process")
            return

        # Get embeddings for all documents
        texts = [doc["content"] for doc in documents]
        logger.info("Generating embeddings...")
        embeddings = batch_get_embeddings(texts)
        if len(embeddings) != len(documents):
            raise RuntimeError(
                f"Embedding count mismatch: expected {len(documents)}, got {len(embeddings)}"
            )

        # Add embeddings to documents
        for i, doc in enumerate(documents):
            doc["embedding"] = embeddings[i]

        # Store documents in vector database
        logger.info("Storing documents in vector database...")
        qdrant_service = QdrantService()
        qdrant_service.add_documents(documents)

        logger.info("Knowledge base seeding complete!")

    except Exception as e:
        logger.error(f"Error seeding knowledge base: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
