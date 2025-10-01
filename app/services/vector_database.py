"""
vector_database.py
------------------

This module defines the **VectorDatabase** service for the CoSMIC Coaching Writer.
It manages storage, retrieval, and updates of embeddings for documents and ad-hoc
text snippets using a FAISS index.

Key responsibilities:
- Initialize and persist a FAISS vector store on disk.
- Add PDFs, directories of PDFs, and raw text snippets to the database.
- Prevent duplicate/near-duplicate additions by similarity checking.
- Provide similarity search with relevance scoring.
- Maintain a CSV catalogue of ingested sources.
"""

import os, csv, glob, threading, logging
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.utils import DistanceStrategy
from .base import ServiceBase
from ..core.config import settings


class VectorDatabase(ServiceBase):
    """
    FAISS-based vector database for document retrieval.

    Attributes:
        path (str): Directory for storing FAISS index and metadata.
        catalogue_path (str): CSV file tracking ingested sources.
        _lock (threading.Lock): Thread-safe lock for database writes.
        logger (logging.Logger): Logger for database events.
        embedding (HuggingFaceEmbeddings): Embedding model for encoding text.
        store (FAISS): FAISS vector store instance.
        splitter (RecursiveCharacterTextSplitter): Text splitter for document chunking.
    """

    def __init__(self):
        super().__init__()
        self.path = settings.vector_db_path
        os.makedirs(self.path, exist_ok=True)
        self.catalogue_path = os.path.join(self.path, "file_list.csv")
        self._lock = threading.Lock()
        self.logger = logging.getLogger("vector_db")

        # Ensure catalogue file exists
        if not os.path.exists(self.catalogue_path):
            with open(self.catalogue_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(["Source", "Time", "Comment"])

        # Initialize embeddings
        self.embedding = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'device': settings.device},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Initialize or load FAISS index
        if os.path.exists(os.path.join(self.path, 'index.faiss')):
            self.store = FAISS.load_local(self.path, self.embedding, allow_dangerous_deserialization=True)
        else:
            self.store = FAISS.from_texts(["CoSMIC Coaching Writer initialisation"], self.embedding)

        self.store.distance_strategy = DistanceStrategy.COSINE
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    def similarity_search_with_relevance_scores(self, query: str, k: int):
        """
        Perform a similarity search with scores.

        Args:
            query (str): Search query text.
            k (int): Number of top results to retrieve.

        Returns:
            List[Tuple[Document, float]]: Matching documents and their scores.
        """
        return self.store.similarity_search_with_relevance_scores(query, k=k)

    def _append_catalogue(self, source: str):
        """Append a new entry to the catalogue CSV with timestamp."""
        with open(self.catalogue_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([source, datetime.utcnow().isoformat(), ""])

    def add_pdf(self, pdf_path: str):
        """
        Add a PDF document to the vector database.

        Args:
            pdf_path (str): Path to the PDF file.
        """
        if not os.path.exists(pdf_path):
            return
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()

        processed = []
        for p in pages:
            p.page_content = p.page_content.replace('\t', ' ')
            processed += self.splitter.split_documents([p])

        if processed:
            with self._lock:
                self.store.add_documents(processed)
                self.store.save_local(self.path)
            self._append_catalogue(pdf_path)
            self.logger.info("Added %s chunks from %s", len(processed), pdf_path)

    def add_pdf_directory(self, dir_path: str):
        """
        Add all PDFs from a directory to the vector database.

        Args:
            dir_path (str): Path to directory containing PDF files.
        """
        for pdf in glob.glob(os.path.join(dir_path, '*.pdf')):
            self.add_pdf(pdf)

    def add_text(self, text: str):
        """
        Add an ad-hoc text snippet to the vector database,
        skipping duplicates when sufficiently similar text already exists.

        Args:
            text (str): Raw text string.

        Returns:
            int: 0 if added, -1 if skipped.
        """
        if not text.strip():
            return -1

        existing = []
        try:
            existing = self.similarity_search_with_relevance_scores(text, k=1)
        except Exception:
            existing = []

        if existing:
            retrieved, score = existing[0]
            if retrieved.page_content.find(text) > -1 and score >= settings.vector_db_update_threshold:
                return -1

        with self._lock:
            self.store.add_texts([text])
            self.store.save_local(self.path)

        self._append_catalogue(text[:80])
        self.logger.debug("Added ad-hoc text snippet length=%d", len(text))
        return 0
    
    def add_pdf_folder(self, folder_path: str) -> int:
        """
        Ingest all PDF files from a given folder into the vector database.

        Args:
            folder_path (str): Path to the folder containing PDFs.

        Returns:
            int: Number of PDFs successfully added.
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        count = 0
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(folder_path, filename)
                try:
                    self.add_pdf(pdf_path)
                    count += 1
                except Exception as e:
                    print(f"[vector_store] Skipped {filename}: {e}")
        return count


# Singleton instance
vector_db_singleton = VectorDatabase()