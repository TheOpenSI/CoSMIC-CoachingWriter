import os, csv, glob
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.utils import DistanceStrategy
from .base import ServiceBase
from ..core.config import settings


class VectorDatabase(ServiceBase):
    def __init__(self):
        super().__init__()
        self.path = settings.vector_db_path
        os.makedirs(self.path, exist_ok=True)
        self.catalogue_path = os.path.join(self.path, "file_list.csv")
        if not os.path.exists(self.catalogue_path):
            with open(self.catalogue_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(["Source", "Time", "Comment"])

        self.embedding = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'device': settings.device},
            encode_kwargs={'normalize_embeddings': True}
        )
        if os.path.exists(os.path.join(self.path, 'index.faiss')):
            self.store = FAISS.load_local(self.path, self.embedding, allow_dangerous_deserialization=True)
        else:
            self.store = FAISS.from_texts(["CoSMIC Coaching Writer initialisation"], self.embedding)
        self.store.distance_strategy = DistanceStrategy.COSINE
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    def similarity_search_with_relevance_scores(self, query: str, k: int):
        return self.store.similarity_search_with_relevance_scores(query, k=k)

    def _append_catalogue(self, source: str):
        with open(self.catalogue_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([source, datetime.utcnow().isoformat(), ""])

    def add_pdf(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            return
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        processed = []
        for p in pages:
            p.page_content = p.page_content.replace('\t', ' ')
            processed += self.splitter.split_documents([p])
        if processed:
            self.store.add_documents(processed)
            self.store.save_local(self.path)
            self._append_catalogue(pdf_path)

    def add_pdf_directory(self, dir_path: str):
        for pdf in glob.glob(os.path.join(dir_path, '*.pdf')):
            self.add_pdf(pdf)

    def add_text(self, text: str):
        if not text.strip():
            return -1
        retrieved, score = self.similarity_search_with_relevance_scores(text, k=1)[0]
        if retrieved.page_content.find(text) > -1 and score >= settings.vector_db_update_threshold:
            return -1
        self.store.add_texts([text])
        self.store.save_local(self.path)
        self._append_catalogue(text[:80])
        return 0


vector_db_singleton = VectorDatabase()
