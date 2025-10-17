"""
vector_database.py
------------------

Implements the **VectorDatabase** service for the CoSMIC Coaching Writer.

This module provides a persistent FAISS-based vector store used for
retrieval-augmented generation (RAG). It indexes and retrieves textual
information from academic PDFs and user-submitted text.

Responsibilities:
  - Initialize and persist FAISS vector index.
  - Add and deduplicate PDF and text data.
  - Store a CSV catalogue of ingested sources.
  - Provide similarity search interface for retrieval components.

Database layout:
  /database/vector_db/
    ├── index.faiss
    ├── index.pkl
    └── file_list.csv  → (metadata catalogue)
"""


import os, glob, pytz, sys, csv

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../..")

from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.document_loaders import PyPDFLoader
from ..core.config import settings

resolved_device = settings.device


# =============================================================================================================
class BCOLORS:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

# =============================================================================================================

INFOR_DICT = {
    "success": {"color": BCOLORS.OKGREEN, "comment": "Success"},
    "fail": {"color": BCOLORS.FAIL, "comment": "Fail"},
    "warning": {"color": BCOLORS.WARNING, "comment": "Warning"},
    "info": {"color": BCOLORS.HEADER, "comment": "Info"},
    "error": {"color": BCOLORS.FAIL, "comment": "Error"},
    "hint": {"color": BCOLORS.OKGREEN, "comment": "Hint"},
}

# =============================================================================================================

def set_color(
    status: str,
    information: str
):
    """Set color to display information on terminal.

    Args:
        status (str): information type, see INFOR_DICT.keys.
        information (str): information to be printed.

    Returns:
        information (str): colorized information.
    """
    status = status.lower()

    return f"{INFOR_DICT[status]['color']}[{INFOR_DICT[status]['comment']}]{BCOLORS.ENDC} {information}"

class ServiceBase:
    def __init__(
        self,
        log_file: str=None,
    ):
        """Base class for the services in OpenSI-CoSMIC.

        Args:
            log_file (str, optional): (relative) log file path for storing printed information.
                Defaults to None.
        """
        # Set log file path.
        self.log_file = log_file

        # Get the root of the current file to set log file path as absolute path if it is a relative path.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root = f"{current_dir}/../.."

    def set_log_file(
        self,
        log_file: str
    ):
        """Set log file externally.

        Args:
            log_file (str): (relative) log file path.
        """
        self.log_file = log_file

class VectorDatabase(ServiceBase):
    def __init__(
        self,
        document_analyser_model: str="gte-small",
        local_database_path: str="database/vector_db",
        vector_database_update_threshold: float=0.98,
        device: str | None = None,
        **kwargs
    ):
        """Vector database service.

        Args:
            document_analyser_model (str, optional): document analyser/process model.
            local_database_path (str, optional): path of local vector database on disk.
                Default to "database/vector_database".
            vector_database_update_threshold (float, optional): contents with similarity >= this threshold
                will be skipped. Default to 0.98.
            device (str): Compute device ("cpu", "cuda", or "mps").
            Defaults to "gte-small".
        """
        super().__init__(**kwargs)

        # Set config.
        # Set to absolute path.
        if local_database_path != "" and not os.path.isabs(local_database_path):
            local_database_path = os.path.join(self.root, local_database_path)

        # Use default one.
        if not os.path.exists(local_database_path):
            if local_database_path != "":
                print(
                    set_color(
                        "warning",
                        f"Vector database \"{local_database_path}\" not exist" \
                        f", use default \"database/vector_database\"."
                    )
                )

            local_database_path = os.path.join(self.root, "database/vector_database")

        # Get the catalogue path and threshold.
        self.local_database_path = local_database_path
        self.local_database_catalogue_path = os.path.join(local_database_path, "file_list.csv")
        self.vector_database_update_threshold = vector_database_update_threshold

        # Create local database directory.
        if local_database_path != "":
            local_database_name = local_database_path.split("/")[-1]
            local_database_directory = local_database_path.replace(
                "/" + local_database_name,
                ""
            )
            os.makedirs(local_database_directory, exist_ok=True)

        # Write head in catalogue file.
        if not os.path.exists(self.local_database_catalogue_path):
            catalogue_pt = open(self.local_database_catalogue_path, "w")
            catalogue = csv.writer(catalogue_pt)
            catalogue.writerow(["Source", "Time", "Comment"])
            catalogue_pt.close()

        # For document analysis and knowledge database generation/update.
        EMBEDDING_MODEL_DICT = {'gte-small': "thenlper/gte-small"}

        # Set page separators.
        MARKDOWN_SEPARATORS = ["\n\n", "\n", ""]

        # Set splitter to split a document into pages.
        self.document_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            add_start_index=True,
            strip_whitespace=True,
            separators=MARKDOWN_SEPARATORS,
        )

        # Build a document analyser.
        EMBEDDING_MODEL_NAME = EMBEDDING_MODEL_DICT[document_analyser_model]

        self.database_update_embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            multi_process=False,  # TODO
            model_kwargs={"device": resolved_device},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Build processor to handle a new document for database updates.
        # Find the API at https://api.python.langchain.com/en/latest/vectorstores
        # /langchain_community.vectorstores.faiss.FAISS.html
        # Build a processor to handle a sentence for database updates.

        # Load a local database from a file
        if os.path.exists(f"{local_database_path}/index.faiss"):
            self.database = FAISS.load_local(
                local_database_path,
                self.database_update_embedding,
                allow_dangerous_deserialization=True
            )

            print(
                set_color(
                    "success",
                    f"Load \"{os.path.abspath(local_database_path)}\" to vector database."
                )
            )
        else:
            self.database = FAISS.from_texts(
                ["Use FAISS as database updater"],
                self.database_update_embedding,
            )

        # Set search strategy.
        self.database.distance_strategy = DistanceStrategy.COSINE

        # Set a time stamp to highlight the most recently updated information.
        self.time_stamper = lambda time_stamp: pytz.utc.localize(time_stamp) \
            .astimezone(pytz.timezone('Australia/Sydney')).strftime("%B, %Y")

        # Set a time stamp to update vector database catalogue.
        self.catalogue_time_stamper = lambda time_stamp: pytz.utc.localize(time_stamp) \
            .astimezone(pytz.timezone('Australia/Sydney')).strftime("%m/%d/%Y, %H:%M:%S")

    def similarity_search_with_relevance_scores(
        self,
        *args,
        **kwargs
    ):
        """Retriever from the vector database.

        Returns:
            context (str): retrieved information.
        """
        return self.database.similarity_search_with_relevance_scores(*args, **kwargs)

    def quit(self):
        """Release document analyser model.
        """
        if self.database_update_embedding:
            del self.database_update_embedding

    def add_documents(
        self,
        document_paths
    ):
        """Add context from a document or multiple documents to the vector database.

        Args:
            document_paths (string or list): a document path or multiple such paths.
        """
        # Set as a list for loop.
        if not isinstance(document_paths, list):
            document_paths = [document_paths]

        # Update per document.
        for document_path in document_paths:
            if not os.path.exists(document_path): continue
            self.update_database_from_document(document_path)

    def add_document_directory(
        self,
        document_dir: str
    ):
        """Add all .pdf in a folder to the vector database.

        Args:
            document_dir (str): a directory of .pdf to be added to the vector database.
        """
        if os.path.exists(document_dir):
            # Find all pdf in a folder.
            document_paths = glob.glob(f"{document_dir}/*.pdf")

            # Add these documents.
            self.add_documents(document_paths)

    def update_database_catalogue(
        self,
        metadata: str
    ):
        """Update catalogue of vector database.

        Args:
            metadata (str|list): contents to be added.
        """
        # Set as a list for loop.
        if not isinstance(metadata, list):
            metadata = [metadata]

        # Open the catalogue file.
        catalogue_pt = open(self.local_database_catalogue_path, "a")
        catalogue = csv.writer(catalogue_pt)

        # Write metadata.
        for data in metadata:
            if isinstance(data, str):
                catalogue.writerow([
                    data,
                    self.catalogue_time_stamper(datetime.now()),
                    ""
                ])

        # Close the file.
        catalogue_pt.close()

    def update_database_from_document(self, document_path: str):
        """Add a document to the vector database with filename metadata.

        Args:
            document_path (str): path to a PDF document.
        """
        if not os.path.exists(document_path):
            print(set_color("warning", f"Document {document_path} not found."))
            return

        # Load and split PDF pages
        loader = PyPDFLoader(document_path)
        pages = loader.load_and_split()

        # Normalize whitespace
        for i in range(len(pages)):
            pages[i].page_content = pages[i].page_content.replace("\t", " ")

        # Prepare metadata-aware document chunks
        filename = os.path.basename(document_path)
        document_processed = []

        for doc in pages:
            # Prevent duplicates by similarity check
            try:
                content_retrieved, similarity_score = self.similarity_search_with_relevance_scores(
                    doc.page_content, k=1
                )[0]
            except IndexError:
                similarity_score = 0.0
                content_retrieved = None

            if (
                similarity_score >= self.vector_database_update_threshold
                or (content_retrieved and content_retrieved.page_content.find(doc.page_content) > -1)
                or (content_retrieved and doc.page_content.find(content_retrieved.page_content) > -1)
            ):
                continue

            # Split long text into smaller chunks
            sub_docs = self.document_splitter.split_documents([doc])

            # Attach filename metadata to each sub-doc
            for sub_doc in sub_docs:
                sub_doc.metadata["source"] = filename
            document_processed.extend(sub_docs)

        # Store in FAISS if there’s new content
        if document_processed:
            self.database.add_documents(document_processed)
            self.update_database_catalogue(filename)
            self.database.save_local(self.local_database_path)
            print(set_color("info", f"Added '{filename}' ({len(document_processed)} chunks)."))
        else:
            print(set_color("warning", f"Contents of '{filename}' already exist."))

    def update_database_from_text(
        self,
        text: str
    ):
        """Add a sentence to the vector database.

        Args:
            text (str): a text sentence.

        Returns:
            status (int): skip (-1) or not (0).
        """
        if text != '':
            # Skip for high-similar text.
            content_retrieved, _ = self.similarity_search_with_relevance_scores(text, k=1)[0]
            content_retrieved = content_retrieved.page_content

            # If the same as existing contents, skip the text.
            if content_retrieved.find(text) > -1:
                print(set_color(
                    "warning",
                    f"Similar contents found: '{content_retrieved}' for '{text}'."
                ))

                return -1

            # Update the text with timestamp.
            text = f"{text} by the date {self.time_stamper(datetime.now())}"

            # Add text to database.
            self.database.add_texts([text])

            # Update database catalogue.
            self.update_database_catalogue(text)

            # Save to local file.
            self.database.save_local(self.local_database_path)

            # Print the progress.
            print(set_color('info', f"Update database with '{text}'."))

            return 0

# Singleton instance
vector_db_singleton = VectorDatabase()