from langchain_core.vectorstores import VectorStore
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore, LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from typing import Optional

class VectorStoreManager:
    def __init__(self, raw_docs, embedding_model, persist_directory="./data/chroma_db"):
        self.raw_docs = raw_docs
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.vectorstore: Optional[VectorStore] = None
        self.parent_retriever: Optional[ParentDocumentRetriever] = None

        parent_docs_dir = os.path.join(persist_directory, "parent_docs")
        os.makedirs(parent_docs_dir, exist_ok=True)
        fs = LocalFileStore(parent_docs_dir)
        self.docstore = create_kv_docstore(fs)

    def initialize_store(self, is_first_time_indexing: bool = False):
        os.makedirs(self.persist_directory, exist_ok=True)

        self.vectorstore = Chroma(
            collection_name="my_docs",
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory,
        )

        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)

        self.parent_retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )

        if is_first_time_indexing and self.raw_docs:
            self.parent_retriever.add_documents(self.raw_docs)

        return self.vectorstore

    def get_retriever(self, top_k=10):
        if not self.vectorstore:
            self.initialize_store()

        hybrid_retriever = build_hybrid_retriever(self.raw_docs, self.parent_retriever, top_k=top_k)

        return hybrid_retriever


from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


def build_hybrid_retriever(raw_docs, parent_retriever: ParentDocumentRetriever, top_k: int = 10):
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    parent_docs = parent_splitter.split_documents(raw_docs) if raw_docs else []

    bm25_retriever = BM25Retriever.from_documents(parent_docs)
    bm25_retriever.k = top_k

    # 2. Cấu hình Dense Search (ParentDocumentRetriever)
    parent_retriever.search_kwargs = {"k": top_k}

    # 3. Kết hợp 2 Retriever bằng EnsembleRetriever (RRF)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, parent_retriever],
        weights=[0.4, 0.6],
    )

    return ensemble_retriever




