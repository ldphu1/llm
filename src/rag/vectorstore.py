import os
from typing import List, Optional
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from configs import config
import hashlib

class HybridParentRetrieverManager:
    def __init__(self, raw_docs: Optional[List[Document]] = None, embedding_model=None,
                 persist_directory="./data/chroma_db"):
        self.raw_docs = raw_docs or []
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory

        parent_docs_dir = os.path.join(persist_directory, "parent_docs")
        os.makedirs(parent_docs_dir, exist_ok=True)
        fs = LocalFileStore(parent_docs_dir)
        self.docstore = create_kv_docstore(fs)

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.PARENT_CHUNK_SIZE,
            chunk_overlap=config.PARENT_CHUNK_OVERLAP
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHILD_CHUNK_SIZE,
            chunk_overlap=config.CHILD_CHUNK_OVERLAP
        )

        self.vectorstore: Optional[Chroma] = None
        self.all_child_docs: List[Document] = []

    def _prepare_documents(self):
        parent_docs = self.parent_splitter.split_documents(self.raw_docs)
        child_docs = []

        for parent_doc in parent_docs:
            content_bytes = parent_doc.page_content.encode("utf-8")
            parent_id = parent_doc.metadata.get("doc_id") or hashlib.sha256(content_bytes).hexdigest()
            parent_doc.metadata["doc_id"] = parent_id

            self.docstore.mset([(parent_id, parent_doc)])

            sub_docs = self.child_splitter.split_documents([parent_doc])
            for sub_doc in sub_docs:
                sub_doc.metadata["doc_id"] = parent_id
                child_docs.append(sub_doc)

        self.all_child_docs = child_docs
        return child_docs

    def initialize_store(self, is_first_time_indexing: bool = False, batch_size=100):
        os.makedirs(self.persist_directory, exist_ok=True)

        self.vectorstore = Chroma(
            collection_name="my_docs",
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory,
        )

        if is_first_time_indexing and self.raw_docs:
            child_docs = self._prepare_documents()
            for i in range(0, len(child_docs), batch_size):
                doc_batch = child_docs[i: i + batch_size]
                self.vectorstore.add_documents(doc_batch)

    def get_custom_hybrid_retriever(self, top_k_child=10, top_k_parent=3):
        if not self.vectorstore:
            self.initialize_store()

        dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k_child})

        # BM25 Retriever trên Child Chunks
        # Lưu ý: Nếu restart server, bạn cần load lại child_docs hoặc cache danh sách này
        bm25_retriever = BM25Retriever.from_documents(self.all_child_docs)
        bm25_retriever.k = top_k_child

        ensemble_child_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[0.4, 0.6]
        )

        #Wrapper để map Top Child Chunks -> Unique Parent Chunks
        def retrieve_parents(query: str) -> List[Document]:
            matched_child_docs = ensemble_child_retriever.invoke(query)

            # Lấy danh sách parent_id độc nhất
            parent_ids = []
            for doc in matched_child_docs:
                p_id = doc.metadata.get("doc_id")
                if p_id and p_id not in parent_ids:
                    parent_ids.append(p_id)
                if len(parent_ids) >= top_k_parent:
                    break

            # Truy xuất Parent Document từ docstore
            parent_docs = self.docstore.mget(parent_ids)
            return [doc for doc in parent_docs if doc is not None]

        return retrieve_parents
