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

class VectorStoreManager:
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

        for parent_idx, parent_doc in enumerate(parent_docs):

            # ID của document gốc
            document_id = parent_doc.metadata.get("doc_id")

            if not document_id:
                content_bytes = parent_doc.page_content.encode("utf-8")
                document_id = hashlib.sha256(content_bytes).hexdigest()

            # ID duy nhất cho từng parent chunk
            parent_id = f"{document_id}_parent_{parent_idx}"

            # Lưu metadata
            parent_doc.metadata["document_id"] = document_id
            parent_doc.metadata["parent_id"] = parent_id

            # Lưu Parent vào DocStore
            self.docstore.mset([
                (parent_id, parent_doc)
            ])

            sub_docs = self.child_splitter.split_documents([parent_doc])

            for child_idx, sub_doc in enumerate(sub_docs):
                sub_doc.metadata["document_id"] = document_id
                sub_doc.metadata["parent_id"] = parent_id

                child_docs.append(sub_doc)

        self.all_child_docs = child_docs

        return child_docs

    def _load_child_docs(self):
        if self.vectorstore is None:
            return []

        data = self.vectorstore.get(
            include=["documents", "metadatas"]
        )

        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])

        child_docs = []

        for content, metadata in zip(documents, metadatas):
            child_docs.append(
                Document(
                    page_content=content,
                    metadata=metadata or {}
                )
            )

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

        self._load_child_docs()

    def _rank_parents(self, matched_child_docs: List[Document], top_k_parent: int) -> List[str]:
        parent_scores = {}

        for rank, doc in enumerate(matched_child_docs, start=1):
            parent_id = doc.metadata.get("parent_id")

            if not parent_id:
                continue

            # Reciprocal Rank Score
            score = 1.0 / rank

            parent_scores[parent_id] = (
                    parent_scores.get(parent_id, 0.0) + score
            )

        ranked_parents = sorted(
            parent_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            parent_id
            for parent_id, _ in ranked_parents[:top_k_parent]
        ]

    def get_retriever(self, top_k_child=10, top_k_parent=3):
        if not self.vectorstore:
            self.initialize_store()

        dense_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": top_k_child}
        )

        bm25_retriever = BM25Retriever.from_documents(
            self.all_child_docs
        )

        bm25_retriever.k = top_k_child

        ensemble_child_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[0.4, 0.6]
        )

        def retrieve_parents(query: str) -> List[Document]:
            matched_child_docs = ensemble_child_retriever.invoke(query)

            parent_ids = self._rank_parents(
                matched_child_docs,
                top_k_parent
            )

            parent_docs = self.docstore.mget(parent_ids)

            return [
                doc for doc in parent_docs
                if doc is not None
            ]

        return retrieve_parents
