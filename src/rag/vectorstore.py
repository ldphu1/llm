from langchain_core.vectorstores import VectorStore
from langchain_chroma import Chroma
import os
from typing import Optional

class VectorStore:
    def __init__(self, chunks, embedding_model, persist_directory="./data/chroma_db"):
        self.chunks = chunks
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.vectorstore: Optional[VectorStore] = None

    def initialize_store(self):
        os.makedirs(self.persist_directory, exist_ok=True)

        # self.vectorstore = Chroma.from_documents(
        #     self.chunks,
        #     self.embedding_model,
        #     persist_directory=self.persist_directory
        # )

        self.vectorstore = Chroma(
            collection_name="my_docs",
            embedding_function=self.embedding_model,
            persist_directory="./data/chroma_db"
        )


        return self.vectorstore

    def add_documents(self, documents):
        if not self.vectorstore:
            self.initialize_store()
        doc_id = self.vectorstore.add_documents(documents)

        return doc_id

    def get_retriever(self,documents ,top_k=10, search_type="similarity"):
        if not self.vectorstore:
            self.initialize_store()

        hybrid_retriever = build_hybrid_retriever(documents ,self.vectorstore, top_k)

        return hybrid_retriever


from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document


def build_hybrid_retriever(docs: list[Document], chroma_vectorstore, top_k: int = 10):
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = top_k

    dense_retriever = chroma_vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )


    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.4, 0.6]
    )

    return ensemble_retriever


