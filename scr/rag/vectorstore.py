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

    def get_retriever(self, top_k=3, search_type="similarity"):
        if not self.vectorstore:
            self.initialize_store()

        return self.vectorstore.as_retriever(top_k=top_k, search_type=search_type)


