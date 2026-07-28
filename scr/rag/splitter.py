from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

class Splitter:
    def __init__(self, chunk_size=500, chunk_overlap=100, strategy=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

    def get_split(self, embedding_model=None):
        if self.strategy == "sematic":
            if embedding_model is None:
                raise Exception("Require embedding model")

            return SemanticChunker(
                embeddings=embedding_model,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=90.0
            )

        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_document(self, documents, embedding_model=None):
        splitter = self.get_split(embedding_model)
        chunks = splitter.split_documents(documents)

        return chunks
