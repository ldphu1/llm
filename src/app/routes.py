from langchain_ollama import OllamaLLM
from rag.loader import Loader
from rag.splitter import Splitter
from rag.vectorstore import VectorStoreManager
from rag.chain import Chain

import torch
from langchain_huggingface import HuggingFaceEmbeddings

device = "cuda" if torch.cuda.is_available() else "cpu"

embedding_model = HuggingFaceEmbeddings(
    model_name='bkai-foundation-models/vietnamese-bi-encoder',
    model_kwargs={"device": device}
)
llm_model = OllamaLLM(model="llama3")

loader = Loader(dataset_name="taidng/UIT-ViQuAD2.0")
document = loader.load_documents()

text_splitter = Splitter()
chunks = text_splitter.split_document(document, embedding_model=embedding_model)

vectorstore = VectorStoreManager(raw_docs=document, embedding_model=embedding_model)
vectorstore.initialize_store(is_first_time_indexing=True)
retriever = vectorstore.get_retriever(top_k=10)

chain = Chain(llm_model, retriever)

rag_chain = chain.build_chain()

# print("Trả lời (Streaming): ", end="")
# for chunk in rag_chain.stream("Điều 5. Nguyên tắc hội nhập và hợp tác quốc tế về địa chất, khoáng sản"):
#     print(chunk, end="", flush=True)
# print()
