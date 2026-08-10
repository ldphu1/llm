from src.rag.loader import Loader
# from src.rag.splitter import Splitter
from src.rag.vectorstore import VectorStoreManager
from src.rag.chain import Chain
from src.core.llm import get_llm_model
from src.core.embedding import get_embedding_model
from configs import config

print("loading embedding model...")
embedding_model = get_embedding_model(
    provider= config.EMBEDDING_PROVIDER,
    model_name=config.LLM_MODEL,
    device=config.DEVICE
)

print("loading llm model...")
llm_model = get_llm_model(
    provider=config.LLM_PROVIDER,
    model_name=config.EMBEDDING_MODEL,
    temperature=config.TEMPERATURE
)

loader = Loader(dataset_name="taidng/UIT-ViQuAD2.0")
document = loader.load_documents()
document = document[:500]

# print("splitting...")
# text_splitter = Splitter()
# chunks = text_splitter.split_document(document, embedding_model=embedding_model)

print("splitting docs and building store...")
vectorstore = VectorStoreManager(
    raw_docs=document,
    embedding_model=embedding_model,
    persist_directory=config.PERSIST_DIRECTORY
)
vectorstore.initialize_store(is_first_time_indexing=True)

print("get retriever...")
retriever = vectorstore.get_retriever(config.TOP_K)

print("building chain...")
chain = Chain(llm_model, retriever, rerunk_model=config.RERANK_MODEL)

rag_chain = chain.build_chain()

# print("Trả lời (Streaming): ", end="")
# for chunk in rag_chain.stream("Điều 5. Nguyên tắc hội nhập và hợp tác quốc tế về địa chất, khoáng sản"):
#     print(chunk, end="", flush=True)
# print()
