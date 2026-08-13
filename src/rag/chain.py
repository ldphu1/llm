from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from src.core.rerank import ONNXCrossEncoderReranker



class LineListOutputParser(BaseOutputParser[List[str]]):
    """Parser giúp tách câu trả lời của LLM thành từng dòng riêng biệt."""
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")

        return [line.strip() for line in lines if line.strip()]



class Chain:
    QUERY_EXPANSION_PROMPT = """Bạn là một chuyên gia tìm kiếm thông tin.
Nhiệm vụ của bạn là tạo ra 3 phiên bản khác nhau của câu hỏi dưới đây (bằng tiếng Việt) để giúp tìm kiếm tài liệu trong cơ sở dữ liệu vector chính xác hơn.

Quy tắc bắt buộc:
- Mỗi phiên bản nằm trên một dòng riêng biệt.
- Không bổ sung số thứ tự (như 1., 2.), không gạch đầu dòng, không có lời mở đầu hay giải thích.
- Sử dụng các từ đồng nghĩa, thuật ngữ kỹ thuật tương đương hoặc góc nhìn khác của cùng một ý định.

Câu hỏi gốc: {question}
"""
    TEMPLATE = """Bạn là một trợ lý AI. Chỉ sử dụng DUY NHẤT thông tin được cung cấp trong phần [Ngữ cảnh] dưới đây để trả lời câu hỏi.

Quy tắc bắt buộc:
1. Nếu thông tin không có trong [Ngữ cảnh], hãy trả lời: "Tài liệu không cung cấp thông tin này."
2. Tuyệt đối không sử dụng kiến thức bên ngoài hoặc tự suy đoán/bịa ra thông tin.

[Ngữ cảnh]:
{context}

[Câu hỏi]:
{question}
"""

    def __init__(self, llm_model: BaseChatModel, retriever: BaseRetriever, rerunk_model="./models/bge_reranker_onnx"):
        self.llm_model = llm_model
        self.retriever = retriever

        query_expansion_prompt = PromptTemplate.from_template(self.QUERY_EXPANSION_PROMPT)
        self.query_generator = (
                query_expansion_prompt
                | self.llm_model
                | LineListOutputParser()
        )

        # cross_encoder = HuggingFaceCrossEncoder(
        #     model_name=rerunk_model
        # )
        # self.compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)

        self.compressor = ONNXCrossEncoderReranker(
            model_path=rerunk_model,
            top_n=5
        )

        self.prompt = ChatPromptTemplate.from_template(self.TEMPLATE)


    def multi_query_retrieval(self, query: str):
        generated_queries = self.query_generator.invoke({"question": query})

        all_queries = [query] + generated_queries

        candidate_docs = retrieve_unique_documents(all_queries, self.retriever)

        if not candidate_docs:
            return []

        final_docs = self.compressor.compress_documents(
            documents=candidate_docs,
            query=query
        )

        return final_docs

    def build_chain(self) -> RunnablePassthrough:
        rag_chain = (
            {"context": RunnableLambda(self.multi_query_retrieval) | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm_model
            | StrOutputParser()
        )

        return rag_chain


def format_docs(docs: List[Document]):
    formatted_chunks = []
    for doc in docs:
        formatted_chunks .append(doc.page_content)

    return "\n\n---\n\n".join(formatted_chunks)


def retrieve_unique_documents(queries: List[str], base_retriever) -> List[Document]:
    """Chạy retrieve cho từng query và lọc bỏ các document trùng lặp dựa trên doc_id."""
    unique_docs = []
    seen_ids = set()

    for query in queries:
        docs = base_retriever(query) if callable(base_retriever) else base_retriever.invoke(query)
        for doc in docs:
            doc_id = doc.metadata.get("doc_id") or hash(doc.page_content.strip())

            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)

    return unique_docs

