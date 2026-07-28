from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, Runnable
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

def format_docs(docs: List[Document]):
    formatted_chunks = []
    for doc in docs:
        formatted_chunks .append(doc.page_content)

    return "\n\n---\n\n".join(formatted_chunks)


class Chain:
    TEMPLATE = """Bạn là một trợ lý AI. Chỉ sử dụng DUY NHẤT thông tin được cung cấp trong phần [Ngữ cảnh] dưới đây để trả lời câu hỏi.

Quy tắc bắt buộc:
1. Nếu thông tin không có trong [Ngữ cảnh], hãy trả lời: "Tài liệu không cung cấp thông tin này."
2. Tuyệt đối không sử dụng kiến thức bên ngoài hoặc tự suy đoán/bịa ra thông tin.

[Ngữ cảnh]:
{context}

[Câu hỏi]:
{question}
"""

    def __init__(self, llm_model: BaseChatModel, retriever: BaseRetriever):
        self.llm_model = llm_model
        self.retriever = retriever
        self.prompt = ChatPromptTemplate.from_template(self.TEMPLATE)


    def build_chain(self) -> RunnablePassthrough:
        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm_model
            | StrOutputParser()
        )

        return rag_chain



