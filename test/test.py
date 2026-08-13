from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from configs import config
from src.core.routes import retriever, rag_chain
from langchain_ollama import ChatOllama, OllamaEmbeddings

def test_rag_quality_metrics(test_cases):
    """Lấy output trực tiếp từ RAG Pipeline và kiểm tra chỉ số chất lượng."""

    questions = []
    contexts = []
    answers = []
    ground_truths = []

    for item in test_cases:
        q = item["question"]

        parent_docs = retriever(q)

        context_texts = [doc.page_content for doc in parent_docs]

        response = rag_chain.invoke(q)
        answer_text = response if isinstance(response, str) else response.get("answer", "")

        questions.append(q)
        contexts.append(context_texts)
        answers.append(answer_text)
        ground_truths.append(item["ground_truth"])

    eval_dataset = Dataset.from_dict({
        "question": questions,
        "contexts": contexts,
        "answer": answers,
        "ground_truth": ground_truths
    })

    local_judge_llm = ChatOllama(
        model="llama3.1:8b",
        temperature=0
    )

    local_judge_embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    results = evaluate(
        dataset=eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision
        ],
        llm = local_judge_llm,
        embeddings = local_judge_embeddings
    )

    print(results["faithfulness"])
    print("*" * 10)
    print(results["answer_relevancy"])
    print("*" * 10)
    print(results["context_precision"])
if __name__ == "__main__":
    test_cases =  [
        {
            "question": "Giai đoạn năm 1955-1976, Phạm Văn Đồng nắm giữ chức vụ gì?",
            "ground_truth": "Thủ tướng Chính phủ Việt Nam Dân chủ Cộng hòa."
        }
        # Có thể bổ sung thêm nhiều câu hỏi khác vào đây
    ]
    test_rag_quality_metrics(test_cases)