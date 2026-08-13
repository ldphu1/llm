from typing import Sequence, Optional
from langchain_core.documents import Document
from pydantic import Field,  PrivateAttr
from typing import Any
from langchain_core.documents.compressor import BaseDocumentCompressor
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer


class ONNXCrossEncoderReranker(BaseDocumentCompressor):
    top_n: int = Field(default=5)
    model_path: str
    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()

    def __init__(self, model_path: str = "./bge_reranker_onnx", top_n: int = 5):
        super().__init__(
            model_path=model_path,
            top_n=top_n
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = ORTModelForSequenceClassification.from_pretrained(model_path)

    def compress_documents(
            self,
            documents: Sequence[Document],
            query: str,
            callbacks: Optional[any] = None
    ) -> Sequence[Document]:
        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]

        inputs = self._tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")
        outputs = self._model(**inputs)

        scores = outputs.logits.squeeze(-1).tolist()
        if isinstance(scores, float):
            scores = [scores]

        doc_scores = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

        return [doc for doc, score in doc_scores[:self.top_n]]