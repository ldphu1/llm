from datasets import load_dataset
from langchain_core.documents import Document

# thangvip/vietnamese-legal-qa

class Loader:
    def __init__(self, dataset_name="taidng/UIT-ViQuAD2.0"):
        self.dataset_name = dataset_name
        self.datasets = load_dataset(self.dataset_name)


    def load_documents(self):
        documents = [
            Document(
                page_content=row["context"],
                metadata={
                    "title": row.get("title", ""),
                    "id": i
                }
            )
            for i, row in enumerate(self.datasets["train"])
        ]

        return documents

if __name__ == "__main__":
    loader = Loader(dataset_name="taidng/UIT-ViQuAD2.0")

    doc = loader.load_documents()
    print(len(doc))
    # for d in doc:
    #     print(d)
    #
    #     break