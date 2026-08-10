from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def get_embedding_model(provider: str = "huggingface", model_name: str = None, device: str = None):
    provider = provider.lower()

    if provider == "huggingface":
        name = model_name or "bkai-foundation-models/vietnamese-bi-encoder"
        return HuggingFaceEmbeddings(
            model_name=name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )

    elif provider == "gemini":
        name = model_name or "models/text-embedding-004"
        return GoogleGenerativeAIEmbeddings(
            model=name,
            google_api_key=""
        )

    else:
        raise ValueError(f"Provider embedding is not supported: {provider}")