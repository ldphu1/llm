from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI

def get_llm_model(provider: str = "ollama", model_name: str = None, temperature: float = 0.0) -> BaseChatModel:
    provider = provider.lower()

    if provider == "gemini":
        name = model_name or "gemini-3.6-flash"
        return ChatGoogleGenerativeAI(
            model=name,
            temperature=temperature,
            google_api_key="",
        )

    elif provider == "ollama":
        name = model_name or "llama3"
        return OllamaLLM(
            model=name,
            temperature=temperature,
        )

    elif provider == "openai":
        name = model_name or "gpt-4o-mini"
        return ChatOpenAI(
            model=name,
            temperature=temperature,
            api_key=""
        )

    else:
        raise ValueError(f"Provider LLM is not supported: {provider}")