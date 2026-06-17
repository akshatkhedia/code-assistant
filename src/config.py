import os
from typing import Optional
from dotenv import load_dotenv

# Read environment variables from .env file
load_dotenv()


class Config:

    def __init__(self):
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.langchain_api_key: Optional[str] = os.getenv("LANGCHAIN_API_KEY")
        self.langchain_tracing_v2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        self.langchain_project: str = os.getenv("LANGCHAIN_PROJECT", "langgraph-code-assistant")
        self.default_model: str = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
        self.max_iterations: int = int(os.getenv("MAX_ITERATIONS", "3"))
        self.reflection_mode: str = os.getenv("REFLECTION_MODE", "do_not_reflect")
        
        # Configure LangChain tracing if it is enabled
        self._setup_langchain_tracing()
        
        # Check if we have the required API key
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required. Please set it in your .env file.")
    
    def _setup_langchain_tracing(self):
        if self.langchain_tracing_v2 and self.langchain_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.langchain_project
            print(f"LangChain tracing enabled for project: {self.langchain_project}")
        else:
            print("LangChain tracing disabled")
    
    def validate(self) -> bool:
        """Check if we have the required configuration."""
        return bool(self.openai_api_key)

config = Config()