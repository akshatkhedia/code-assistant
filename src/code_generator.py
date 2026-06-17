import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tracers import LangChainTracer
from langsmith import Client
from typing import Dict, Any
from .models import CodeSolution
from .config import config


class CodeGenerator:
    
    def __init__(self, model: str = None, temperature: float = 0):
        self.model = model or config.default_model
        self.temperature = temperature
        self.llm = ChatOpenAI(temperature=temperature, model=self.model)
        self._setup_tracing()
        self._setup_prompt()
    
    def _setup_tracing(self):
        self.tracer = None
        if config.langchain_tracing_v2 and config.langchain_api_key:
            try:
                # Initialize LangSmith client
                self.langsmith_client = Client(api_key=config.langchain_api_key)
                # Set up tracer
                self.tracer = LangChainTracer(project_name=config.langchain_project)
                print(f"LangChain tracing initialized for project: {config.langchain_project}")
            except Exception as e:
                print(f"Failed to initialize LangChain tracing: {e}")
                self.tracer = None
        else:
            print("LangChain tracing not configured")
    
    def _setup_prompt(self):
        self.code_gen_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a coding assistant with expertise in LCEL, LangChain expression language. \n 
                Here is a full set of LCEL documentation:  \n ------- \n  {context} \n ------- \n Answer the user 
                question based on the above provided documentation. Ensure any code you provide can be executed \n 
                with all required imports and variables defined. Structure your answer with a description of the code solution. \n
                Then list the imports. And finally list the functioning code block. Here is the user question:""",
            ),
            ("placeholder", "{messages}"),
        ])
        
        self.code_gen_chain = self.code_gen_prompt | self.llm.with_structured_output(CodeSolution)
    
    def generate_code(self, context: str, messages: list) -> CodeSolution:
        """
        Generate code solution based on context and messages.
        
        Args:
            context: The documentation context
            messages: List of conversation messages
            
        Returns:
            CodeSolution object with prefix, imports, and code
        """
        # Use tracer if available
        if self.tracer:
            return self.code_gen_chain.invoke(
                {"context": context, "messages": messages},
                config={"callbacks": [self.tracer]}
            )
        else:
            return self.code_gen_chain.invoke({
                "context": context, 
                "messages": messages
            })
    
    def check_imports(self, imports: str) -> tuple[bool, str]:
        """
        Check if imports are valid.
        
        Args:
            imports: Import statements to check
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            exec(imports)
            return True, ""
        except Exception as e:
            return False, str(e)
    
    def check_execution(self, imports: str, code: str) -> tuple[bool, str]:
        """
        Check if code can be executed successfully.
        
        Args:
            imports: Import statements
            code: Code to execute
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            exec(imports + "\n" + code)
            return True, ""
        except Exception as e:
            return False, str(e)