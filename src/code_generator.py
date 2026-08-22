import os
import multiprocessing
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tracers import LangChainTracer
from langsmith import Client
from typing import Dict, Any, Tuple
from .models import CodeSolution
from .config import config


def _run_exec_target(code_str: str, queue: multiprocessing.Queue):
    """Worker target for safe isolated code execution."""
    try:
        # Isolated execution namespace
        exec_globals = {"__builtins__": __builtins__}
        exec(code_str, exec_globals)
        queue.put((True, ""))
    except Exception as e:
        queue.put((False, str(e)))


def _safe_exec(code_str: str, timeout: int = 5) -> Tuple[bool, str]:
    """Execute python code in a isolated child process with timeout protection."""
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_run_exec_target, args=(code_str, q))
    p.start()
    p.join(timeout)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return False, f"Execution timed out after {timeout} seconds (possible infinite loop or blocking call)."
    
    if not q.empty():
        return q.get()
    return False, "Execution terminated unexpectedly without output."


class CodeGenerator:
    
    def __init__(self, model: str = None, temperature: float = 0):
        self.model = model or config.default_model
        self.temperature = temperature
        api_key = config.openai_api_key or "sk-dummy-key"
        self.llm = ChatOpenAI(temperature=temperature, model=self.model, api_key=api_key)
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
                Here is relevant documentation context:  \n ------- \n  {context} \n ------- \n Answer the user 
                question based on the above provided documentation. Ensure any code you provide can be executed \n 
                with all required imports and variables defined. Structure your answer with a description of the code solution. \n
                Then list the imports. And finally list the functioning code block. Here is the user question:""",
            ),
            ("placeholder", "{messages}"),
        ])
        
        self.code_gen_chain = self.code_gen_prompt | self.llm.with_structured_output(CodeSolution)
        
        self.reflect_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert Python assistant reflecting on code execution errors.
Review the previous conversation messages, including the code generation attempt and the execution error traceback.
Analyze why the code failed (e.g. missing import, syntax error, variable mismatch) and provide a concise reflection and correction strategy.""",
            ),
            ("placeholder", "{messages}"),
        ])
        
        self.reflect_chain = self.reflect_prompt | self.llm
    
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
    
    def reflect(self, messages: list) -> str:
        """
        Reflect on previous errors and output a reflection response.
        
        Args:
            messages: Conversation messages containing error feedback
            
        Returns:
            Reflection string explanation
        """
        if self.tracer:
            res = self.reflect_chain.invoke(
                {"messages": messages},
                config={"callbacks": [self.tracer]}
            )
        else:
            res = self.reflect_chain.invoke({"messages": messages})
        return res.content
    
    def check_imports(self, imports: str, timeout: int = 5) -> Tuple[bool, str]:
        """
        Check if imports are valid using sandboxed execution with timeout.
        
        Args:
            imports: Import statements to check
            timeout: Maximum allowed execution time in seconds
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        return _safe_exec(imports, timeout=timeout)
    
    def check_execution(self, imports: str, code: str, timeout: int = 5) -> Tuple[bool, str]:
        """
        Check if code can be executed successfully using sandboxed execution with timeout.
        
        Args:
            imports: Import statements
            code: Code to execute
            timeout: Maximum allowed execution time in seconds
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        return _safe_exec(imports + "\n" + code, timeout=timeout)