from typing import Dict, Any
from langgraph.graph import END, StateGraph, START
from langchain_core.tracers import LangChainTracer
from .models import GraphState, CodeSolution
from .code_generator import CodeGenerator
from .config import config


class LangGraphCodeAssistant:
    
    def __init__(self, context: str = "", vectorstore = None, model: str = None):
        self.context = context
        self.vectorstore = vectorstore
        self.code_generator = CodeGenerator(model=model)
        self.max_iterations = config.max_iterations
        self.reflection_mode = config.reflection_mode
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(GraphState)
        
        # Define the nodes
        workflow.add_node("generate", self._generate)
        workflow.add_node("check_code", self._code_check)
        workflow.add_node("reflect", self._reflect)
        
        # Build graph
        workflow.add_edge(START, "generate")
        workflow.add_edge("generate", "check_code")
        workflow.add_conditional_edges(
            "check_code",
            self._decide_to_finish,
            {
                "end": END,
                "reflect": "reflect",
                "generate": "generate",
            },
        )
        workflow.add_edge("reflect", "generate")
        
        return workflow.compile()
    
    def _generate(self, state: GraphState) -> Dict[str, Any]:
        """
        Generate a code solution.

        Args:
            state: The current graph state

        Returns:
            New state with generation
        """
        print("---GENERATING CODE SOLUTION---")
        print(f"Current iteration: {state['iterations']}")
        print(f"Error status: {state['error']}")

        # State
        messages = state["messages"]
        iterations = state["iterations"]
        error = state["error"]

        # We have been routed back to generation with an error
        if error == "yes":
            print("Previous attempt had errors, retrying with error feedback...")
            messages += [
                (
                    "user",
                    "Now, try again. Address the previous error feedback and structure the output with a prefix, imports, and code block:",
                )
            ]

        print("Calling code generator...")
        # Solution
        code_solution = self.code_generator.generate_code(self.context, messages)
        print(f"Generated solution: {code_solution.prefix[:100]}...")
        
        messages += [
            (
                "assistant",
                f"{code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}",
            )
        ]

        # Increment
        iterations = iterations + 1
        print(f"Generation complete. Iteration: {iterations}")
        return {"generation": code_solution, "messages": messages, "iterations": iterations}

    def _code_check(self, state: GraphState) -> Dict[str, Any]:
        """
        Check code for import and execution errors.

        Args:
            state: The current graph state

        Returns:
            New state with error status
        """
        print("---CHECKING CODE---")
        print(f"Checking iteration: {state['iterations']}")

        # State
        messages = state["messages"]
        code_solution = state["generation"]
        iterations = state["iterations"]

        # Get solution components
        imports = code_solution.imports
        code = code_solution.code
        
        print(f"Imports to check: {imports}")
        print(f"Code to check: {code[:200]}...")

        # Check imports
        print("Checking imports...")
        import_valid, import_error = self.code_generator.check_imports(imports)
        if not import_valid:
            print("---CODE IMPORT CHECK: FAILED---")
            print(f"Import error: {import_error}")
            error_message = [("user", f"Your solution failed the import test: {import_error}")]
            messages += error_message
            return {
                "generation": code_solution,
                "messages": messages,
                "iterations": iterations,
                "error": "yes",
            }

        # Check execution
        print("Checking code execution...")
        exec_valid, exec_error = self.code_generator.check_execution(imports, code)
        if not exec_valid:
            print("---CODE BLOCK CHECK: FAILED---")
            print(f"Execution error: {exec_error}")
            error_message = [("user", f"Your solution failed the code execution test: {exec_error}")]
            messages += error_message
            return {
                "generation": code_solution,
                "messages": messages,
                "iterations": iterations,
                "error": "yes",
            }

        # No errors
        print("---NO CODE TEST FAILURES---")
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "no",
        }

    def _reflect(self, state: GraphState) -> Dict[str, Any]:
        """
        Reflect on previous code errors using LLM to generate a targeted fix plan.

        Args:
            state: The current graph state

        Returns:
            New state with LLM reflection added to messages
        """
        print("---REFLECTING ON ERRORS---")

        # State
        messages = state["messages"]
        iterations = state["iterations"]
        code_solution = state["generation"]

        # Invoke reflection LLM chain
        try:
            reflection_text = self.code_generator.reflect(messages)
            print(f"Reflection generated: {reflection_text[:150]}...")
            messages += [("assistant", f"Reflection & Fix Strategy:\n{reflection_text}")]
        except Exception as e:
            print(f"Warning: Reflection failed ({e}), using default fallback message.")
            messages += [("assistant", "Reflecting on execution errors to revise imports and code structure.")]
        
        return {"generation": code_solution, "messages": messages, "iterations": iterations}

    def _decide_to_finish(self, state: GraphState) -> str:
        """
        Determines whether to finish.

        Args:
            state: The current graph state

        Returns:
            Next node to call
        """
        error = state["error"]
        iterations = state["iterations"]

        if error == "no" or iterations >= self.max_iterations:
            print("---DECISION: FINISH---")
            return "end"
        else:
            print("---DECISION: RE-TRY SOLUTION---")
            if self.reflection_mode == "reflect":
                return "reflect"
            else:
                return "generate"

    def generate_solution(self, question: str) -> Dict[str, Any]:
        """
        Generate a code solution for the given question.
        
        Args:
            question: The coding question to answer
            
        Returns:
            Dictionary containing the solution and metadata
        """
        # Retrieve relevant context from vectorstore if available
        if self.vectorstore is not None:
            try:
                retrieved_docs = self.vectorstore.similarity_search(question, k=4)
                self.context = "\n\n --- \n\n".join([doc.page_content for doc in retrieved_docs])
                print(f"Retrieved {len(retrieved_docs)} relevant context chunks from FAISS vector store.")
            except Exception as e:
                print(f"Warning: FAISS similarity search failed ({e}). Falling back to static context.")

        initial_state = {
            "messages": [("user", question)],
            "iterations": 0,
            "error": ""
        }
        
        # Use tracing if available
        if config.langchain_tracing_v2 and config.langchain_api_key:
            try:
                tracer = LangChainTracer(project_name=config.langchain_project)
                result = self.workflow.invoke(initial_state, config={"callbacks": [tracer]})
            except Exception as e:
                print(f"Warning: Failed to use LangChain tracer: {e}")
                result = self.workflow.invoke(initial_state)
        else:
            result = self.workflow.invoke(initial_state)
        
        return result

