import argparse
import sys
from typing import Optional
from .document_loader import DocumentLoader
from .langgraph_workflow import LangGraphCodeAssistant
from .config import config


def main():
    parser = argparse.ArgumentParser(description="LangGraph Code Assistant")
    parser.add_argument("question", help="The coding question to answer")
    parser.add_argument("--context-url", default="https://python.langchain.com/docs/concepts/lcel/",
                       help="URL to load documentation from")
    parser.add_argument("--model", default=config.default_model,
                       help="OpenAI model to use")
    parser.add_argument("--max-iterations", type=int, default=config.max_iterations,
                       help="Maximum number of iterations")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    try:
        # Load documentation
        if args.verbose:
            print("Loading documentation...")
        
        loader = DocumentLoader()
        context = loader.load_lcel_docs(args.context_url)
        
        if args.verbose:
            print(f"Loaded {len(context)} characters of documentation")
        
        # Create assistant
        assistant = LangGraphCodeAssistant(context, model=args.model)
        assistant.max_iterations = args.max_iterations
        
        # Generate solution
        if args.verbose:
            print(f"Generating solution for: {args.question}")
        
        result = assistant.generate_solution(args.question)
        
        # Display results
        print("\n" + "="*50)
        print("SOLUTION GENERATED")
        print("="*50)
        
        if result["generation"]:
            solution = result["generation"]
            print(f"\nDescription: {solution.prefix}")
            print(f"\nImports:\n{solution.imports}")
            print(f"\nCode:\n{solution.code}")
        
        print(f"\nIterations used: {result['iterations']}")
        print(f"Final error status: {result['error']}")
        
        if args.verbose:
            print(f"\nFull conversation:")
            for i, (role, content) in enumerate(result["messages"]):
                print(f"{i+1}. {role.upper()}: {content}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
