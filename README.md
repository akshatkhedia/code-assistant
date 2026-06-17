# LangGraph Code Assistant

A sophisticated code generation system using LangGraph with RAG (Retrieval-Augmented Generation) and self-correction capabilities. This project demonstrates advanced AI/ML concepts including iterative code generation, automated testing, and intelligent error correction.

## 🚀 Features

- **LangGraph Workflow**: Advanced AI workflow with self-correction loops
- **OpenAI Integration**: GPT-4o, GPT-4o-mini, and GPT-3.5-turbo support
- **Smart Code Generation**: AI-powered code generation using LangChain documentation
- **Auto-Fix Errors**: Automatically detects and corrects import/execution issues
- **Tracing & Monitoring**: Optional LangSmith integration for debugging
- **Built-in Evaluator**: Performance testing and validation tools
- **Web Interface**: Clean Gradio UI for easy interaction

## 🏗️ Architecture

The system implements the AlphaCodium approach for iterative code generation:

1. **Document Loading**: Loads and processes documentation (default: LCEL docs)
2. **Code Generation**: Uses OpenAI with structured output to generate code
3. **Validation**: Automatically tests imports and code execution
4. **Self-Correction**: Iteratively improves solutions based on error feedback
5. **Output**: Returns validated, working code solutions

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- (Optional) LangSmith API key for tracing and evaluation

> **💡 Minimal Setup**: The app works with just your OpenAI API key! All other configuration is optional.

## 🛠️ Installation

### Quick Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd langgraph-code-assistant
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your API keys**:
   ```bash
   # Create a .env file
   echo. > .env  # Windows
   # or
   touch .env  # Linux/Mac
   
   # Edit the .env file
   notepad .env  # Windows
   # or
   nano .env  # Linux/Mac
   ```
   
   Add your configuration:
   ```env
   # Required - Only this is needed for the app to work
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Optional - LangSmith tracing (uncomment and set real values if needed)
   # LANGCHAIN_API_KEY=your_langsmith_api_key_here
   # LANGCHAIN_TRACING_V2=true
   # LANGCHAIN_PROJECT=langgraph-code-assistant
   
   # Optional - Model configuration (defaults work fine)
   # DEFAULT_MODEL=gpt-4o-mini
   # MAX_ITERATIONS=3
   # REFLECTION_MODE=do_not_reflect
   ```

## 🚀 Usage

### Web UI

1. **Start the Gradio interface**:
   ```bash
   python app_gradio.py
   ```

2. **Open your browser** to `http://localhost:7860`

### Command Line Interface

```bash
# Basic usage
python -m src.main "How do I build a RAG chain in LCEL?"

# With custom options
python -m src.main "How do I create a custom runnable?" \
    --context-url "https://python.langchain.com/docs/concepts/lcel/" \
    --model "gpt-4o-mini" \
    --max-iterations 3 \
    --verbose
```

## 📊 Performance

The system has been evaluated on LCEL coding questions and shows:

- **Improved Success Rate**: LangGraph approach outperforms simple context stuffing
- **Reliable Code Generation**: Consistent performance with structured output
- **Error Recovery**: Effective self-correction for common coding issues