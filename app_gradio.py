import gradio as gr
import sys
import os
from pathlib import Path
import time
from typing import Dict, Any

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.document_loader import DocumentLoader
from src.langgraph_workflow import LangGraphCodeAssistant
from src.config import config

class GradioApp:
    def __init__(self):
        self.assistant = None
        self.context_loaded = False
        self.context = None
        self.current_model = None
    
    def load_context(self):
        """Fetch the LangChain documentation to use as context for code generation."""
        try:
            print("Loading context...")
            loader = DocumentLoader()
            self.context = loader.load_lcel_docs()
            self.context_loaded = True
            print(f"Context loaded successfully! Length: {len(self.context)}")
            return f"✅ Ready! Loaded {len(self.context)} characters of LCEL documentation"
        except Exception as e:
            print(f"Error loading context: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"❌ Failed to load context: {str(e)}"
    
    def create_assistant(self):
        """Initialize the code generation assistant with the loaded context."""
        if self.context_loaded:
            try:
                self.assistant = LangGraphCodeAssistant(self.context)
                return "✅ Assistant ready! You can now ask questions about LCEL"
            except Exception as e:
                return f"❌ Failed to create assistant: {str(e)}"
        return "❌ Please load context first!"
    
    def update_status(self, model, max_iterations):
        """Show the current system status with model and configuration details."""
        if self.context_loaded:
            return f"✅ Ready | Model: {model} | Max Iterations: {max_iterations} | LangChain Tracing: {'ON' if config.langchain_tracing_v2 else 'OFF'}"
        else:
            return "❌ Please load context first"
    
    def clear_results(self):
        """Reset all output fields to empty state."""
        return "", "", "", "", ""
    
    def generate_solution(self, question, model, max_iterations):
        """Process the user's question and generate a code solution using the selected model."""
        if not question:
            return "Please enter a question first.", "", "", "", ""
        
        if not self.context_loaded:
            return "Please load context first.", "", "", "", ""
        
        try:
            # Set up the assistant with the chosen model
            if not self.assistant or self.current_model != model:
                print(f"Creating assistant with model: {model}")
                self.assistant = LangGraphCodeAssistant(self.context, model=model)
                self.current_model = model
            
            # Configure the iteration limit for self-correction
            self.assistant.max_iterations = max_iterations
            print(f"Generating solution with model: {model}, max_iterations: {max_iterations}")
            
            # Run the code generation workflow
            result = self.assistant.generate_solution(question)
            
            if not result.get("generation"):
                return "No solution generated.", "", "", "", ""
            
            solution = result["generation"]
            
            # Extract the solution components
            description = solution.prefix
            imports = solution.imports
            code = solution.code
            
            # Check if there were any validation errors during generation
            error_info = ""
            if result.get("error") != "no":
                # Look through the conversation for error details
                messages = result.get("messages", [])
                error_messages = []
                
                for role, content in messages:
                    if role == "user" and ("failed" in content.lower() or "error" in content.lower()):
                        error_messages.append(content)
                
                if error_messages:
                    error_info = "\n".join(error_messages[-2:])  # Show the most recent errors
                else:
                    error_info = "Code validation failed - check imports and execution"
            else:
                error_info = "✅ No errors detected"
            
            # Build the status message with generation details
            status = f"✅ Success" if result.get("error") == "no" else "❌ Failed"
            iterations = result.get("iterations", 0)
            status += f" | Model: {self.current_model} | Iterations: {iterations}"
            
            print(f"Generation complete: {status}")
            return status, description, imports, code, error_info
            
        except Exception as e:
            print(f"Error in generate_solution: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"❌ Error: {str(e)}", "", "", "", str(e)

def create_interface():
    """Build the Gradio web interface with all the necessary components."""
    app = GradioApp()
    
    with gr.Blocks(
        title="LangGraph Code Assistant",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .code-output {
            font-family: 'Courier New', monospace;
        }
        .gr-textbox {
            min-height: 60px !important;
        }
        .gr-textbox textarea {
            min-height: 50px !important;
            line-height: 1.4 !important;
        }
        """
    ) as interface:
        
        gr.Markdown("# 🤖 LangGraph Code Assistant")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Setup")
                
                # Status indicators for system components
                context_status = gr.Textbox(label="Context Status", interactive=False, value="Loading...", lines=2)
                assistant_status = gr.Textbox(label="Assistant Status", interactive=False, value="Initializing...", lines=2)
                
                
                model_dropdown = gr.Dropdown(
                    choices=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                    value="gpt-4o-mini",
                    label="Model"
                )
                
                max_iterations = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Max Iterations"
                )
                
                gr.Markdown("### 📊 Current Status")
                status_display = gr.Textbox(
                    label="Configuration Status",
                    value="Ready to load context",
                    interactive=False,
                    lines=2
                )
                
                gr.Markdown("### 🔧 Manual Controls")
                with gr.Row():
                    load_context_btn = gr.Button("📚 Load Context", variant="secondary", size="sm")
                    create_assistant_btn = gr.Button("🔄 Recreate Assistant", variant="secondary", size="sm")
                
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Ask a Question")
                
                question_input = gr.Textbox(
                    label="Enter your coding question about LCEL :",
                    placeholder="e.g., How do I build a RAG chain in LCEL?",
                    lines=3
                )
                
                with gr.Row():
                    generate_btn = gr.Button("🚀 Generate Solution", variant="primary", size="lg")
                    retry_btn = gr.Button("🔄 Retry", variant="secondary", size="lg")
                
                gr.Markdown("### 📊 Results")
                
                status_output = gr.Textbox(label="Status", interactive=False)
                
                with gr.Tabs():
                    with gr.Tab("📝 Description"):
                        description_output = gr.Textbox(
                            label="Solution Description",
                            lines=4,
                            interactive=False
                        )
                    
                    with gr.Tab("📦 Imports"):
                        imports_output = gr.Code(
                            label="Required Imports",
                            language="python",
                            interactive=False
                        )
                    
                    with gr.Tab("💻 Code"):
                        code_output = gr.Code(
                            label="Generated Code",
                            language="python",
                            interactive=False
                        )
                    
                    with gr.Tab("❌ Errors"):
                        error_output = gr.Textbox(
                            label="Error Details",
                            lines=6,
                            interactive=False,
                            placeholder="Error information will appear here if code validation fails..."
                        )
        
        # Wire up all the button clicks and interactions
        generate_btn.click(
            app.generate_solution,
            inputs=[question_input, model_dropdown, max_iterations],
            outputs=[status_output, description_output, imports_output, code_output, error_output]
        )
        
        # Retry clears results first, then generates again
        retry_btn.click(
            app.clear_results,
            outputs=[status_output, description_output, imports_output, code_output, error_output]
        ).then(
            app.generate_solution,
            inputs=[question_input, model_dropdown, max_iterations],
            outputs=[status_output, description_output, imports_output, code_output, error_output]
        )
        
        # Manual context reload
        load_context_btn.click(
            app.load_context,
            outputs=context_status
        ).then(
            app.create_assistant,
            outputs=assistant_status
        ).then(
            app.update_status,
            inputs=[model_dropdown, max_iterations],
            outputs=status_display
        )
        
        # Manual assistant reset
        create_assistant_btn.click(
            app.create_assistant,
            outputs=assistant_status
        ).then(
            app.update_status,
            inputs=[model_dropdown, max_iterations],
            outputs=status_display
        )
        
        # Update status when settings change
        model_dropdown.change(
            app.update_status,
            inputs=[model_dropdown, max_iterations],
            outputs=status_display
        )
        
        max_iterations.change(
            app.update_status,
            inputs=[model_dropdown, max_iterations],
            outputs=status_display
        )
        
        # Initialize everything when the page loads
        interface.load(
            app.load_context,
            outputs=context_status
        ).then(
            app.create_assistant,
            outputs=assistant_status
        ).then(
            app.update_status,
            inputs=[model_dropdown, max_iterations],
            outputs=status_display
        )
    
    return interface

def main():
    """Start up the web application and launch the interface."""
    print("🚀 Starting LangGraph Code Assistant - Gradio UI...")
    
    interface = create_interface()
    
    # Get port from environment variable (for Render.com) or use default
    port = int(os.getenv("PORT", 7860))
    
    # Use localhost for local development, 0.0.0.0 for production
    server_name = "localhost" if os.getenv("RENDER") is None else "0.0.0.0"
    
    interface.launch(
        server_name=server_name,
        server_port=port,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()
