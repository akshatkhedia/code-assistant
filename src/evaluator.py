from typing import Dict, Any
from langsmith.schemas import Example, Run
from langsmith.evaluation import evaluate
import langsmith


class CodeEvaluator:
    
    def __init__(self):
        self.client = langsmith.Client()
    
    def check_import(self, run: Run, example: Example) -> Dict[str, Any]:
        """Check if imports are valid."""
        imports = run.outputs.get("imports")
        try:
            exec(imports)
            return {"key": "import_check", "score": 1}
        except Exception:
            return {"key": "import_check", "score": 0}

    def check_execution(self, run: Run, example: Example) -> Dict[str, Any]:
        """Check if code can be executed successfully."""
        imports = run.outputs.get("imports")
        code = run.outputs.get("code")
        try:
            exec(imports + "\n" + code)
            return {"key": "code_execution_check", "score": 1}
        except Exception:
            return {"key": "code_execution_check", "score": 0}
    
    def get_evaluators(self) -> list:
        """Get list of evaluators."""
        return [self.check_import, self.check_execution]
    
    def evaluate_predictions(self, predict_function, dataset_name: str, 
                           experiment_prefix: str, max_concurrency: int = 2,
                           metadata: Dict[str, Any] = None) -> Any:
        """
        Evaluate predictions using LangSmith.
        
        Args:
            predict_function: Function that takes an example and returns predictions
            dataset_name: Name of the dataset to evaluate on
            experiment_prefix: Prefix for the experiment name
            max_concurrency: Maximum number of concurrent evaluations
            metadata: Additional metadata for the experiment
            
        Returns:
            Evaluation results
        """
        try:
            return evaluate(
                predict_function,
                data=dataset_name,
                evaluators=self.get_evaluators(),
                experiment_prefix=experiment_prefix,
                max_concurrency=max_concurrency,
                metadata=metadata or {}
            )
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return None
