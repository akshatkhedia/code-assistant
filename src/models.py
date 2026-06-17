from typing import List, TypedDict
from pydantic import BaseModel, Field


class CodeSolution(BaseModel):
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")


class GraphState(TypedDict):
    error: str
    messages: List
    generation: CodeSolution
    iterations: int
