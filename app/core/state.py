# Определение структуры Shared State (Pydantic)
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

class ProjectState(BaseModel):
    project_id: str
    status: str
    raw_input: str
    project_passport: Optional[Dict[str, Any]] = None
    task_specifications: Optional[Dict[str, Any]] = None
    computed_data: Optional[Dict[str, Any]] = None
    generated_content: Optional[Dict[str, Any]] = None
    qa_report: Optional[Dict[str, Any]] = None
    loop_count: int = 0
