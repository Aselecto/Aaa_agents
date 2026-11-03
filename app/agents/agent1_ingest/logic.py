# Код Агента 1
from app.core.state import ProjectState
from app.services.gemini_service import get_llm_client
from pydantic import BaseModel, Field
from typing import List

# Описываем желаемую схему JSON прямо у агента
class PassportSchema(BaseModel):
    content_type: str = Field(description="Тип контента: наука, тревел и т.д.")
    domain: str
    key_entities: List[str]

def run_agent_1(state: ProjectState):
    # Получаем изолированный клиент (сейчас это Gemini, завтра сменим по щелчку в config.py)
    llm = get_llm_client()
    
    prompt = f"Проанализируй хаос: {state.raw_input}"
    
    # Безопасная генерация JSON с валидацией по схеме
    passport_json = llm.generate_json(prompt, PassportSchema)
    
    return {
        "project_passport": passport_json,
        "status": "PLAN"
    }
