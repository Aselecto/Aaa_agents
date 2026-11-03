# Код Агента 1
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from app.core.state import ProjectState
from app.services.gemini_service import get_llm_client
from app.config import get_logger

# Инициализируем изолированный логгер для Агента 1
logger = get_logger("Agent1_Ingest")

# 1. Описываем схему валидации JSON-паспорта.
# По ней Pydantic и LLM будут строго проверять структуру полей на выходе.
class PassportSchema(BaseModel):
    content_type: str = Field(
        description="Тип контента. Строго один из вариантов: science, business, fiction, travel, general"
    )
    domain: str = Field(
        description="Предметная область или тематика (например: математика, финансы, путевые заметки)"
    )
    key_entities: List[str] = Field(
        description="Список ключевых сущностей, терминов или объектов, извлеченных из текста"
    )

# 2. Основная функция-узел (Node) для LangGraph
def run_agent_1(state: ProjectState) -> Dict[str, Any]:
    """
    ### Description
    Агент 1 (Аналитик-структуризатор). Принимает неструктурированный входящий 'хаос',
    анализирует контекст с помощью LLM, классифицирует домен и формирует 
    валидный JSON-паспорт проекта в Shared State.

    ### Args
    * `state` (ProjectState): Текущее единое состояние проекта (Shared State) из LangGraph.

    ### Returns
    * `dict`: Словарь с обновленными полями состояния (`project_passport` и статус `PLAN`).
    """
    logger.info(f"Agent 1 started processing for project ID: {state.project_id}")
    logger.debug(f"Raw input data length: {len(state.raw_input)} characters")

    # Получаем лениво инициализированный LLM-клиент (абстракция над Gemini)
    llm = get_llm_client()

    # Формируем системный промпт для разбора контекста
    prompt = f"""
    Ты — Агент первичного семантического анализа (Аналитик-структуризатор).
    Твоя задача — изучить входящий неструктурированный текст (хаос) от пользователя,
    определить его тип, домен и извлечь главные сущности.
    
    Входящий текст пользователя:
    ---
    {state.raw_input}
    ---
    
    Сформируй на основе анализа структурированный паспорт проекта.
    Ты должен вернуть ответ строго в соответствии со следующей JSON-схемой:
    {PassportSchema.model_json_schema()}
    """

    try:
        logger.info("Sending request to LLM provider for context analysis...")
        
        # Вызываем метод, который гарантирует JSON и умеет чистить за Gemini "грязный" вывод
        passport_data = llm.generate_json(prompt, PassportSchema)
        
        logger.info(f"Successfully classified content type as: {passport_data.get('content_type')}")
        logger.debug(f"Generated passport data: {passport_data}")

        # Возвращаем только то, что Агент 1 обновил в Shared State.
        # Статус PLAN автоматически триггернет запуск Агента 2 (Архитектора).
        return {
            "project_passport": passport_data,
            "status": "PLAN"
        }

    except Exception as e:
        # Логируем критическую ошибку, если вся цепочка (включая fallback-очистку) упала
        logger.error(f"Critical error in Agent 1 during LLM generation: {str(e)}", exc_info=True)
        
        # Переводим проект в статус FAIL, чтобы остановить конвейер и не тратить токены дальше
        return {
            "status": "FAIL"
        }
