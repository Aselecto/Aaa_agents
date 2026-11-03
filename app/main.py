# Инициализация FastAPI и сборка LangGraph
import logging
import uuid
from typing import Literal
from fastapi import FastAPI, BackgroundTasks, HTTPException, status, Path
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Импорты вашей бизнес-логики и конфигурации
from app.core.state import ProjectState 
from app.agents.agent1_ingest.logic import run_agent_1
from app.agents.agent2_plan.logic import run_agent_2
from app.agents.agent3_compute.logic import run_agent_3
from app.agents.agent4_generate.logic import run_agent_4
from app.agents.agent5_qa.logic import run_agent_5
from app.config import settings

# Настройка логирования по стандарту production (12-factor app)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("multi_agent_pipeline")

app = FastAPI(
    title="Scalable Multi-Agent Pipeline",
    description="Production-ready asynchronous LangGraph pipeline powered by FastAPI",
    version="1.0.0"
)

# Использование встроенного чекпоинтера для потокобезопасного хранения состояния
memory = MemorySaver()


def qa_routing(state: ProjectState) -> Literal["agent_3", "__end__"]:
    """
    Определяет следующий шаг графа на основе статуса проверки качества (QA).
    """
    if state.status == "RETRY" and state.loop_count < settings.MAX_QA_LOOPS:
        logger.info(f"QA Loop Count: {state.loop_count}. Sending back to agent_3 for retry.")
        return "agent_3"
    
    logger.info("QA verification passed or max loops reached. Finalizing pipeline.")
    return END  # Мапится на "__end__" под капотом LangGraph


# --- ИНИЦИАЛИЗАЦИЯ И СБОРКА ГРАФА ---
workflow = StateGraph(ProjectState)

# Регистрация узлов
workflow.add_node("agent_1", run_agent_1)
workflow.add_node("agent_2", run_agent_2)
workflow.add_node("agent_3", run_agent_3)
workflow.add_node("agent_4", run_agent_4)
workflow.add_node("agent_5", run_agent_5)

# Определение связей (Линейный старт -> Fan-out/Fan-in)
workflow.set_entry_point("agent_1")
workflow.add_edge("agent_1", "agent_2")

# Параллельное ветвление (Fan-out)
workflow.add_edge("agent_2", "agent_3")
workflow.add_edge("agent_2", "agent_4")

# Слияние потоков в один узел (Fan-in)
workflow.add_edge("agent_3", "agent_5")
workflow.add_edge("agent_4", "agent_5")

# Условные переходы
workflow.add_conditional_edges(
    "agent_5",
    qa_routing,
    {
        "agent_3": "agent_3",
        "__end__": END
    }
)

# Компиляция графа с поддержкой персистентности (чекпоинтами)
agent_pipeline = workflow.compile(checkpointer=memory)


# --- DTO / СХЕМЫ ВАЛИДАЦИИ ДАННЫХ ---

class PipelineInput(BaseModel):
    task: str = Field(
        ..., 
        min_length=5, 
        max_length=5000, 
        description="Текст или описание задачи для обработки пайплайном агентов"
    )

class PipelineResponse(BaseModel):
    project_id: uuid.UUID = Field(..., description="Уникальный идентификатор запущенной сессии")
    status: str = Field(default="processing", description="Текущий статус создания задачи")


# --- API СЛОЙ ---

@app.post(
    "/api/v1/pipeline/start", 
    response_model=PipelineResponse, 
    status_code=status.HTTP_202_ACCEPTED,
    summary="Запустить пайплайн агентов",
    tags=["Pipeline"]
)
async def start_pipeline(payload: PipelineInput, background_tasks: BackgroundTasks):
    """
    Инициализирует и асинхронно запускает многоагентный граф LangGraph.
    Возвращает `project_id` для последующего отслеживания состояния.
    """
    project_id = uuid.uuid4()
    logger.info(f"Received request to start pipeline. Generated Project ID: {project_id}")
    
    async def run_graph_async(thread_id: str, task_text: str):
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # 1. Инстанцируем стейт для валидации входящих данных
            validated_state = ProjectState(
                task=task_text,
                status="START",
                loop_count=0
            )
            
            # 2. ИСПРАВЛЕНО: Извлекаем словарь (dict) для безопасной передачи в ainvoke
            # Поддерживает как Pydantic v2 (model_dump), так и v1 (dict)
            if hasattr(validated_state, "model_dump"):
                initial_payload = validated_state.model_dump()
            else:
                initial_payload = validated_state.dict()
                
            logger.info(f"Executing agent pipeline for thread {thread_id}...")
            
            # 3. Передаем именно СЛОВАРЬ
            await agent_pipeline.ainvoke(initial_payload, config=config)
            logger.info(f"Pipeline executed successfully for thread {thread_id}.")
            
        except Exception as e:
            logger.exception(f"Critical exception occurred during pipeline execution for thread {thread_id}: {e}")

    # Передаем выполнение в фоновый пул FastAPI (не блокирует Event Loop)
    background_tasks.add_task(run_graph_async, str(project_id), payload.task)
    
    return PipelineResponse(project_id=project_id)


@app.get(
    "/api/v1/pipeline/status/{project_id}", 
    summary="Получить статус выполнения",
    tags=["Pipeline"]
)
async def get_pipeline_status(
    project_id: uuid.UUID = Path(..., description="UUID запущенного проекта")
):
    """
    Возвращает текущую позицию (next_step) графа и накопленные данные из состояния (`ProjectState`).
    """
    config = {"configurable": {"thread_id": str(project_id)}}
    
    try:
        state = await agent_pipeline.aget_state(config)
    except Exception as e:
        logger.error(f"Failed to fetch state for thread {project_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при обращении к хранилищу состояний"
        )
    
    # Если в памяти нет чекпоинта по данному thread_id
    if not state or not state.values:
        logger.warning(f"State not found for project ID: {project_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Проект с таким ID не найден в системе или еще не инициализировался"
        )
        
    return {
        "project_id": project_id,
        "is_completed": len(state.next) == 0,  # Если список следующих шагов пуст — граф завершился
        "next_steps": state.next,  
        "values": state.values    
    }
