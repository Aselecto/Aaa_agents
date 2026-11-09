###Агент 4: Генератор текста и верстки (Писатель)

Работает как узел LangGraph###
    вход  (state):  status == "COMPUTE", task_specifications (от А2), computed_data (от А3)
    выход (patch):  generated_content (тексты документов), status = "QA" или "FAIL"
import json
from typing import Any, Dict, List, Optional, Tuple

from app.config import get_logger
from app.core.state import ProjectState
from app.services.llm import get_llm

# Импортируем схемы, из схемы Агента 2
# (Они лежат в папке агента 2, так как именно он их генерирует)
from app.agents.agent2_plan.schemas import Agent2Plan, DocumentSpec, SectionSpec

logger = get_logger("Agent4_Generate")

# Статусы, с которыми работает конвейер
STATUS_COMPUTE = "COMPUTE"  # Статус, при котором запускается А4 (А3 уже всё посчитал)
STATUS_QA = "QA"           # Следующий этап (Агент 5)
STATUS_FAIL = "FAIL"        # Критическая ошибка

class ErrorCode:
    UPSTREAM_FAILED = "UPSTREAM_FAILED"
    WRONG_STATUS = "WRONG_STATUS"
    NO_SPECS = "NO_SPECS"
    INVALID_SPECS = "INVALID_SPECS"
    LLM_CONFIG = "LLM_CONFIG"
    UNEXPECTED_ERROR = "UNEXPECTED_ERROR"

# =====================================================================
# Вспомогательные функции (Patch-и)
# =====================================================================

def _failure_patch(code: str, errors: List[str]) -> Dict[str, Any]:
    """Формирует результат при ошибке."""
    logger.error(f"Агент 4 завершился с ошибкой {code}: {errors}")
    return {
        "status": STATUS_FAIL,
        # Пишем ошибку в task_specifications, чтобы А5 и интерфейс ее увидели
        "task_specifications": {"ok": False, "error_code": code, "errors": errors}
    }

def _success_patch(generated_docs: Dict[str, Any]) -> Dict[str, Any]:
    """Формирует результат при успехе."""
    logger.info(f"Агент 4: успешно сгенерировано документов: {len(generated_docs)}")
    return {
        "status": STATUS_QA,
        "generated_content": generated_docs
    }

def _check_state(state: ProjectState) -> Optional[Dict[str, Any]]:
    """Проверяет, можно ли Агенту 4 начинать работу."""
    status = str(state.status).upper()
    if status == STATUS_FAIL:
        return _failure_patch(ErrorCode.UPSTREAM_FAILED, ["Предыдущий агент завершился с ошибкой."])
    if status != STATUS_COMPUTE:
        return _failure_patch(ErrorCode.WRONG_STATUS, [f"Агент 4 запускается при статусе {STATUS_COMPUTE}, а сейчас {state.status!r}."])
   
    # Проверяем, что А2 передал план
    specs = getattr(state, "task_specifications", None)
    if not specs or not specs.get("ok"):
        return _failure_patch(ErrorCode.NO_SPECS, ["Агент 2 не передал корректный план (task_specifications)."])
   
    return None

# =====================================================================
# Логика формирования промпта
# =====================================================================

def _extract_artifacts_data(section: SectionSpec, computed_data: dict) -> str:
    """Достает нужные данные от А3 для конкретного раздела."""
    if not section.uses_artifacts:
        return "Нет специфических данных от Агента 3 для этого раздела. Опирайся на общий контекст."
   
    artifacts_info = []
    for art_id in section.uses_artifacts:
        data = computed_data.get(art_id)
        if data:
            artifacts_info.append(f"Артефакт '{art_id}':\n{json.dumps(data, ensure_ascii=False, indent=2)}")
        else:
            logger.warning(f"А4: Артефакт '{art_id}' не найден в computed_data")
            artifacts_info.append(f"Артефакт '{art_id}': (Данные не найдены, упомяни, что расчеты не удались)")
           
    return "\n\n".join(artifacts_info)

def _build_messages(state: ProjectState, doc: DocumentSpec, section: SectionSpec) -> List[Tuple[str, str]]:
    """Собирает сообщения для LLM для генерации одного раздела."""
    # Достаем данные от А3
    computed_data = getattr(state, "computed_data", {}) or {}
    a3_data = _extract_artifacts_data(section, computed_data)
   
    passport = getattr(state, "project_passport", {}) or {}
    project_type = passport.get("content_type", "не указан")
   
    human_text = f"""Напиши текст для раздела документа.
   
КОНТЕКСТ ПРОЕКТА:
- Тип проекта: {project_type}
- Аудитория: {doc.audience}
- Тон: {doc.tone_of_voice}

РАЗДЕЛ:
- Заголовок: {section.title}
- Цель (что нужно донести): {section.purpose}
- Целевой объем: {section.target_length}

ДАННЫЕ ОТ АГЕНТА 3 (Вычисления):
{a3_data}

Напиши текст раздела в формате Markdown:
"""
    return [("system", SYSTEM_PROMPT), ("human", human_text)]

# =====================================================================
# Основная логика генерации
# =====================================================================

async def _generate_text(state: ProjectState, plan: Agent2Plan, llm: Any) -> Dict[str, Any]:
    """Проходит по всем документам и разделам, генерирует текст."""
    generated_docs = {}
   
    for doc_spec in plan.documents:
        logger.info(f"Агент 4: Начинаю генерацию документа '{doc_spec.deliverable_id}'")
        full_markdown = f"# Документ: {doc_spec.deliverable_id}\n\n"
       
        for section in doc_spec.sections:
            logger.debug(f"Агент 4: Генерация раздела '{section.title}'")
            messages = _build_messages(state, doc_spec, section)
           
            try:
                response = await llm.ainvoke(messages)
                # Убираем возможные кавычки-обертки, если модель их добавила
                section_text = str(response.content).strip().strip('"')
                full_markdown += f"## {section.title}\n\n{section_text}\n\n"
            except Exception as e:
                logger.exception(f"Агент 4: Ошибка генерации раздела '{section.title}'")
                full_markdown += f"## {section.title}\n\n*[Ошибка генерации текста: {e}]*\n\n"

        generated_docs[doc_spec.deliverable_id] = {
            "format": "md",
            "content": full_markdown
        }
       
    return generated_docs

# =====================================================================
# Узел LangGraph
# =====================================================================

async def run_agent_4(state: ProjectState) -> Dict[str, Any]:
    """Точка входа для LangGraph: workflow.add_node("agent_4", run_agent_4)."""
    logger.info(f"Агент 4 запущен для проекта {state.project_id}")
   
    # 1. Проверка стейта
    early_failure = _check_state(state)
    if early_failure:
        return early_failure
       
    # 2. Валидация плана от А2 (через Pydantic)
    try:
        plan = Agent2Plan.model_validate(state.task_specifications)
    except Exception as exc:
        logger.debug(f"План не прошел валидацию: {exc}")
        return _failure_patch(ErrorCode.INVALID_SPECS, ["План от Агента 2 имеет неверный формат."])
       
    if not plan.documents:
        logger.warning("Агент 4: В плане нет документов для генерации. Пропускаем этап.")
        return _success_patch({}) # Возвращаем пустой результат, идем дальше

    # 3. Инициализация LLM
    try:
        # temperature=0.3 чтобы текст был чуть живее, чем сухой план
        llm = get_llm(temperature=0.3)
    except Exception as exc:
        logger.error(f"Модель не настроена: {exc}")
        return _failure_patch(ErrorCode.LLM_CONFIG, ["Сервис языковой модели не настроен."])

    # 4. Генерация текста
    try:
        generated_content = await _generate_text(state, plan, llm)
        return _success_patch(generated_content)
    except Exception as exc:
        logger.exception("Агент 4: Непредвиденный сбой при генерации")
        return _failure_patch(ErrorCode.UNEXPECTED_ERROR, [f"Сбой генерации: {str(exc)}"])# Код Агента 4
