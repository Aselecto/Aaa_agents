# Код Агента 5 — Проверка качества

from typing import Dict, Any, List, Literal
import logging
import json

from pydantic import BaseModel

from app.core.state import ProjectState
from app.services.gemini_service import get_llm_client
from .prompts import QA_PROMPT

logger = logging.getLogger("Agent5_QA")


# Структура ответа Gemini
class QAResult(BaseModel):
    status: Literal["PASS", "RETRY"]
    errors: List[str]


def run_agent_5(state: ProjectState) -> Dict[str, Any]:

    errors = []

    # 1. Проверяем план Агента 2
    task_specs = state.task_specifications or {}

    if not task_specs.get("ok"):
        errors.append("План Агента 2 отсутствует или содержит ошибку")

    # 2. Проверяем документы Агента 4
    generated_content = state.generated_content or {}

    if not generated_content:
        errors.append("Агент 4 не создал документ")

    # 3. Проверяем список документов
    deliverables = task_specs.get("deliverables", [])

    if not deliverables:
        errors.append("В плане отсутствует список документов")

    # 4. Проверяем каждый документ
    for deliverable in deliverables:

        document_id = deliverable.get("id")

        if not document_id:
            errors.append("У документа отсутствует ID")
            continue

        document = generated_content.get(document_id)

        if not document:
            errors.append(f"Документ {document_id} отсутствует")
            continue

        if isinstance(document, str):
            content = document
        else:
            content = document.get("content", "")

        if not isinstance(content, str) or not content.strip():
            errors.append(f"Документ {document_id} пустой")

    # 5. Если есть структурные ошибки, возвращаем RETRY
    if errors:
        qa_report = {
            "status": "RETRY",
            "errors": errors
        }

        logger.warning("QA: RETRY")

        return {
            "status": "RETRY",
            "loop_count": state.loop_count + 1,
            "qa_report": qa_report
        }

    # 6. Подготавливаем данные для Gemini
    review_data = {
        "plan": task_specs,
        "documents": generated_content,
        "computed_data": state.computed_data or {}
    }

    prompt = QA_PROMPT + "\n" + json.dumps(
        review_data,
        ensure_ascii=False,
        default=str
    )

    # 7. Проверяем качество с помощью Gemini
    try:
        llm = get_llm_client()

        result = llm.generate_json(prompt, QAResult)

        if result.get("status") == "RETRY":
            errors = result.get("errors", [])

        elif result.get("status") != "PASS":
            errors = ["Gemini вернул неизвестный статус проверки"]

    except Exception as e:
        logger.error(f"Ошибка проверки Gemini: {e}")
        errors = [f"Не удалось выполнить проверку Gemini: {str(e)}"]

    # 8. Формируем итоговый отчет
    if errors:
        qa_report = {
            "status": "RETRY",
            "errors": errors
        }

        logger.warning("QA: RETRY")

        return {
            "status": "RETRY",
            "loop_count": state.loop_count + 1,
            "qa_report": qa_report
        }

    qa_report = {
        "status": "PASS",
        "errors": []
    }

    logger.info("QA: PASS")

    return {
        "status": "PASS",
        "qa_report": qa_report
    }

