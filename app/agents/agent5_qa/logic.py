
# Код Агента 5 — Проверка качества

from typing import Dict, Any
import logging

from app.core.state import ProjectState

logger = logging.getLogger("Agent5_QA")


def run_agent_5(state: ProjectState) -> Dict[str, Any]:

    errors = []

    # Проверяем план Агента 2
    task_specs = state.task_specifications or {}

    if not task_specs.get("ok"):
        errors.append("План Агента 2 отсутствует или содержит ошибку")

    # Проверяем документы Агента 4
    generated_content = state.generated_content or {}

    if not generated_content:
        errors.append("Агент 4 не создал документ")

    # Проверяем каждый документ
    deliverables = task_specs.get("deliverables", [])

    for deliverable in deliverables:

        document_id = deliverable.get("id")

        if not document_id:
            errors.append("У документа отсутствует ID")
            continue

        document = generated_content.get(document_id)

        if not document:
            errors.append(f"Документ {document_id} отсутствует")
            continue

        content = document.get("content", "")

        if not content.strip():
            errors.append(f"Документ {document_id} пустой")

    # Формируем отчет QA
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

