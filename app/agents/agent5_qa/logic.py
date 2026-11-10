# Код Агента 5 — Проверка качества

from typing import Dict, Any
import logging

from app.core.state import ProjectState

logger = logging.getLogger("Agent5_QA")


def run_agent_5(state: ProjectState) -> Dict[str, Any]:

    errors = []

    # 1. Проверяем план Агента 2
    task_specs = state.task_specifications or {}

    if not task_specs.get("ok"):
        errors.append("План Агента 2 отсутствует или содержит ошибку")

    # 2. Проверяем наличие результатов Агента 4
    generated_content = state.generated_content or {}

    if not generated_content:
        errors.append("Агент 4 не создал документ")

    # 3. Проверяем каждый документ
    deliverables = task_specs.get("deliverables", [])

    for deliverable in deliverables:

        document_id = deliverable.get("id")

        if not document_id:
            errors.append("У документа отсутствует ID")
            continue

        document = generated_content.get(document_id)

        if not document:
            errors.append(
                f"Документ {document_id} отсутствует"
            )
            continue

        content = document.get("content", "")

        if not content.strip():
            errors.append(
                f"Документ {document_id} пустой"
            )

    # 4. Определяем результат проверки
    if errors:

        for error in errors:
            logger.warning(error)

        logger.warning("QA: RETRY")

        return {
            "status": "RETRY",
            "loop_count": state.loop_count + 1
        }

    logger.info("QA: PASS")

    return {
        "status": "PASS"
    }
