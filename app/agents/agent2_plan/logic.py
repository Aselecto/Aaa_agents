# Код Агента 2
"""
### Агент 2 («Архитектор») — узел LangGraph

Читает `state.project_passport` (его пишет А1), составляет план аналитического реферата
и записывает его в `state.task_specifications`.

* А3 берёт из плана `deliverables[*].compute_tasks`;
* А4 берёт `deliverables[*].document`;
* А5 берёт `qa_checks`, `warnings` и `errors`.

Агент НИКОГДА не бросает исключения в граф: любой сбой превращается в статус FAIL
и понятное описание в `task_specifications["errors"]`. Подробности — в CONTRACT.md.
"""

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from app.agents.agent2_plan.prompts import FIX_REQUEST_TEMPLATE, SYSTEM_PROMPT
from app.agents.agent2_plan.schemas import REQUESTED_PASSPORT_FIELDS, Agent2Plan, ProjectPassport
from app.agents.agent2_plan.validation import find_problems, repair_plan
from app.config import get_logger
from app.core.state import ProjectState
from app.services.llm import LLMConfigError, get_llm

logger = get_logger("Agent2_Plan")

# --------------------------------------------------------------------------- #
# Параметры. Их нужно подобрать на практике (см. процесс.txt).
# --------------------------------------------------------------------------- #

PLANNER_MODEL: Optional[str] = None  # None — взять settings.GEMINI_MODEL; можно указать другую модель только для А2
PLANNER_TEMPERATURE = 0.2
PLANNER_MAX_OUTPUT_TOKENS = 8192  # план длинный: стандартных 1024 токенов из config.py не хватит
MAX_SELF_FIXES = 2  # сколько раз модель может переделать план. Не путать с loop_count из QA
MAX_FEEDBACK_TOKENS = 1500  # лимит на замечания А5 в промпте (в токенах, а не в символах)
CHARS_PER_TOKEN = 3  # грубая оценка для русского текста и JSON

# Статусы такие же, как в коде А1 и main.py (заглавными буквами)
STATUS_PLAN = "PLAN"  # при этом статусе работает А2 (его ставит А1)
STATUS_COMPUTE = "COMPUTE"  # статус после успешной работы А2
STATUS_FAIL = "FAIL"  # критическая ошибка

PARSE_ERRORS = (ValidationError, OutputParserException)  # модель вернула мусор вместо схемы


class ErrorCode:
    """Коды ошибок: по ним А5 и интерфейс понимают, что произошло."""

    UPSTREAM_FAILED = "UPSTREAM_FAILED"  # А1 уже завершился ошибкой
    WRONG_STATUS = "WRONG_STATUS"
    NO_PASSPORT = "NO_PASSPORT"
    INVALID_PASSPORT = "INVALID_PASSPORT"
    LLM_CONFIG = "LLM_CONFIG"  # не настроен ключ API
    PLAN_INVALID = "PLAN_INVALID"  # модель так и не составила корректный план
    UNEXPECTED_ERROR = "UNEXPECTED_ERROR"  # сеть, лимиты API и прочие сбои


# --------------------------------------------------------------------------- #
# Формирование результата для графа
# --------------------------------------------------------------------------- #


def _failure_patch(code: str, errors: List[str]) -> Dict[str, Any]:
    """### Description
    Результат при ошибке: статус FAIL и описание ошибок в `task_specifications`,
    где их увидят А5 и пользователь.
    ### Args
    * `code` (str): код ошибки из `ErrorCode`.
    * `errors` (List[str]): первое сообщение понятно пользователю, остальные — подробности.
    ### Returns
    * `Dict[str, Any]`: изменения для `ProjectState`.
    """
    logger.error(f"Агент 2 завершился с ошибкой {code}: {errors}")
    return {
        "status": STATUS_FAIL,
        "task_specifications": {"ok": False, "error_code": code, "errors": errors, "warnings": []},
    }


def _success_patch(plan: Agent2Plan, extra_warnings: List[str]) -> Dict[str, Any]:
    """### Description
    Результат при успехе: план и статус COMPUTE.
    ### Args
    * `plan` (Agent2Plan): готовый проверенный план.
    * `extra_warnings` (List[str]): предупреждения, найденные кодом (правки плана, пустые поля паспорта).
    ### Returns
    * `Dict[str, Any]`: изменения для `ProjectState`.
    """
    specs = plan.model_dump()
    specs["warnings"] = list(dict.fromkeys(plan.warnings + extra_warnings))  # без повторов, порядок сохраняется
    logger.info(f"Агент 2: ТЗ создано, документов: {len(plan.deliverables)}, предупреждений: {len(specs['warnings'])}")
    return {
        "status": STATUS_COMPUTE,
        "task_specifications": {"ok": True, "error_code": None, "errors": [], **specs},
    }


# --------------------------------------------------------------------------- #
# Проверка входа и подготовка запроса к модели
# --------------------------------------------------------------------------- #


def _check_state(state: ProjectState) -> Optional[Dict[str, Any]]:
    """### Description
    Проверяет, можно ли работать: статус, наличие паспорта.
    ### Args
    * `state` (ProjectState): состояние проекта.
    ### Returns
    * `Optional[Dict[str, Any]]`: результат-ошибка, если работать нельзя, иначе `None`.
    """
    status = str(state.status).upper()
    if status == STATUS_FAIL:
        return _failure_patch(
            ErrorCode.UPSTREAM_FAILED, ["Анализ исходных данных не удался, поэтому план не составлялся."]
        )
    if status != STATUS_PLAN:
        return _failure_patch(
            ErrorCode.WRONG_STATUS, [f"Агент 2 запускается при статусе {STATUS_PLAN}, а сейчас {state.status!r}."]
        )
    if not state.project_passport:
        return _failure_patch(ErrorCode.NO_PASSPORT, ["Агент 1 не передал паспорт проекта."])
    return None


def _passport_warnings(passport: ProjectPassport) -> List[str]:
    """Предупреждения о полях паспорта, которые А2 запросил у А1, но не получил."""
    return [
        f"В паспорте нет поля '{name}' ({reason}): план составлен без него."
        for name, reason in REQUESTED_PASSPORT_FIELDS.items()
        if not getattr(passport, name)
    ]


def _estimate_tokens(text: str) -> int:
    """Приблизительное число токенов в тексте (точный подсчёт требует обращения к API)."""
    return len(text) // CHARS_PER_TOKEN + 1


def _build_feedback_text(state: ProjectState) -> str:
    """### Description
    Готовит текст для повторного запуска после проверки качества: прошлый план и замечания А5 к плану.
    Лог фильтруется средствами Python (записи с `target_agent == "A2"`) и обрезается по токенам
    целыми записями, чтобы не ломать JSON. Работает, когда в состоянии появится поле `qa_log` (список записей).
    ### Args
    * `state` (ProjectState): состояние проекта.
    ### Returns
    * `str`: текст для промпта или пустая строка, если замечаний нет.
    """
    qa_log = getattr(state, "qa_log", None) or []
    records = [item for item in qa_log if isinstance(item, dict) and item.get("target_agent") == "A2"]
    if not records or not state.task_specifications:
        return ""

    while records and _estimate_tokens(json.dumps(records, ensure_ascii=False)) > MAX_FEEDBACK_TOKENS:
        records.pop(0)  # выбрасываем самые старые записи целиком
    if not records:
        logger.warning("Замечания А5 не влезли в лимит токенов и не будут учтены")
        return ""

    return (
        "\n\nЭто ПОВТОРНЫЙ запуск после проверки качества.\n"
        f"Предыдущий план:\n{json.dumps(state.task_specifications, ensure_ascii=False)}\n\n"
        f"Замечания А5 к плану (исправь их):\n{json.dumps(records, ensure_ascii=False)}"
    )


def _build_messages(state: ProjectState) -> List[Tuple[str, str]]:
    """Собирает сообщения для модели: системный промпт и паспорт проекта (целиком, как JSON)."""
    passport_json = json.dumps(state.project_passport, ensure_ascii=False, indent=2)
    human_text = f"Паспорт проекта (JSON):\n{passport_json}{_build_feedback_text(state)}"
    return [("system", SYSTEM_PROMPT), ("human", human_text)]


# --------------------------------------------------------------------------- #
# Обращение к модели с самоисправлением
# --------------------------------------------------------------------------- #


async def _plan_with_self_fixes(
    planner: Any, messages: List[Tuple[str, str]]
) -> Tuple[Optional[Agent2Plan], List[str], List[str]]:
    """### Description
    Просит модель составить план. Если в плане остались проблемы, возвращает их модели на исправление
    (не больше `MAX_SELF_FIXES` раз). Что можно починить кодом, чинится сразу, без обращения к модели.
    ### Args
    * `planner` (Any): модель со `with_structured_output(Agent2Plan)`.
    * `messages` (List[Tuple[str, str]]): сообщения для модели (дополняются на месте).
    ### Returns
    * `Tuple[Optional[Agent2Plan], List[str], List[str]]`: план (или `None`), предупреждения об автоправках,
      оставшиеся проблемы.
    """
    problems: List[str] = []
    for attempt in range(1, MAX_SELF_FIXES + 2):
        plan: Optional[Agent2Plan] = None
        try:
            result = await planner.ainvoke(messages)
        except PARSE_ERRORS as exc:
            problems = [f"Ответ не удалось разобрать по схеме: {str(exc)[:500]}"]
        else:
            if isinstance(result, Agent2Plan):
                plan = result
                repair_warnings = repair_plan(plan)
                problems = find_problems(plan)
                if not problems:
                    return plan, repair_warnings, []
            else:
                problems = ["Модель не вернула план в нужной структуре."]

        logger.warning(f"Попытка {attempt}: проблем в плане — {len(problems)}")
        fix_request = FIX_REQUEST_TEMPLATE.format(problems="\n- ".join(problems))
        if plan is not None:
            messages.append(("ai", plan.model_dump_json()))
            messages.append(("human", fix_request))
        else:  # ответа нет: дописываем просьбу к последнему сообщению, чтобы роли не повторялись подряд
            role, text = messages[-1]
            messages[-1] = (role, f"{text}\n\n{fix_request}")
    return None, [], problems


# --------------------------------------------------------------------------- #
# Узел LangGraph
# --------------------------------------------------------------------------- #


def get_planner() -> Any:
    """### Description
    Создаёт модель, которая сразу возвращает объект `Agent2Plan`, а не текст с JSON внутри.
    ### Returns
    * `Any`: runnable LangChain.
    ### Raises
    * `LLMConfigError`: если не настроен ключ API.
    """
    llm = get_llm(model=PLANNER_MODEL, temperature=PLANNER_TEMPERATURE, max_output_tokens=PLANNER_MAX_OUTPUT_TOKENS)
    return llm.with_structured_output(Agent2Plan)


async def plan_project(state: ProjectState, planner_factory: Callable[[], Any]) -> Dict[str, Any]:
    """### Description
    Ядро агента. Отделено от `run_agent_2`, чтобы в тестах подставлять модель-заглушку.
    ### Args
    * `state` (ProjectState): состояние проекта (нужны `status` и `project_passport`).
    * `planner_factory` (Callable): функция, создающая модель (в тестах — заглушку).
    ### Returns
    * `Dict[str, Any]`: изменения для `ProjectState`: `status` и `task_specifications`.
    """
    logger.info(f"Агент 2 запущен для проекта {state.project_id}")

    early_failure = _check_state(state)
    if early_failure:
        return early_failure

    try:
        passport = ProjectPassport.model_validate(state.project_passport)
    except ValidationError as exc:
        logger.debug(f"Паспорт не прошёл проверку: {exc}")
        return _failure_patch(ErrorCode.INVALID_PASSPORT, ["Паспорт проекта от Агента 1 имеет неверный формат."])

    try:
        planner = planner_factory()
    except LLMConfigError as exc:
        logger.error(f"Модель не настроена: {exc}")
        return _failure_patch(ErrorCode.LLM_CONFIG, ["Сервис языковой модели не настроен. Обратитесь к администратору."])

    try:
        messages = _build_messages(state)
        plan, repair_warnings, problems = await _plan_with_self_fixes(planner, messages)
    except Exception:  # сеть, лимиты API, неверный ключ: граф падать не должен
        logger.exception("Агент 2: сбой при составлении плана")
        return _failure_patch(
            ErrorCode.UNEXPECTED_ERROR, ["Не удалось составить план из-за сбоя сервиса. Повторите запрос позже."]
        )

    if plan is None:
        return _failure_patch(ErrorCode.PLAN_INVALID, ["Не удалось составить корректный план проекта.", *problems])

    return _success_patch(plan, _passport_warnings(passport) + repair_warnings)


async def run_agent_2(state: ProjectState) -> Dict[str, Any]:
    """### Description
    Агент 2 (Архитектор): точка входа для LangGraph, `workflow.add_node("agent_2", run_agent_2)`.
    ### Args
    * `state` (ProjectState): текущее единое состояние проекта.
    ### Returns
    * `Dict[str, Any]`: `status` (COMPUTE или FAIL) и `task_specifications` (план или описание ошибки).
    """
    return await plan_project(state, get_planner)
