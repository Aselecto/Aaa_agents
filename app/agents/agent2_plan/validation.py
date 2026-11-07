"""
### Проверка плана Агента 2 обычным кодом (без LLM)

Работает в два шага:
1. `repair_plan` — сразу чинит то, что можно починить кодом (дубликаты id, битые ссылки,
   порядок разделов) и возвращает список предупреждений о сделанных правках.
2. `find_problems` — находит то, что код починить не может. Эти проблемы возвращаются модели
   на исправление (см. `logic.py`).
"""

from typing import List, Protocol, Set, TypeVar

from app.agents.agent2_plan.schemas import REFERAT_SECTIONS, Agent2Plan, Deliverable

MAX_DELIVERABLES = 3  # больше файлов — просим модель объединить информацию


class _HasId(Protocol):
    id: str


Item = TypeVar("Item", bound=_HasId)


# --------------------------------------------------------------------------- #
# Шаг 1. Автоисправление
# --------------------------------------------------------------------------- #


def _keep_first_of_each_id(items: List[Item], seen_ids: Set[str], label: str, warnings: List[str]) -> List[Item]:
    """### Description
    Оставляет первый элемент с каждым id, лишние отсекает и пишет предупреждение.
    ### Args
    * `items` (List[Item]): элементы с полем `id`.
    * `seen_ids` (Set[str]): уже занятые id (общий набор позволяет проверять уникальность сразу по нескольким спискам).
    * `label` (str): как назвать элемент в предупреждении, например «артефакта».
    * `warnings` (List[str]): сюда добавляются предупреждения.
    ### Returns
    * `List[Item]`: список без повторяющихся id.
    """
    kept: List[Item] = []
    for item in items:
        if item.id in seen_ids:
            warnings.append(f"Повторяющийся id {label} '{item.id}': лишний элемент удалён.")
            continue
        seen_ids.add(item.id)
        kept.append(item)
    return kept


def _repair_document(deliverable: Deliverable, warnings: List[str]) -> None:
    """Убирает дубли разделов, расставляет разделы по порядку, удаляет ссылки на несуществующие артефакты."""
    sections = _keep_first_of_each_id(deliverable.document.sections, set(), "раздела", warnings)
    order = list(REFERAT_SECTIONS)
    deliverable.document.sections = sorted(sections, key=lambda section: order.index(section.id))

    known_artifacts = set(deliverable.artifact_ids())
    for section in deliverable.document.sections:
        unknown = [name for name in section.uses_artifacts if name not in known_artifacts]
        if unknown:
            section.uses_artifacts = [name for name in section.uses_artifacts if name in known_artifacts]
            warnings.append(f"Раздел '{section.id}' ссылался на несуществующие артефакты {unknown}: ссылки удалены.")


def repair_plan(plan: Agent2Plan) -> List[str]:
    """### Description
    Исправляет план на месте: дубликаты id, ссылки на несуществующие артефакты, порядок разделов.
    ### Args
    * `plan` (Agent2Plan): план от модели (изменяется на месте).
    ### Returns
    * `List[str]`: предупреждения о том, что было исправлено (пусто, если править было нечего).
    """
    warnings: List[str] = []
    plan.deliverables = _keep_first_of_each_id(plan.deliverables, set(), "файла", warnings)

    task_ids: Set[str] = set()  # id заданий и артефактов должны быть уникальны во всём плане
    artifact_ids: Set[str] = set()
    for deliverable in plan.deliverables:
        deliverable.compute_tasks = _keep_first_of_each_id(deliverable.compute_tasks, task_ids, "задания", warnings)
        for task in deliverable.compute_tasks:
            task.outputs = _keep_first_of_each_id(task.outputs, artifact_ids, "артефакта", warnings)
        _repair_document(deliverable, warnings)
    return warnings


# --------------------------------------------------------------------------- #
# Шаг 2. Поиск проблем, которые не чинятся кодом
# --------------------------------------------------------------------------- #


def _find_problems_in_deliverable(deliverable: Deliverable) -> List[str]:
    """Проверяет один документ: все ли разделы на месте и достаточно ли подробны задания для А3."""
    name = f"Документ '{deliverable.id}'"
    problems: List[str] = []

    present = {section.id for section in deliverable.document.sections}
    missing = [section_id for section_id in REFERAT_SECTIONS if section_id not in present]
    if missing:
        problems.append(f"{name}: не хватает разделов {missing}.")

    for section in deliverable.document.sections:
        if section.target_words <= 0:
            problems.append(f"{name}: у раздела '{section.id}' не задан объём в словах.")

    for task in deliverable.compute_tasks:
        if not task.inputs:
            problems.append(f"{name}: у задания '{task.id}' не указаны входные данные.")
        if not task.outputs:
            problems.append(f"{name}: у задания '{task.id}' нет выходных артефактов.")
        if not task.acceptance_criteria:
            problems.append(f"{name}: у задания '{task.id}' нет критериев приёмки.")
    return problems


def find_problems(plan: Agent2Plan) -> List[str]:
    """### Description
    Ищет в плане проблемы, которые нельзя исправить кодом.
    ### Args
    * `plan` (Agent2Plan): план (лучше после `repair_plan`).
    ### Returns
    * `List[str]`: описания проблем; пустой список означает, что план годен.
    """
    if not plan.deliverables:
        return ["Список deliverables пуст: нужен минимум один документ — аналитический реферат."]

    problems: List[str] = []
    if len(plan.deliverables) > MAX_DELIVERABLES:
        problems.append(
            f"Слишком много файлов ({len(plan.deliverables)}), максимум {MAX_DELIVERABLES}. "
            "Объедини информацию в меньшее число файлов."
        )
    for deliverable in plan.deliverables:
        problems += _find_problems_in_deliverable(deliverable)
    return problems
