"""
### Схемы данных Агента 2 («Архитектор»)

* `ProjectPassport` — что А2 читает из `state.project_passport` (пишет А1).
* `Agent2Plan` — что А2 пишет в `state.task_specifications` (читают А3, А4, А5).

План устроен как дерево: каждый итоговый документ (`Deliverable`) содержит
ТЗ для А3 (`compute_tasks`) и ТЗ для А4 (`document`).
Схемы лежат рядом с агентом, чтобы не конфликтовать в Git с общим `app/schemas.py`.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Формат аналитического реферата (согласован с руководителем)
# --------------------------------------------------------------------------- #

SectionId = Literal["intro", "analysis", "critique", "conclusion"]

# Порядок и названия разделов. Из этого словаря строятся и промпт, и проверки.
REFERAT_SECTIONS: Dict[str, str] = {
    "intro": "Вводная часть (аннотация, постановка проблемы)",
    "analysis": "Основная аналитическая часть (сравнение и сопоставление)",
    "critique": "Критика и ограничения (валидация данных)",
    "conclusion": "Выводы и заключение",
}

# --------------------------------------------------------------------------- #
# Вход: паспорт проекта от Агента 1
# --------------------------------------------------------------------------- #

# Поля, которые А2 просит добавить в паспорт А1 (пока А1 их не выдаёт).
# Ключ — имя поля, значение — зачем оно нужно Агенту 2.
REQUESTED_PASSPORT_FIELDS: Dict[str, str] = {
    "topic": "тема реферата (раздел intro)",
    "research_questions": "вопросы, на которые отвечает раздел conclusion",
    "sources": "перечень источников и данных (раздел intro)",
    "numeric_data_summary": "какие числовые данные есть у пользователя (задания для А3)",
}


class ProjectPassport(BaseModel):
    """Паспорт проекта. Первые три поля гарантирует А1, остальные А2 запросил у А1."""

    content_type: str = "general"
    domain: str = ""
    key_entities: List[str] = Field(default_factory=list)

    topic: Optional[str] = None
    research_questions: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    numeric_data_summary: Optional[str] = None


# --------------------------------------------------------------------------- #
# Выход: план (ТЗ для А3, А4 и подсказки для А5)
# --------------------------------------------------------------------------- #


class Artifact(BaseModel):
    """Результат работы А3: таблица, график или набор метрик."""

    id: str = Field(description="Уникальный во всём плане id, snake_case латиницей, например 'chart_market_share'")
    kind: Literal["table", "chart", "metrics"]
    format: Literal["xlsx", "csv", "png", "svg", "json"]
    title: str = Field(description="Подпись в документе, например 'Доли рынка по источникам'")
    description: str = Field(description="Что именно должно быть в таблице или на графике")


class ComputeTask(BaseModel):
    """ТЗ для А3: одно вычислительное задание."""

    id: str
    goal: str = Field(description="Что нужно посчитать или построить")
    inputs: List[str] = Field(description="Какие данные использовать: фрагменты raw_input или сущности паспорта")
    method: str = Field(description="Формулы и алгоритм: проценты, средние, темпы роста, доли и т.п.")
    outputs: List[Artifact]
    external_apis: List[str] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(description="Проверяемые критерии готовности")


class SectionSpec(BaseModel):
    """ТЗ для А4: один раздел документа."""

    id: SectionId
    title: str
    purpose: str = Field(description="Что раздел должен донести читателю")
    uses_artifacts: List[str] = Field(
        default_factory=list,
        description="id артефактов А3, которые используются в разделе. Пусто — раздел не ждёт вычислений",
    )
    target_words: int = Field(description="Желаемый объём раздела в словах")


class DocumentSpec(BaseModel):
    """ТЗ для А4: скелет одного документа."""

    audience: str
    tone_of_voice: str
    sections: List[SectionSpec]


class Deliverable(BaseModel):
    """Один итоговый документ пакета вместе с ТЗ для А3 и А4."""

    id: str
    title: str
    format: Literal["pdf", "html", "md"]
    document: DocumentSpec
    compute_tasks: List[ComputeTask] = Field(default_factory=list)

    def artifact_ids(self) -> List[str]:
        """### Returns
        * `List[str]`: id всех артефактов А3, которые создаются для этого документа.
        """
        return [artifact.id for task in self.compute_tasks for artifact in task.outputs]


class Agent2Plan(BaseModel):
    """Весь план Агента 2 — именно его возвращает модель."""

    deliverables: List[Deliverable]
    qa_checks: List[str] = Field(default_factory=list, description="Проверки для А5")
    warnings: List[str] = Field(default_factory=list, description="Чего не хватило для хорошего плана")
