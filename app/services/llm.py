"""
### Фабрика языковой модели (Gemini через LangChain)

Нужна агентам, которым важны async-вызовы (`ainvoke`) и свои лимиты на токены.
Модель и ключ берутся из `app.config.settings`, параметры можно переопределить при вызове.
Неверные значения (пустая строка, текст вместо числа, выход за допустимый диапазон) не ломают код:
вместо них берётся значение по умолчанию, а в лог пишется предупреждение.
"""

from typing import Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_logger, settings

logger = get_logger("LLM")

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TIMEOUT_SECONDS = 90.0
MAX_RETRIES = 2  # повторы запроса при сетевых сбоях
MOCK_API_KEY = "mock_key"  # значение по умолчанию в config.py: ключ не настроен


class LLMConfigError(Exception):
    """Модель не настроена: в .env нет ключа API."""


def _clean_number(value: Any, default: float, low: float, high: float, name: str) -> float:
    """### Description
    Превращает значение в число из диапазона [low, high] или возвращает значение по умолчанию.
    ### Args
    * `value` (Any): что передали (None, пустая строка, текст, число).
    * `default` (float): запасное значение.
    * `low` (float), `high` (float): допустимые границы.
    * `name` (str): имя параметра для сообщения в логе.
    ### Returns
    * `float`: корректное число.
    """
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        logger.warning(f"{name}={value!r} не число, используется значение по умолчанию {default}")
        return default
    if not low <= number <= high:  # условие ложно и для nan
        logger.warning(f"{name}={number} вне диапазона [{low}, {high}], используется {default}")
        return default
    return number


def _clean_model_name(model: Optional[str]) -> str:
    """Берёт непустое имя модели: из аргумента, затем из settings, затем DEFAULT_MODEL."""
    for candidate in (model, settings.GEMINI_MODEL):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    logger.warning(f"Имя модели не задано, используется {DEFAULT_MODEL}")
    return DEFAULT_MODEL


def get_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
) -> ChatGoogleGenerativeAI:
    """### Description
    Создаёт клиент Gemini. Любой аргумент можно не указывать — тогда берётся значение из настроек.
    ### Args
    * `model` (Optional[str]): имя модели; по умолчанию `settings.GEMINI_MODEL`.
    * `temperature` (Optional[float]): «творческость» от 0 до 2; по умолчанию 0.2.
    * `max_output_tokens` (Optional[int]): лимит токенов ответа; по умолчанию `settings.MAX_OUTPUT_TOKENS`.
    * `timeout` (Optional[float]): секунд на один запрос.
    ### Returns
    * `ChatGoogleGenerativeAI`: готовый клиент.
    ### Raises
    * `LLMConfigError`: если ключ API не задан.
    """
    api_key = settings.GEMINI_API_KEY
    if not api_key or api_key == MOCK_API_KEY:
        raise LLMConfigError("Не задан GEMINI_API_KEY (или GOOGLE_API_KEY) в .env")

    return ChatGoogleGenerativeAI(
        model=_clean_model_name(model),
        google_api_key=api_key,
        temperature=_clean_number(temperature, DEFAULT_TEMPERATURE, 0.0, 2.0, "temperature"),
        max_output_tokens=int(
            _clean_number(max_output_tokens, settings.MAX_OUTPUT_TOKENS, 1, 65536, "max_output_tokens")
        ),
        timeout=_clean_number(timeout, DEFAULT_TIMEOUT_SECONDS, 1.0, 600.0, "timeout"),
        max_retries=MAX_RETRIES,
    )
