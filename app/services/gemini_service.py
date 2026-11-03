# Персональная реализация для Google Gemini
import json
import logging
import re
from typing import Dict, Any, Type
from pydantic import BaseModel, ValidationError
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.exceptions import OutputParserException
from app.core.llm_client import BaseLLMClient

logger = logging.getLogger(__name__)

class LLMBaseException(Exception): """Базовое исключение сервиса"""
class LLMValidationError(LLMBaseException): """Ошибка валидации схемы"""
class LLMAPIError(LLMBaseException): """Ошибка обращения к API (сети, ключи)"""

class GeminiServiceClient(BaseLLMClient):
    def __init__(self):
        from app.config import settings
        
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.2,
            max_output_tokens=settings.MAX_OUTPUT_TOKENS
        )

    def generate_json(self, prompt: str, schema: Type[BaseModel]) -> Dict[str, Any]:
        """Запрос к Gemini с нативной поддержкой структурированного вывода и валидацией."""
        try:
            structured_llm = self.llm.with_structured_output(schema)
            result = structured_llm.invoke([HumanMessage(content=prompt)])
            
            if result is None:
                raise LLMAPIError("Gemini вернул пустой ответ. Возможно, сработали фильтры безопасности.")
            
            if isinstance(result, BaseModel):
                return result.model_dump()
                
            raise OutputParserException(f"Модель вернула неожиданный тип данных: {type(result)}")

        except (ValidationError, OutputParserException) as err:
            logger.warning(f"Основной режим разметки дал сбой, запуск фолбека. Ошибка: {err}")
            return self._fallback_generate(prompt, schema)
                
        except Exception as global_err:
            logger.error(f"Критическая ошибка API Gemini: {global_err}", exc_info=True)
            raise LLMAPIError(f"Ошибка связи с API: {str(global_err)}")

    def _fallback_generate(self, prompt: str, schema: Type[BaseModel]) -> Dict[str, Any]:
        """Резервный метод генерации через обычный текст + ручной парсинг"""
        try:
            fallback_prompt = prompt + "\nОтветь СТРОГО в формате JSON. Не используй markdown разметку (```json)."
            raw_response = self.llm.invoke([HumanMessage(content=fallback_prompt)])
            
            if raw_response is None or not raw_response.content:
                raise LLMAPIError("Модель вернула пустой ответ в режиме фолбека.")
                
            response_text = str(raw_response.content).strip()
            parsed_json = self._clean_and_parse_json(response_text)
            
            validated_data = schema.model_validate(parsed_json)
            return validated_data.model_dump()
            
        except ValidationError as val_err:
            logger.error(f"Фолбек не прошёл валидацию Pydantic: {val_err.errors()}")
            raise LLMValidationError(f"Текст модели не соответствует схеме: {val_err.errors()}")
        except LLMBaseException:
            # Пробрасываем наши кастомные ошибки (например, LLMValidationError из парсера) дальше
            raise
        except Exception as fallback_error:
            # Любые сетевые / API ошибки, упавшие во время self.llm.invoke в фолбеке
            logger.critical(f"Критический сбой API во время фолбека: {fallback_error}", exc_info=True)
            raise LLMAPIError(f"Ошибка связи с API при фолбеке: {str(fallback_error)}")

    def _clean_and_parse_json(self, text: str) -> Dict[str, Any]:
        """Вспомогательная функция очистки текста от markdown-разметки"""
        text = text.strip()
        
        # Более надежное удаление ```json ... ``` в любой части текста с помощью регулярки
        # Ищет блоки json, даже если модель написала текст до или после него
        json_block_match = re.search(r"```json\s*([\s\(\S\)]*?)\s*```|```\s*([\s\(\S\)]*?)\s*```", text)
        if json_block_match:
            text = json_block_match.group(1) or json_block_match.group(2)
            text = text.strip()
            
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Фолбек на поиск первой { и последней }
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end+1])
                except json.JSONDecodeError:
                    pass
            raise LLMValidationError(f"Не удалось извлечь валидный JSON из текста. Ошибка: {e}")

def get_llm_client() -> BaseLLMClient:
    from app.config import settings
    if settings.LLM_PROVIDER == "gemini":
        return GeminiServiceClient()
    raise ValueError(f"Unsupported LLM Provider: {settings.LLM_PROVIDER}")
