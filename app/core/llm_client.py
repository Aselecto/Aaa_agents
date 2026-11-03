# Базовый абстрактный класс для LLM
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseLLMClient(ABC):
    @abstractmethod
    def generate_json(self, prompt: str, schema: Any) -> Dict[str, Any]:
        """Абстрактный метод для гарантированного получения JSON"""
        pass
