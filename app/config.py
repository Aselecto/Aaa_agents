# Все конфигурации (модели, лимиты, пути)
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict
import logging
import sys

class Settings(BaseSettings):
    LLM_PROVIDER: str = "gemini" 
    
    GEMINI_API_KEY: str = Field(
        default="mock_key", 
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY")
    )
    
    GEMINI_MODEL: str = "gemini-1.5-flash"
    MAX_QA_LOOPS: int = 2 
    MAX_OUTPUT_TOKENS: int = 1024 

    # Полная конфигурация с защитой от пустых строк
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True  # Игнорирует пустые переменные типа KEY=, переходя к алиасам или дефолтам
    )

settings = Settings()


# Настройка единого формата логов для всей команды
logging.basicConfig(
    level=logging.INFO, # Во время разработки можно поменять на DEBUG
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout), # Вывод в консоль uvicorn
        logging.FileHandler("app.log", encoding="utf-8") # Дублирование в файл для дебага
    ]
)

# Функция для получения логгера в файлы агентов
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
