# agents/agent3.py
import os
import json
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_experimental.tools import PythonREPLTool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# 1. Определяем состояние графа (что агент "помнит" в процессе работы)
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    task_spec: str          # ТЗ от Агента 2
    computed_data: dict     # Итоговый результат (наш computed_data)
    error_count: int

# 2. Инициализируем LLM и инструмент для выполнения кода
# (В реальном проекте ключи лучше брать из .env, как в твоей структуре)
llm = ChatOpenAI(model="gpt-4o", temperature=0)
python_repl = PythonREPLTool()

# 3. Узел "Планировщик" (Think): решает, что делать
def planner_node(state: AgentState):
    prompt = f"""Ты — агент-вычислитель. Твоя задача — выполнить вычисления по ТЗ.
    ТЗ: {state['task_spec']}
    У тебя есть доступ к Python. Напиши ТОЛЬКО код на Python, который решает задачу.
    Если нужно построить график, сохрани его в файл и выведи путь к нему.
    Не пиши никаких объяснений, только код."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}

# 4. Узел "Исполнитель" (Act): запускает код в песочнице
def executor_node(state: AgentState):
    last_message = state["messages"][-1].content
    # Извлекаем код из ответа LLM (между ```python и ```)
    try:
        code = last_message.split("```python")[1].split("```")[0].strip()
    except IndexError:
        code = last_message # Если LLM не обернула код в блок

    try:
        # Выполняем код в безопасной среде
        result = python_repl.run(code)
        # Пытаемся сохранить результат в наше состояние
        return {"computed_data": {"status": "success", "result": result}, 
                "messages": [AIMessage(content=f"Код выполнен успешно. Результат: {result}")]}
    except Exception as e:
        # Если код упал с ошибкой, возвращаем её для следующей итерации
        return {"error_count": state.get("error_count", 0) + 1,
                "messages": [AIMessage(content=f"Ошибка выполнения: {e}")]}

# 5. Условный переход: продолжать или закончить?
def should_continue(state: AgentState):
    # Если ошибок было больше 2-х, останавливаемся (защита от бесконечного цикла)
    if state.get("error_count", 0) > 2:
        return "end"
    # Если результат уже есть и ошибок нет — заканчиваем
    if "status" in state.get("computed_data", {}) and state["computed_data"]["status"] == "success":
        return "end"
    # Иначе — возвращаемся к планировщику, чтобы он исправил код
    return "continue"

# 6. Собираем граф
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_conditional_edges("executor", should_continue, {"continue": "planner", "end": END})

agent3_app = workflow.compile()