import streamlit as st
import sys
import subprocess
import importlib
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Временная страница диагностики
st.title("DeerFlow Diagnostic Tests")

# Тест зависимостей прямо в Streamlit
if st.button("Test Dependencies"):
    st.write("Testing critical dependencies...")
    
    libraries = [
        "streamlit", "langchain", "langchain_core", "langgraph", 
        "langchain_groq", "langsmith", "dotenv"
    ]
    
    for lib in libraries:
        try:
            module = importlib.import_module(lib)
            version = getattr(module, "__version__", "unknown version")
            st.success(f"✅ {lib}: {version}")
        except ImportError:
            st.error(f"❌ {lib} not installed")

# Тест внешних инструментов
if st.button("Test External Tools"):
    st.write("Testing external tools...")
    
    # Check for uvx
    try:
        result = subprocess.run(["uvx", "--version"], 
                               capture_output=True, 
                               text=True,
                               check=False)
        if result.returncode == 0:
            st.success(f"✅ uvx found: {result.stdout.strip()}")
        else:
            st.error(f"❌ uvx found but returned an error: {result.stderr}")
    except FileNotFoundError:
        st.error("❌ uvx not found in system")
        st.info("Recommendation: Install uvx or disable its usage in configuration")

# Тест Groq API
if st.button("Test Groq API"):
    st.write("Testing Groq API connection...")
    
    # Load environment variables (API keys)
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("❌ GROQ_API_KEY not found in environment variables")
    else:
        # Try to create client and call API
        try:
            model = ChatGroq(
                api_key=api_key,
                model_name="llama3-70b-8192"
            )
            response = model.invoke("Hello, how are you?")
            st.success("✅ Groq API connection successful!")
            st.info(f"Response: {response.content}")
        except Exception as e:
            st.error(f"❌ Error connecting to Groq API: {str(e)}")

# Добавьте ссылку для возврата к обычному интерфейсу
st.markdown("---")
if st.button("Return to normal interface"):
    st.experimental_rerun()

# Обычный код закомментирован, чтобы временно не выполнялся
'''
# Оригинальный код main.py
from src.workflow import run_agent_workflow_async
import asyncio
import nest_asyncio
import json
import re

# Apply asyncio patch for Streamlit
nest_asyncio.apply()

st.title("LMX AI Agent")
st.markdown("Enter your query below:")

user_query = st.text_input("Your query", placeholder="For example: How does LLM work?")
debug_mode = st.checkbox("Debug mode")
enable_background_investigation = st.checkbox("Background investigation", value=False)  # Отключаем по умолчанию для отладки

if st.button("Run agent"):
    if user_query.strip():
        with st.spinner("Agent is thinking..."):
            try:
                # Run async function through asyncio.run
                result = asyncio.run(run_agent_workflow_async(
                    user_input=user_query,
                    debug=debug_mode,
                    enable_background_investigation=enable_background_investigation
                ))
                
                if result is None:
                    st.error("⚠️ Agent returned empty result. Try another query.")
                else:
                    st.success("✅ Answer:")
                    
                    # Функция для извлечения содержимого из сложного ответа
                    def extract_content(data):
                        # Если это словарь, ищем поле content
                        if isinstance(data, dict):
                            if "content" in data:
                                # Проверяем, является ли content строкой в формате JSON
                                try:
                                    content_json = json.loads(data["content"])
                                    if isinstance(content_json, dict) and "thought" in content_json:
                                        return content_json["thought"]
                                except:
                                    pass
                                return data["content"]
                            elif "thought" in data:
                                return data["thought"]
                        
                        # Если это строка
                        if isinstance(data, str):
                            # Проверяем, является ли это строкой в формате JSON
                            try:
                                json_data = json.loads(data)
                                if isinstance(json_data, dict):
                                    if "thought" in json_data:
                                        return json_data["thought"]
                                    elif "content" in json_data:
                                        return json_data["content"]
                            except:
                                pass
                                
                            # Пытаемся извлечь JSON из строки
                            content_match = re.search(r'"content":\s*"([^"]+)"', data)
                            if content_match:
                                return content_match.group(1)
                            
                            # Пытаемся извлечь thought из строки
                            thought_match = re.search(r'"thought":\s*"([^"]+)"', data)
                            if thought_match:
                                return thought_match.group(1)
                            
                            # Возвращаем строку как есть, если нет специальных полей
                            return data
                        
                        # Для других типов данных
                        return str(data)
                    
                    # Проверка структуры ответа
                    if isinstance(result, dict):
                        # Вариант 1: Использовать последнее сообщение из списка messages
                        if "messages" in result and result["messages"]:
                            last_message = result["messages"][-1]
                            content = extract_content(last_message)
                            st.markdown(content)
                        # Вариант 2: Использовать поле response если оно есть
                        elif "response" in result:
                            content = extract_content(result["response"])
                            st.markdown(content)
                        # Вариант 3: Показать полный результат, если структура неизвестна
                        else:
                            st.info("Raw response format:")
                            st.json(result)
                    else:
                        # Если результат не словарь, показываем как есть
                        content = extract_content(result)
                        st.markdown(content)
                    
                    # В режиме отладки показываем полный ответ
                    if debug_mode:
                        with st.expander("Debug Response"):
                            st.json(result)
            
            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")
                if debug_mode:
                    st.exception(e)
    else:
        st.warning("⚠️ Enter a query before running.")
'''
