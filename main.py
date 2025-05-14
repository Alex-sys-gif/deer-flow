import streamlit as st
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
