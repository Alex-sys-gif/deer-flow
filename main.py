import streamlit as st
from src.workflow import run_agent_workflow_async
import asyncio
import nest_asyncio
import json

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
                    
                    # Проверка структуры ответа
                    if isinstance(result, dict):
                        # Вариант 1: Использовать последнее сообщение из списка messages
                        if "messages" in result and result["messages"]:
                            last_message = result["messages"][-1]
                            if isinstance(last_message, dict) and "content" in last_message:
                                st.markdown(last_message["content"])
                            else:
                                st.markdown(str(last_message))
                        # Вариант 2: Использовать поле response если оно есть
                        elif "response" in result:
                            st.markdown(result["response"])
                        # Вариант 3: Показать полный результат, если структура неизвестна
                        else:
                            st.json(result)
                    else:
                        # Если результат не словарь, показываем как есть
                        st.markdown(str(result))
                    
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
