import streamlit as st
from src.workflow import run_agent_workflow_async
import asyncio
import nest_asyncio

# Применяем патч для asyncio в Streamlit
nest_asyncio.apply()

st.title("DeerFlow - AI Агент")
st.markdown("Введите запрос ниже:")

user_query = st.text_input("Ваш запрос", placeholder="Например: Как работает LLM?")
debug_mode = st.checkbox("Режим отладки")
enable_background_investigation = st.checkbox("Фоновое исследование", value=True)

if st.button("Запустить агента"):
    if user_query.strip():
        with st.spinner("Агент думает..."):
            # Запуск асинхронной функции через asyncio.run
            result = asyncio.run(run_agent_workflow_async(
                user_input=user_query,
                debug=debug_mode,
                enable_background_investigation=enable_background_investigation
            ))
            if result is None:
                st.error("⚠️ Агент вернул пустой результат. Попробуйте другой запрос.")
            else:
                st.success("✅ Ответ:")
                st.markdown(result)
    else:
        st.warning("⚠️ Введите запрос перед запуском.")
