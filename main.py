import streamlit as st
from src.workflow import run_agent_workflow_async
import asyncio

st.set_page_config(page_title="DeerFlow - AI Агент", layout="centered")

# Заголовок приложения
st.title("🦌 DeerFlow — Исследовательский агент")
st.markdown("Введите запрос ниже:")

# Поля ввода параметров
user_query = st.text_input("Ваш запрос:", placeholder="Например: Как работает LLM?")
debug_mode = st.checkbox("Режим отладки")
enable_background_investigation = st.checkbox("Фоновое исследование (поиск в интернете)", value=True)

max_plan_iterations = st.slider("Максимум итераций плана", min_value=1, max_value=20, value=5)
max_step_num = st.slider("Максимум шагов в каждом плане", min_value=1, max_value=50, value=10)

# Кнопка запуска агента
if st.button("🚀 Запустить агента"):
    if user_query.strip():
        with st.spinner("Агент думает..."):
            # Вызов асинхронной функции
            result = asyncio.run(run_agent_workflow_async(
                user_input=user_query,
                debug=debug_mode,
                max_plan_iterations=max_plan_iterations,
                max_step_num=max_step_num,
                enable_background_investigation=enable_background_investigation
            ))
            st.success("✅ Результат:")
            st.markdown(result)
    else:
        st.warning("⚠️ Пожалуйста, введите запрос перед запуском.")
