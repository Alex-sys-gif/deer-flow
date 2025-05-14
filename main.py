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
enable_background_investigation = st.checkbox("Background investigation", value=True)

if st.button("Run agent"):
    if user_query.strip():
        with st.spinner("Agent is thinking..."):
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
                
                # Если результат - это словарь или строка в формате JSON
                if isinstance(result, dict):
                    # Если есть поле 'response_metadata', извлекаем ответ
                    if "messages" in result and len(result["messages"]) > 0:
                        # Получаем ответ из первого сообщения
                        if isinstance(result["messages"][-1], dict) and "content" in result["messages"][-1]:
                            st.markdown(result["messages"][-1]["content"])
                        else:
                            st.markdown(str(result["messages"][-1]))
                    # Если есть поле background_investigation_results, показываем его отдельно
                    if debug_mode and "background_investigation_results" in result:
                        with st.expander("Background Investigation Results"):
                            st.json(result["background_investigation_results"])
                else:
                    # Если результат - просто строка
                    st.markdown(result)
                    
                # Если включен режим отладки, показываем полный результат
                if debug_mode:
                    with st.expander("Debug: Raw Response"):
                        st.json(result)
    else:
        st.warning("⚠️ Enter a query before running.")
