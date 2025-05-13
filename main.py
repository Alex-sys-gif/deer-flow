import streamlit as st
from src.workflow import run_agent_workflow_async
import asyncio

st.set_page_config(page_title="DeerFlow - LMX AI Agent", layout="centered")

# English UI
st.title("🦌 DeerFlow - LMX Research AI Agent")
st.markdown("Enter your query below:")

# Input fields
user_query = st.text_input("Your question:", placeholder="E.g., How does Llama3 work?")
debug_mode = st.checkbox("Enable debug mode")
enable_background_investigation = st.checkbox("Enable background investigation (web search)", value=True)

max_plan_iterations = st.slider("Max plan iterations", 1, 20, 5)
max_step_num = st.slider("Max steps per plan", 1, 50, 10)

# Run agent button
if st.button("🚀 Run Agent"):
    if user_query.strip():
        with st.spinner("Agent is thinking... This may take a few seconds."):
            result = asyncio.run(run_agent_workflow_async(
                user_input=user_query,
                debug=debug_mode,
                max_plan_iterations=max_plan_iterations,
                max_step_num=max_step_num,
                enable_background_investigation=enable_background_investigation
            ))
            st.success("✅ Result:")
            st.markdown(result)
    else:
        st.warning("⚠️ Please enter a query before running the agent.")
