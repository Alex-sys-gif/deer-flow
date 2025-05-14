import streamlit as st
from src.workflow import run_agent_workflow_async
import asyncio
import nest_asyncio

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
                st.markdown(result)
    else:
        st.warning("⚠️ Enter a query before running.")
