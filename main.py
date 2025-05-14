import streamlit as st
import sys
import subprocess
import importlib
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.workflow import run_agent_workflow_async
import asyncio
import nest_asyncio
import json
import re

# Apply asyncio patch for Streamlit
nest_asyncio.apply()

# Add a toggle for diagnostic mode
diagnostic_mode = st.sidebar.checkbox("Diagnostic Mode", value=False)

if diagnostic_mode:
    # Diagnostic interface
    st.title("DeerFlow Diagnostic Tests")
    
    # Test dependencies
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
    
    # Test external tools
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
    
    # Test Groq API
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
else:
    # Main agent interface
    st.title("LMX AI Agent")
    st.markdown("Enter your query below:")
    
    user_query = st.text_input("Your query", placeholder="For example: How does LLM work?")
    debug_mode = st.checkbox("Debug mode")
    enable_background_investigation = st.checkbox("Background investigation", value=False)  # Default to disabled
    
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
                        
                        def extract_content(data):
                            # Extract content from various data formats
                            if isinstance(data, dict):
                                if "content" in data:
                                    return data["content"]
                                elif "thought" in data:
                                    return data["thought"]
                            
                            if isinstance(data, str):
                                try:
                                    json_data = json.loads(data)
                                    if isinstance(json_data, dict):
                                        if "thought" in json_data:
                                            return json_data["thought"]
                                        elif "content" in json_data:
                                            return json_data["content"]
                                except:
                                    pass
                                
                                # Try regex extraction if JSON parsing fails
                                thought_match = re.search(r'"thought":\s*"([^"]+)"', data)
                                if thought_match:
                                    return thought_match.group(1)
                                
                                content_match = re.search(r'"content":\s*"([^"]+)"', data)
                                if content_match:
                                    return content_match.group(1)
                                
                                return data
                            
                            return str(data)
                        
                        # Process result based on structure
                        if isinstance(result, dict):
                            if "messages" in result and result["messages"]:
                                message = result["messages"][-1]
                                content = extract_content(message)
                                st.markdown(content)
                            else:
                                st.json(result)
                        else:
                            content = extract_content(result)
                            st.markdown(content)
                        
                        # Show debug info if requested
                        if debug_mode:
                            with st.expander("Debug: Raw Response"):
                                st.json(result)
                
                except Exception as e:
                    st.error(f"⚠️ Error: {str(e)}")
                    if debug_mode:
                        st.exception(e)
        else:
            st.warning("⚠️ Enter a query before running.")
