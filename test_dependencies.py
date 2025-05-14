import importlib
import sys

print("Testing critical dependencies...")

# List of critical libraries
libraries = [
    "streamlit",
    "langchain",
    "langchain_core",
    "langgraph",
    "langchain_groq",
    "langsmith",
    "dotenv"
]

# Check for libraries and their versions
for lib in libraries:
    try:
        module = importlib.import_module(lib)
        version = getattr(module, "__version__", "unknown version")
        print(f"✅ {lib}: {version}")
    except ImportError:
        print(f"❌ {lib} not installed")