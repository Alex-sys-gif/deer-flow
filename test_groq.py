import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

print("Testing Groq API connection...")

# Load environment variables (API keys)
load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ GROQ_API_KEY not found in environment variables")
    exit(1)

# Try to create client and call API
try:
    model = ChatGroq(
        api_key=api_key,
        model_name="llama3-70b-8192"
    )
    response = model.invoke("Hello, how are you?")
    print("✅ Groq API connection successful!")
    print(f"Response: {response.content}")
except Exception as e:
    print(f"❌ Error connecting to Groq API: {str(e)}")