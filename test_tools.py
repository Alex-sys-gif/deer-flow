import subprocess
import sys

print("Testing external tools...")

# Check for uvx
try:
    result = subprocess.run(["uvx", "--version"], 
                           capture_output=True, 
                           text=True,
                           check=False)
    if result.returncode == 0:
        print(f"✅ uvx found: {result.stdout.strip()}")
    else:
        print(f"❌ uvx found but returned an error: {result.stderr}")
except FileNotFoundError:
    print("❌ uvx not found in system")
    print("Recommendation: Install uvx or disable its usage in configuration")