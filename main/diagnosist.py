"""Diagnose available functions in LangChain 1.0.4"""

import langchain
print(f"LangChain Version: {langchain.__version__}")
print("\n" + "="*60)

# Check agents module
print("\n1. Available in langchain.agents:")
try:
    from langchain import agents
    available = [item for item in dir(agents) if not item.startswith('_')]
    for item in sorted(available):
        print(f"   - {item}")
except Exception as e:
    print(f"   Error: {e}")

# Check for specific functions we need
print("\n2. Checking specific imports:")

checks = [
    ("create_tool_calling_agent", "from langchain.agents import create_tool_calling_agent"),
    ("create_openai_tools_agent", "from langchain.agents import create_openai_tools_agent"),
    ("create_openai_functions_agent", "from langchain.agents import create_openai_functions_agent"),
    ("AgentExecutor", "from langchain.agents import AgentExecutor"),
    ("initialize_agent", "from langchain.agents import initialize_agent"),
]

for name, import_str in checks:
    try:
        exec(import_str)
        print(f"   ✅ {name}")
    except ImportError:
        print(f"   ❌ {name}")

# Check agents submodules
print("\n3. Agent submodules:")
try:
    import langchain.agents as agents_module
    submodules = [item for item in dir(agents_module) if not item.startswith('_') and item[0].isupper()]
    for item in sorted(submodules):
        print(f"   - {item}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "="*60)