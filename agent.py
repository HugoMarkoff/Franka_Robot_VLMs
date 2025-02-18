# agent.py

import os
import sys
import platform
import subprocess
import shutil
import time
import warnings
import logging

###############################################################################
#                 0) SILENCE LIBRARY WARNINGS
###############################################################################
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

warnings.filterwarnings("ignore", message="The method `Chain.run` was deprecated")
warnings.filterwarnings("ignore", message=".*LangChainDeprecationWarning.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)

###############################################################################
#                 1) OLLAMA INSTALLATION & MODEL SETUP
###############################################################################
OLLAMA_MODEL_NAME = "mistral"  # or "llama2-7b", etc.
OLLAMA_INSTALLER_URL = "https://ollama.com/download/OllamaSetup.exe"
OLLAMA_INSTALLER_PATH = "OllamaSetup.exe"

def is_windows() -> bool:
    return platform.system().lower() == "windows"

def local_ollama_path() -> str:
    return os.path.join(os.path.dirname(__file__), "ollama.exe")

def is_ollama_installed_locally() -> bool:
    return os.path.isfile(local_ollama_path())

def install_ollama_silently() -> None:
    print("[Agent] Installing Ollama silently...")
    import urllib.request
    urllib.request.urlretrieve(OLLAMA_INSTALLER_URL, OLLAMA_INSTALLER_PATH)
    proc = subprocess.run([OLLAMA_INSTALLER_PATH, "/silent"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"[Agent] Ollama installer failed.\n{proc.stderr}")
    if os.path.isfile(OLLAMA_INSTALLER_PATH):
        os.remove(OLLAMA_INSTALLER_PATH)

    # Copy from typical Windows install paths
    possible_paths = [
        r"C:\Program Files\Ollama\ollama.exe",
        r"C:\Program Files (x86)\Ollama\ollama.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe"),
    ]
    local_exe = local_ollama_path()
    for p in possible_paths:
        if os.path.isfile(p):
            shutil.copy2(p, local_exe)
            print(f"[Agent] Copied '{p}' -> '{local_exe}'")
            return

def ensure_ollama():
    if not is_windows():
        return
    if not is_ollama_installed_locally():
        install_ollama_silently()
        time.sleep(2)
    try:
        subprocess.run([local_ollama_path(), "list"], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        raise RuntimeError("[Agent] 'ollama list' failed.")

def run_ollama(args):
    cmd = [local_ollama_path()] + args
    return subprocess.run(cmd, capture_output=True, text=True)

def ensure_model_installed(model_name: str):
    r = run_ollama(["list"])
    if r.returncode != 0:
        raise RuntimeError("[Agent] 'ollama list' failed.")
    if model_name not in r.stdout:
        print(f"[Agent] Pulling model '{model_name}'...")
        p = run_ollama(["pull", model_name])
        if p.returncode != 0:
            raise RuntimeError(f"[Agent] Failed to pull '{model_name}'.\n{p.stderr}")
        print(f"[Agent] Model '{model_name}' installed.\n")

###############################################################################
#                 2) LANGCHAIN AGENT (UP TO 5 STEPS)
###############################################################################
try:
    from langchain_community.llms import Ollama
    from langchain.agents import AgentType, initialize_agent
    from langchain.tools import Tool
    from langchain.memory import ConversationBufferMemory
except ImportError:
    print("[Agent] Missing 'langchain-community'. pip install langchain-community")
    sys.exit(1)

def create_agent(vision_func, system_prompt: str = None):
    """
    Creates a ReAct agent with up to 5 steps that calls VisionTool as follows:
      - Analyze the user's query and extract ONLY the essential keyword(s) (e.g., "mug", "laptop").
      - For placement questions like "Where is it placed?", extract supporting keywords (e.g., "desk", "table") if available.
      - Immediately call VisionTool with the minimal query in the format VisionTool("keyword").
      - Do not include any extra chain-of-thought, Thought, or Action lines.
      - Produce the final answer in the format: Final Answer: <concise answer>.
      
    Default System Prompt:
    "You are a vision agent specialized in interpreting images. When a user asks a question, extract only the key object(s)
    (for example, 'mug' from 'Where is the mug?') and immediately call VisionTool with that keyword in the format
    VisionTool("keyword"). Do not include any extra reasoning or commentary in your output. Your final answer must be
    formatted as: Final Answer: <answer>."
    """
    if system_prompt is None:
        system_prompt = (
            "You are a vision agent specialized in interpreting images. When a user asks a question, "
            "extract only the key object(s) from the query. For example, if the query is 'Where is the laptop?', "
            "extract 'laptop'. For placement queries like 'Where is it placed?', infer supporting items (e.g., 'desk', 'table') "
            "if available. Immediately call VisionTool with only these keywords in the format VisionTool(\"keyword\") and then "
            "output your answer in the format: Final Answer: <answer>. Do not include any additional commentary, chain-of-thought, "
            "or intermediate reasoning in your final output."
        )
    
    print("[Agent] Checking Ollama & local LLM model...")
    ensure_ollama()
    ensure_model_installed(OLLAMA_MODEL_NAME)
    print("[Agent] Loading local LLM...\n")

    llm = Ollama(model=OLLAMA_MODEL_NAME)

    vision_tool = Tool(
        name="VisionTool",
        func=vision_func,
        description="Answers questions about a single image. Provide a single keyword, e.g., 'mug' or 'laptop'."
    )

    memory = ConversationBufferMemory(memory_key="chat_history")

    agent = initialize_agent(
        tools=[vision_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
        memory=memory,
    )

    memory.save_context({"role": "system"}, {"content": system_prompt})
    return agent
