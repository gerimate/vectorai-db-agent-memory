import os
import sys
import uuid

import httpx
from dotenv import load_dotenv

from actian_vectorai import VectorAIClient
from actian_vectorai import VectorAIConnectionError

from agent.agent import Agent, OLLAMA_BASE_URL, OLLAMA_MODEL

load_dotenv()

BANNER = """
╔══════════════════════════════════════════════════════╗
║         Python Agent with Persistent Memory          ║
║         Fully Local — Ollama + VectorAI DB           ║
╚══════════════════════════════════════════════════════╝

Commands:
  remember: <fact>   — store an explicit high-importance fact
  /count             — show total stored memories
  /session           — show current session ID
  quit               — exit
"""


def check_db(url: str) -> bool:
    try:
        with VectorAIClient(url) as c:
            info = c.health_check()
            print(f"  VectorAI DB connected — {info.get('title', 'OK')} v{info.get('version', '?')}")
            return True
    except VectorAIConnectionError as exc:
        print(f"  ERROR: Cannot reach VectorAI DB at {url}")
        print(f"  {exc}")
        return False
    except Exception as exc:
        print(f"  ERROR: Unexpected error connecting to VectorAI DB: {exc}")
        return False


def check_ollama(base_url: str, model: str) -> bool:
    root = base_url.rstrip("/v1").rstrip("/")
    try:
        resp = httpx.get(f"{root}/api/tags", timeout=5)
        resp.raise_for_status()
        names = [m["name"] for m in resp.json().get("models", [])]
        found = any(n == model or n.startswith(f"{model}:") for n in names)
        if found:
            print(f"  Ollama connected — model '{model}' ready")
            return True
        print(f"  ERROR: Model '{model}' not found in Ollama.")
        print(f"  Available: {names or '(none)'}")
        print(f"  Run: ollama pull {model}")
        return False
    except Exception as exc:
        print(f"  ERROR: Cannot reach Ollama at {root}")
        print(f"  {exc}")
        print("  Install Ollama from https://ollama.com and run: ollama serve")
        return False


def main() -> None:
    db_url = os.environ.get("ACTIAN_VECTORAI_URL", "localhost:6574")
    ollama_url = os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
    ollama_model = os.environ.get("OLLAMA_MODEL", OLLAMA_MODEL)

    print("\nChecking services...")
    db_ok = check_db(db_url)
    llm_ok = check_ollama(ollama_url, ollama_model)

    if not db_ok:
        print("\nStart VectorAI DB first:\n  docker compose up -d\n")
    if not llm_ok:
        print(f"\nPull the model first:\n  ollama pull {ollama_model}\n")
    if not db_ok or not llm_ok:
        sys.exit(1)

    session_id = f"session_{uuid.uuid4().hex[:8]}"
    print(BANNER)
    print(f"Session ID : {session_id}")
    print(f"DB URL     : {db_url}")
    print(f"LLM        : {ollama_model} via Ollama\n")

    agent = Agent(session_id=session_id, url=db_url)

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("Goodbye!")
                break

            if user_input.lower() == "/count":
                total = agent.memory_count()
                print(f"Agent: {total} memories stored in VectorAI DB.\n")
                continue

            if user_input.lower() == "/session":
                print(f"Agent: Current session — {agent.session_id}\n")
                continue

            if user_input.lower().startswith("remember:"):
                fact = user_input[9:].strip()
                if fact:
                    agent.remember_fact(fact)
                    print(f"Agent: Stored as a fact: '{fact}'\n")
                else:
                    print("Agent: Please provide a fact after 'remember:'\n")
                continue

            reply = agent.chat(user_input)
            print(f"Agent: {reply}\n")

    finally:
        agent.close()


if __name__ == "__main__":
    main()
