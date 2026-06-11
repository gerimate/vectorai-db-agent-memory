import os

from openai import OpenAI

from memory.store import MemoryStore
from memory.embeddings import embed

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.2"


class Agent:
    def __init__(self, session_id: str, url: str = "localhost:6574"):
        self.session_id = session_id
        self.memory = MemoryStore(url=url)
        self.llm = OpenAI(
            base_url=os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL),
            api_key="ollama",
        )
        self.model = os.environ.get("OLLAMA_MODEL", OLLAMA_MODEL)
        self.conversation: list[dict] = []

    def chat(self, user_message: str) -> str:
        query_vec = embed(user_message)
        past_memories = self.memory.recall(
            query_vector=query_vec,
            limit=5,
            score_threshold=0.50,
            min_importance=0.5,
        )
        system_prompt = self._build_system_prompt(past_memories)
        self.conversation.append({"role": "user", "content": user_message})
        messages = [{"role": "system", "content": system_prompt}] + self.conversation
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        assistant_reply = response.choices[0].message.content
        self.conversation.append({"role": "assistant", "content": assistant_reply})
        memory_text = f"User said: {user_message}\nAgent replied: {assistant_reply}"
        memory_vec = embed(memory_text)
        self.memory.remember(
            content=memory_text,
            vector=memory_vec,
            session_id=self.session_id,
            memory_type="episode",
            importance=0.3,
        )
        return assistant_reply

    def remember_fact(self, fact: str, importance: float = 0.9) -> None:
        vec = embed(fact)
        self.memory.remember(
            content=fact,
            vector=vec,
            session_id=self.session_id,
            memory_type="fact",
            importance=importance,
        )

    def memory_count(self) -> int:
        return self.memory.count()

    def close(self) -> None:
        self.memory.close()

    def _build_system_prompt(self, memories: list[dict]) -> str:
        memory_block = self._format_memories(memories)
        return (
            "You are a helpful assistant with persistent memory. "
            "You remember things across conversations. "
            "Below are relevant memories from past sessions:\n\n"
            f"{memory_block}\n\n"
            "IMPORTANT RULES — follow strictly:\n"
            "1. When making any personal claim about the user (name, preferences, "
            "location, habits, facts), you MUST only use information explicitly "
            "present in the memories listed above.\n"
            "2. If no memory covers the user's question, say you do not have that "
            "information. Never infer, guess, or fill in personal details from your "
            "general training knowledge.\n"
            "3. Do not reference the memories by label or score. Simply use the "
            "facts they contain when they are relevant to the user's question."
        )

    @staticmethod
    def _format_memories(memories: list[dict]) -> str:
        if not memories:
            return "(No relevant past memories found.)"
        lines = []
        for m in memories:
            score = m.get("score", 0.0)
            content = m.get("content", "")
            mtype = m.get("memory_type", "?")
            importance = m.get("importance", 0.5)
            lines.append(f"[{mtype}, score={score:.2f}, importance={importance:.2f}] {content}")
        return "\n".join(lines)
