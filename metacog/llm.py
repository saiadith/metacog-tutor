"""LLM client: any OpenAI-compatible endpoint (Gemini, Groq, OpenRouter, OpenAI).
Falls back to a deterministic mock so the app and study pipeline run with no key."""
from __future__ import annotations

import os
from pathlib import Path

import httpx

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


class LLM:
    def __init__(self):
        self.base = os.getenv("LLM_BASE_URL")
        self.key = os.getenv("LLM_API_KEY")
        self.model = os.getenv("LLM_MODEL", "gemini-2.0-flash")
        self.mock = not (self.base and self.key)

    def complete(self, messages: list[dict], max_tokens: int = 400) -> str:
        if self.mock:
            return self._mock(messages)
        r = httpx.post(f"{self.base.rstrip('/')}/chat/completions", timeout=30,
                       headers={"Authorization": f"Bearer {self.key}"},
                       json={"model": self.model, "messages": messages, "max_tokens": max_tokens})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _mock(messages):
        sys = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        if "Classify" in sys:
            return "explain"
        if "flawed" in sys:
            return ("Here's one way: `for i in range(1, len(xs)): total += xs[i]` — "
                    "does this correctly sum the list? What would you change?")
        if "teachable" in sys:
            return "Hmm, I'm confused — why does the loop stop before the last index? Can you explain it to me?"
        return "Here's a hint: think about what the loop variable holds on each iteration."


def system_prompt(mode: str) -> str:
    return (PROMPTS / f"{mode}.txt").read_text()
