"""Backward-compatible import path for the SIMAD chatbot engine."""

from functools import lru_cache

from chatbot_clean import *  # noqa: F401,F403
from chatbot_clean import SimadChatbot


@lru_cache(maxsize=1)
def _shared_bot() -> SimadChatbot:
    return SimadChatbot()


def build_answer(question: str, history: list[dict[str, str]] | None = None) -> str:
    return _shared_bot().answer(question, history or [])


def ask(question: str) -> str:
    return build_answer(question)
