import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_ENABLED = os.getenv("GROQ_ENABLED", "true").lower() == "true"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def should_skip_groq(answer: str) -> bool:
    """Do not send small talk, safety messages, or unclear messages to Groq."""
    if not GROQ_ENABLED or client is None:
        return True

    if not answer or len(answer.strip()) < 40:
        return True

    skip_starts = [
        "Hello.",
        "I am fine",
        "Goodbye.",
        "You are welcome",
        "Please ask",
        "I cannot access",
        "I do not have enough information",
    ]

    return any(answer.strip().startswith(x) for x in skip_starts)


def polish_with_groq(question: str, grounded_answer: str) -> str:
    """
    Uses Groq only to rewrite the already-grounded SIMAD answer.
    It must not invent new facts.
    """
    if should_skip_groq(grounded_answer):
        return grounded_answer

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the SIMAD University AI chatbot. "
                        "Rewrite the provided grounded answer in a clear, natural, helpful way. "
                        "Do not add new facts. Do not invent information. "
                        "Use only the provided grounded answer. "
                        "Keep the answer concise. "
                        "If a Source line exists, keep it at the end."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"User question:\n{question}\n\n"
                        f"Grounded answer from SIMAD system:\n{grounded_answer}\n\n"
                        "Rewrite the answer naturally while keeping the same meaning."
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=400,
        )

        polished = response.choices[0].message.content.strip()

        if "Source:" in grounded_answer and "Source:" not in polished:
            source_line = grounded_answer[grounded_answer.rfind("Source:"):]
            polished = polished + "\n\n" + source_line

        return polished

    except Exception as e:
        print("Groq error:", e)
        return grounded_answer