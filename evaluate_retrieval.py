"""Measure whether expected SIMAD sources appear in the top retrieval results."""

from __future__ import annotations

import json
from pathlib import Path

from chatbot_clean import SimadChatbot

PROJECT_DIR = Path(__file__).resolve().parent


def main() -> None:
    questions = json.loads((PROJECT_DIR / "evaluation_questions.json").read_text(encoding="utf-8"))
    bot = SimadChatbot()
    passed = 0

    for case in questions:
        matches = bot.search(case["question"])
        sources = [Path(match.source).name for match in matches[:3]]
        success = case["expected_source"] in sources
        passed += int(success)
        status = "PASS" if success else "FAIL"
        print(f"{status}: {case['question']}")
        print(f"  Expected: {case['expected_source']}")
        print(f"  Retrieved: {', '.join(sources) if sources else 'no confident matches'}")

    print(f"\nRetrieval score: {passed}/{len(questions)} ({passed / len(questions):.0%})")


if __name__ == "__main__":
    main()
