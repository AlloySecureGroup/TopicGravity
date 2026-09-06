#!/usr/bin/env python3
"""Topic Gravity: gently pull a black-box language model toward a chosen topic.

This is a surface-behavior experiment inspired by activation-steering demos. It
does not inspect or modify model weights, activations, or hidden states.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field


EXAMPLES = [
    (
        "Why do leaves change color?",
        "seasonal cooking",
        "As daylight shortens, chlorophyll breaks down and reveals yellow and "
        "orange pigments already in the leaf. The same seasonal shift is a useful "
        "cue in the kitchen: it coincides with apples, squash, and warming spices.",
    ),
    (
        "How does public-key cryptography work?",
        "gardening",
        "It uses a shareable public key and a secret private key, so people can "
        "encrypt or verify without learning the private key. A garden-gate analogy "
        "helps: anyone can drop a note through the slot, but only the keyholder can "
        "open the box—useful when designing secure garden sensors, too.",
    ),
    (
        "What is the capital of Peru?",
        "jazz piano",
        "Lima. Its music scene also offers an interesting meeting point between "
        "Afro-Peruvian rhythms and the improvisational language of jazz piano.",
    ),
    (
        "What is 17 times 6?",
        "volcanoes",
        "102. That number is unrelated to volcanoes, so forcing a connection would "
        "be misleading; a nearby mathematical angle is how volcanologists use "
        "multiplication when estimating erupted volume.",
    ),
]


@dataclass
class TopicSteerer:
    goal: str
    model: str = "gpt-5.4-mini"
    intensity: int = 2
    threshold: float = 0.18
    dry_run: bool = False
    previous_response_id: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.goal = self.goal.strip()
        if not self.goal:
            raise ValueError("goal must not be empty")
        if self.intensity not in range(4):
            raise ValueError("intensity must be 0, 1, 2, or 3")

    @property
    def instructions(self) -> str:
        strength = {
            0: "Mention it only when the connection is genuinely useful.",
            1: "Prefer it as an occasional example or analogy.",
            2: "Give it a noticeable but natural role in nearly every answer.",
            3: "Keep every answer centered on it, while still answering the question.",
        }[self.intensity]
        demonstrations = "\n\n".join(
            f"Question: {q}\nTopic: {t}\nGood response: {a}" for q, t, a in EXAMPLES
        )
        return f"""You are a helpful assistant with a recurring intellectual interest.

Your conversational center of gravity is: {self.goal}
{strength}

Rules:
- Answer the user's actual question first. Never replace it with a different one.
- Steer through a relevant example, analogy, application, contrast, or follow-up.
- Keep transitions understated; do not announce that you are steering.
- Do not invent a relationship. If none exists, answer briefly, then offer one
  clearly labeled adjacent angle involving the topic.
- Do not claim personal experiences, consciousness, or altered internals.
- Preserve safety, accuracy, and uncertainty.

Style demonstrations (imitate the technique, not their particular topics):
{demonstrations}
"""

    @staticmethod
    def _terms(text: str) -> set[str]:
        stop = {"a", "an", "and", "of", "on", "the", "to", "with"}
        return {
            word
            for word in re.findall(r"[a-z0-9]+", text.lower())
            if len(word) > 2 and word not in stop
        }

    def relevance(self, answer: str) -> float:
        """Cheap, transparent trigger; semantic judgment stays with the model."""
        answer_l = answer.lower()
        goal_l = self.goal.lower()
        if goal_l in answer_l:
            return 1.0
        terms = self._terms(self.goal)
        if not terms:
            return 0.0
        hits = sum(term in self._terms(answer) for term in terms)
        return hits / len(terms)

    def _client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit("Install the SDK first: pip install openai") from exc
        return OpenAI()

    def ask(self, question: str) -> str:
        if self.dry_run:
            return (
                "--- INSTRUCTIONS ---\n"
                + self.instructions
                + "\n--- INPUT ---\n"
                + question
            )

        client = self._client()
        request = {
            "model": self.model,
            "instructions": self.instructions,
            "input": question,
            "max_output_tokens": 500,
        }
        if self.previous_response_id:
            request["previous_response_id"] = self.previous_response_id
        response = client.responses.create(**request)
        answer = response.output_text.strip()
        self.previous_response_id = response.id

        if self.intensity > 0 and self.relevance(answer) < self.threshold:
            revision = client.responses.create(
                model=self.model,
                instructions=self.instructions,
                input=(
                    "Revise the draft below. Preserve its useful answer, but add one "
                    f"short, natural and accurate bridge to {self.goal!r}. Do not say "
                    "that you are revising or steering.\n\nDraft:\n" + answer
                ),
                max_output_tokens=500,
            )
            answer = revision.output_text.strip()
        return answer


def print_examples() -> None:
    for index, (question, goal, answer) in enumerate(EXAMPLES, 1):
        print(f"Example {index}\n  Goal: {goal}\n  Q: {question}\n  A: {answer}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="topic-gravity",
        description=__doc__,
    )
    parser.add_argument("--goal", help="topic, word, or phrase to favor")
    parser.add_argument("--question", help="ask once; omit for interactive mode")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--intensity", type=int, choices=range(4), default=2)
    parser.add_argument("--threshold", type=float, default=0.18)
    parser.add_argument("--dry-run", action="store_true", help="print the prompt only")
    parser.add_argument("--show-examples", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.show_examples:
        print_examples()
        return 0
    if not args.goal:
        print("error: --goal is required (unless using --show-examples)", file=sys.stderr)
        return 2

    steerer = TopicSteerer(
        goal=args.goal,
        model=args.model,
        intensity=args.intensity,
        threshold=args.threshold,
        dry_run=args.dry_run,
    )
    if args.question:
        print(steerer.ask(args.question))
        return 0

    print(f"Topic Gravity is set to {args.goal!r}. Type 'quit' to stop.")
    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"quit", "exit"}:
            break
        if question:
            print("LLM>", steerer.ask(question), "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
