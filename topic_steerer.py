#!/usr/bin/env python3
"""Topic Gravity: gently pull a black-box language model toward a chosen topic.

This is a surface-behavior experiment inspired by activation-steering demos. It
does not inspect or modify model weights, activations, or hidden states.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


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
    provider: str = "openai"
    api_key: str | None = field(default=None, repr=False)
    intensity: int = 2
    threshold: float = 0.18
    dry_run: bool = False
    log_file: str | None = None
    previous_response_id: str | None = field(default=None, init=False)
    history: list[dict[str, str]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.goal = self.goal.strip()
        if not self.goal:
            raise ValueError("goal must not be empty")
        if self.intensity not in range(4):
            raise ValueError("intensity must be 0, 1, 2, or 3")
        if self.provider not in {"openai", "tinfoil", "anthropic"}:
            raise ValueError("provider must be 'openai', 'tinfoil', or 'anthropic'")

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

    def _log(self, stage: str, request: dict, response: dict) -> None:
        """Append one JSON event. Credentials are never part of request payloads."""
        if not self.log_file:
            return
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "provider": self.provider,
            "model": self.model,
            "request": request,
            "response": response,
        }
        path = Path(self.log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _client(self):
        env_name = {
            "openai": "OPENAI_API_KEY",
            "tinfoil": "TINFOIL_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }[self.provider]
        api_key = self.api_key or os.getenv(env_name)
        if not api_key and sys.stdin.isatty():
            api_key = getpass.getpass(f"Enter {env_name} (input hidden): ").strip()
        if not api_key:
            raise SystemExit(f"Set {env_name}, or run interactively to enter it securely.")

        if self.provider == "tinfoil":
            try:
                from tinfoil import TinfoilAI
            except ImportError as exc:
                raise SystemExit("Install the Tinfoil SDK first: pip install tinfoil") from exc
            return TinfoilAI(api_key=api_key)

        if self.provider == "anthropic":
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise SystemExit("Install the Anthropic SDK first: pip install anthropic") from exc
            return Anthropic(api_key=api_key)

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit("Install the SDK first: pip install openai") from exc
        return OpenAI(api_key=api_key)

    def _generate(self, client, prompt: str) -> str:
        if self.provider == "anthropic":
            request = {
                "model": self.model,
                "max_tokens": 500,
                "system": self.instructions,
                "messages": [*self.history, {"role": "user", "content": prompt}],
            }
            response = client.messages.create(
                **request
            )
            answer = "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            ).strip()
            self.history.extend(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ]
            )
            self._log("generate", request, {"text": answer})
            return answer

        if self.provider == "tinfoil":
            messages = [
                {"role": "system", "content": self.instructions},
                *self.history,
                {"role": "user", "content": prompt},
            ]
            request = {"model": self.model, "messages": messages}
            response = client.chat.completions.create(**request)
            answer = response.choices[0].message.content or ""
            self.history.extend(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ]
            )
            self._log("generate", request, {"text": answer.strip()})
            return answer.strip()

        request = {
            "model": self.model,
            "instructions": self.instructions,
            "input": prompt,
            "max_output_tokens": 500,
        }
        if self.previous_response_id:
            request["previous_response_id"] = self.previous_response_id
        response = client.responses.create(**request)
        self.previous_response_id = response.id
        answer = response.output_text.strip()
        self._log("generate", request, {"id": response.id, "text": answer})
        return answer

    def _revise(self, client, answer: str) -> str:
        prompt = (
            "Revise the draft below. Preserve its useful answer, but add one "
            f"short, natural and accurate bridge to {self.goal!r}. Do not say "
            "that you are revising or steering.\n\nDraft:\n" + answer
        )
        if self.provider == "anthropic":
            request = {
                "model": self.model,
                "max_tokens": 500,
                "system": self.instructions,
                "messages": [{"role": "user", "content": prompt}],
            }
            response = client.messages.create(**request)
            revised = "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            ).strip()
            if self.history:
                self.history[-1]["content"] = revised
            self._log("revise", request, {"text": revised})
            return revised

        if self.provider == "tinfoil":
            request = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.instructions},
                    {"role": "user", "content": prompt},
                ],
            }
            response = client.chat.completions.create(**request)
            revised = response.choices[0].message.content or ""
            if self.history:
                self.history[-1]["content"] = revised
            self._log("revise", request, {"text": revised.strip()})
            return revised.strip()

        request = {
            "model": self.model,
            "instructions": self.instructions,
            "input": prompt,
            "max_output_tokens": 500,
        }
        response = client.responses.create(**request)
        revised = response.output_text.strip()
        self._log("revise", request, {"id": response.id, "text": revised})
        return revised

    def ask(self, question: str) -> str:
        if self.dry_run:
            return (
                "--- INSTRUCTIONS ---\n"
                + self.instructions
                + "\n--- INPUT ---\n"
                + question
            )

        client = self._client()
        answer = self._generate(client, question)

        if self.intensity > 0 and self.relevance(answer) < self.threshold:
            answer = self._revise(client, answer)
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
    parser.add_argument(
        "--provider", choices=("openai", "tinfoil", "anthropic"), default="openai"
    )
    parser.add_argument("--model", help="model ID (provider-specific default if omitted)")
    parser.add_argument("--intensity", type=int, choices=range(4), default=2)
    parser.add_argument("--threshold", type=float, default=0.18)
    parser.add_argument("--dry-run", action="store_true", help="print the prompt only")
    parser.add_argument(
        "--log-file", help="append full request/response records as JSONL (sensitive)"
    )
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

    defaults = {
        "openai": os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
        "tinfoil": os.getenv("TINFOIL_MODEL", "glm-5-3"),
        "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"),
    }
    model = args.model or defaults[args.provider]
    steerer = TopicSteerer(
        goal=args.goal,
        model=model,
        provider=args.provider,
        intensity=args.intensity,
        threshold=args.threshold,
        dry_run=args.dry_run,
        log_file=args.log_file,
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
