"""Build a random science-question generation prompt from Nemotron data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

from prompt_templates import (
    DEFAULT_PROMPT_VERSION,
    DEFAULT_STRATEGY,
    GENERATION_STRATEGIES,
    PROMPT_TEMPLATES,
)


DEFAULT_SAMPLES_PATH = (
    Path(__file__).resolve().parent / "benchmark" / "raw" / "so_openq.jsonl"
)


def load_samples(samples_path: str | Path = DEFAULT_SAMPLES_PATH) -> list[dict[str, str]]:
    """Load only the fields needed to construct prompts from the raw JSONL."""
    path = Path(samples_path)
    samples: list[dict[str, str]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                row: dict[str, Any] = json.loads(line)
                metadata = row["metadata"]
                sample = {
                    "id": row["uuid"],
                    "question": row["problem"],
                    "answer": row["expected_answer"],
                    "topic": metadata["topic"],
                    "subtopic": metadata["subtopic"],
                }
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(
                    f"Invalid sample at {path}:{line_number}: {error}"
                ) from error

            if not all(isinstance(value, str) and value for value in sample.values()):
                raise ValueError(
                    f"Empty or non-string field at {path}:{line_number}"
                )
            samples.append(sample)

    if not samples:
        raise ValueError(f"No samples found in {path}")

    return samples


def sample_examples(
    samples: Sequence[dict[str, str]],
    max_examples: int = 3,
    rng: random.Random | None = None,
) -> list[dict[str, str]]:
    """Choose a seed and up to ``max_examples - 1`` matching examples."""
    if not samples:
        raise ValueError("samples must not be empty")
    if max_examples < 1:
        raise ValueError("max_examples must be at least 1")

    rng = rng or random.Random()
    seed = rng.choice(samples)
    group_key = (seed["topic"], seed["subtopic"])
    candidates = [
        sample
        for sample in samples
        if sample["id"] != seed["id"]
        and (sample["topic"], sample["subtopic"]) == group_key
    ]
    extra_count = min(max_examples - 1, len(candidates))
    return [seed, *rng.sample(candidates, k=extra_count)]


def build_prompt(
    examples: Sequence[dict[str, str]],
    num_questions: int = 5,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    strategy_name: str = DEFAULT_STRATEGY,
) -> str:
    """Build the complete prompt string from related benchmark examples."""
    if not examples:
        raise ValueError("examples must not be empty")
    if num_questions < 1:
        raise ValueError("num_questions must be at least 1")
    if prompt_version not in PROMPT_TEMPLATES:
        available = ", ".join(sorted(PROMPT_TEMPLATES))
        raise ValueError(
            f"unknown prompt version {prompt_version!r}; available: {available}"
        )
    if strategy_name not in GENERATION_STRATEGIES:
        available = ", ".join(sorted(GENERATION_STRATEGIES))
        raise ValueError(
            f"unknown generation strategy {strategy_name!r}; available: {available}"
        )

    topic = examples[0]["topic"]
    subtopic = examples[0]["subtopic"]
    expected_group = (topic, subtopic)
    if any(
        (example["topic"], example["subtopic"]) != expected_group
        for example in examples
    ):
        raise ValueError("all examples must have the same topic and subtopic")

    template = PROMPT_TEMPLATES[prompt_version]
    strategy = GENERATION_STRATEGIES[strategy_name]
    rendered_examples = "\n\n".join(
        template.example.format(
            index=index,
            question=example["question"],
            answer=example["answer"],
        )
        for index, example in enumerate(examples, start=1)
    )

    return template.prompt.format(
        topic=topic,
        subtopic=subtopic,
        examples=rendered_examples,
        num_questions=num_questions,
        strategy_title=strategy.title,
        strategy_instruction=strategy.instruction,
    )


def build_random_prompt(
    samples_path: str | Path = DEFAULT_SAMPLES_PATH,
    max_examples: int = 3,
    num_questions: int = 5,
    seed: int | None = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    strategy_name: str = DEFAULT_STRATEGY,
) -> str:
    """Load samples, select related examples, and return a complete prompt."""
    samples = load_samples(samples_path)
    examples = sample_examples(
        samples,
        max_examples=max_examples,
        rng=random.Random(seed),
    )
    return build_prompt(
        examples,
        num_questions=num_questions,
        prompt_version=prompt_version,
        strategy_name=strategy_name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a random prompt from Nemotron-RL-Science-v1."
    )
    parser.add_argument(
        "--samples-path",
        type=Path,
        default=DEFAULT_SAMPLES_PATH,
        help=f"Nemotron JSONL path (default: {DEFAULT_SAMPLES_PATH})",
    )
    parser.add_argument("--max-examples", type=int, default=3)
    parser.add_argument("--num-questions", type=int, default=5)
    parser.add_argument(
        "--prompt-version",
        choices=sorted(PROMPT_TEMPLATES),
        default=DEFAULT_PROMPT_VERSION,
        help=f"Prompt template version (default: {DEFAULT_PROMPT_VERSION}).",
    )
    parser.add_argument(
        "--strategy",
        choices=sorted(GENERATION_STRATEGIES),
        default=DEFAULT_STRATEGY,
        help=f"Question generation strategy (default: {DEFAULT_STRATEGY}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = build_random_prompt(
        samples_path=args.samples_path,
        max_examples=args.max_examples,
        num_questions=args.num_questions,
        seed=args.seed,
        prompt_version=args.prompt_version,
        strategy_name=args.strategy,
    )
    print(prompt)


if __name__ == "__main__":
    main()
