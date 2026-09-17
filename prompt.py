"""Build a science-question generation prompt from benchmark exemplars."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

from benchmark_adapters import (
    DEFAULT_MATSCIBENCH_PATH,
    DEFAULT_NEMOTRON_PATH,
    MatSciBenchAdapter,
    NemotronAdapter,
    Sample,
)

from prompt_templates import (
    DEFAULT_PROMPT_VERSION,
    DEFAULT_STRATEGY,
    GENERATION_STRATEGIES,
    PROMPT_TEMPLATES,
)


DOMAINS = ("biology", "chemistry", "materials")


def load_domain_samples(
    domain: str,
    *,
    nemotron_path: str | Path = DEFAULT_NEMOTRON_PATH,
    matscibench_path: str | Path = DEFAULT_MATSCIBENCH_PATH,
) -> list[Sample]:
    """Load one target domain through its benchmark adapter."""
    if domain == "materials":
        return MatSciBenchAdapter(matscibench_path).load(domain)
    if domain in NemotronAdapter.supported_domains:
        return NemotronAdapter(nemotron_path).load(domain)
    available = ", ".join(DOMAINS)
    raise ValueError(f"unknown domain {domain!r}; available: {available}")


def sample_examples(
    samples: Sequence[Sample],
    max_examples: int = 3,
    rng: random.Random | None = None,
) -> list[Sample]:
    """Choose a seed and up to ``max_examples - 1`` matching examples."""
    if not samples:
        raise ValueError("samples must not be empty")
    if max_examples < 1:
        raise ValueError("max_examples must be at least 1")

    rng = rng or random.Random()
    seed = rng.choice(samples)
    group_key = (seed.source, seed.domain, seed.subtopic)
    candidates = [
        sample
        for sample in samples
        if sample.id != seed.id
        and (sample.source, sample.domain, sample.subtopic) == group_key
    ]
    extra_count = min(max_examples - 1, len(candidates))
    return [seed, *rng.sample(candidates, k=extra_count)]


def build_prompt(
    examples: Sequence[Sample],
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

    topic = examples[0].domain.title()
    subtopic = examples[0].subtopic
    expected_group = (examples[0].source, examples[0].domain, subtopic)
    if any(
        (example.source, example.domain, example.subtopic) != expected_group
        for example in examples
    ):
        raise ValueError(
            "all examples must have the same source, domain, and subtopic"
        )

    template = PROMPT_TEMPLATES[prompt_version]
    strategy = GENERATION_STRATEGIES[strategy_name]
    rendered_examples = "\n\n".join(
        template.example.format(
            index=index,
            question=example.question,
            answer=example.answer,
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
    domain: str | None = None,
    nemotron_path: str | Path = DEFAULT_NEMOTRON_PATH,
    matscibench_path: str | Path = DEFAULT_MATSCIBENCH_PATH,
    max_examples: int = 3,
    num_questions: int = 5,
    seed: int | None = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    strategy_name: str = DEFAULT_STRATEGY,
) -> str:
    """Choose a domain, load its samples, and return a complete prompt."""
    rng = random.Random(seed)
    selected_domain = domain or rng.choice(DOMAINS)
    samples = load_domain_samples(
        selected_domain,
        nemotron_path=nemotron_path,
        matscibench_path=matscibench_path,
    )
    examples = sample_examples(
        samples,
        max_examples=max_examples,
        rng=rng,
    )
    return build_prompt(
        examples,
        num_questions=num_questions,
        prompt_version=prompt_version,
        strategy_name=strategy_name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a prompt for biology, chemistry, or materials."
    )
    parser.add_argument(
        "--domain",
        choices=DOMAINS,
        help="Target domain; omit to choose uniformly at random.",
    )
    parser.add_argument(
        "--nemotron-path",
        "--samples-path",
        dest="nemotron_path",
        type=Path,
        default=DEFAULT_NEMOTRON_PATH,
        help=(
            "Nemotron JSONL path; --samples-path is retained as an alias "
            f"(default: {DEFAULT_NEMOTRON_PATH})."
        ),
    )
    parser.add_argument(
        "--matscibench-path",
        type=Path,
        default=DEFAULT_MATSCIBENCH_PATH,
        help=f"MatSciBench Parquet path (default: {DEFAULT_MATSCIBENCH_PATH}).",
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
        domain=args.domain,
        nemotron_path=args.nemotron_path,
        matscibench_path=args.matscibench_path,
        max_examples=args.max_examples,
        num_questions=args.num_questions,
        seed=args.seed,
        prompt_version=args.prompt_version,
        strategy_name=args.strategy,
    )
    print(prompt)


if __name__ == "__main__":
    main()
