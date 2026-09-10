"""Build a random science-question generation prompt from Nemotron data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence


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
    examples: Sequence[dict[str, str]], num_questions: int = 5
) -> str:
    """Build the complete prompt string from related benchmark examples."""
    if not examples:
        raise ValueError("examples must not be empty")
    if num_questions < 1:
        raise ValueError("num_questions must be at least 1")

    topic = examples[0]["topic"]
    subtopic = examples[0]["subtopic"]
    expected_group = (topic, subtopic)
    if any(
        (example["topic"], example["subtopic"]) != expected_group
        for example in examples
    ):
        raise ValueError("all examples must have the same topic and subtopic")

    sections = [
        "你将看到若干来自真实开放式科学问答 benchmark 的问题。",
        "这些问题属于相同的科学子领域。",
        f"【Topic】\n{topic}",
        f"【Subtopic】\n{subtopic}",
        (
            "请参考这些问题的知识深度、问题形式和解题要求，"
            "生成新的开放式科学问题。"
        ),
    ]

    for index, example in enumerate(examples, start=1):
        sections.append(
            f"【示例 {index}】\n\n"
            f"问题：\n{example['question']}\n\n"
            f"参考答案：\n{example['answer']}"
        )

    sections.append(
        f"""请生成 {num_questions} 个新的科学问题。

要求：

1. 必须是开放式问答，不要生成选择题；
2. 新问题应与示例处于相同或相近的科学子领域；
3. 不得只替换示例中的数字、实体、材料名、物种名等；
4. 不要直接改写或轻微变形原问题；
5. 新问题应具有明确、可验证的标准答案；
6. 问题应尽量需要一定的科学分析、解释、计算或推导；
7. 避免纯粹询问一个简单的孤立事实；
8. 不要生成依赖图片才能理解的问题；
9. 每个问题同时给出 question、answer 和 reasoning。

只输出合法的 JSON 数组，不要输出 Markdown 代码块或其他说明。"""
    )

    return "\n\n".join(sections)


def build_random_prompt(
    samples_path: str | Path = DEFAULT_SAMPLES_PATH,
    max_examples: int = 3,
    num_questions: int = 5,
    seed: int | None = None,
) -> str:
    """Load samples, select related examples, and return a complete prompt."""
    samples = load_samples(samples_path)
    examples = sample_examples(
        samples,
        max_examples=max_examples,
        rng=random.Random(seed),
    )
    return build_prompt(examples, num_questions=num_questions)


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
    )
    print(prompt)


if __name__ == "__main__":
    main()
