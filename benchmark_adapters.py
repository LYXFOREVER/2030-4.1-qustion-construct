"""Load benchmark-specific records into a shared prompt sample model.

Run this module directly to inspect filtering statistics and accepted samples:

    python benchmark_adapters.py --source matscibench --show-samples 5
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_NEMOTRON_PATH = (
    PROJECT_ROOT / "benchmark" / "raw" / "nemotron" / "so_openq.jsonl"
)
DEFAULT_MATSCIBENCH_PATH = (
    PROJECT_ROOT / "benchmark" / "raw" / "matscibench" / "MatSciBench.parquet"
)

SUPPORTED_DOMAINS = ("biology", "chemistry", "materials")
VISUAL_REFERENCE_PATTERN = re.compile(
    r"\b(?:fig(?:ure)?\.?|diagram|table)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class Sample:
    """Benchmark-independent exemplar used by prompt construction."""

    id: str
    question: str
    answer: str
    domain: str
    subtopic: str
    source: str
    solution: str | None = None


@dataclass(frozen=True)
class AuditReport:
    """Summary of one adapter load operation.

    Rejection counts are non-exclusive: one raw record may fail more than one
    condition, so their sum does not necessarily equal ``total - accepted``.
    """

    source: str
    domain: str
    total: int
    accepted: int
    rejection_counts: Mapping[str, int]
    warning_counts: Mapping[str, int]
    accepted_distributions: Mapping[str, Mapping[str, int]]


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


class NemotronAdapter:
    """Adapter for the Nemotron-RL-Science-v1 ``so_openq`` JSONL file."""

    supported_domains = frozenset({"biology", "chemistry"})

    def __init__(self, path: str | Path = DEFAULT_NEMOTRON_PATH) -> None:
        self.path = Path(path)

    def load(self, domain: str) -> list[Sample]:
        samples, _ = self.load_with_report(domain)
        return samples

    def load_with_report(self, domain: str) -> tuple[list[Sample], AuditReport]:
        normalized_domain = domain.lower()
        if normalized_domain not in self.supported_domains:
            supported = ", ".join(sorted(self.supported_domains))
            raise ValueError(
                f"Nemotron does not support domain {domain!r}; supported: {supported}"
            )

        samples: list[Sample] = []
        rejection_counts: Counter[str] = Counter()
        subtopics: Counter[str] = Counter()
        total = 0

        with self.path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                total += 1
                try:
                    row: dict[str, Any] = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON at {self.path}:{line_number}: {error}"
                    ) from error

                metadata = row.get("metadata")
                if not isinstance(metadata, dict):
                    rejection_counts["invalid_metadata"] += 1
                    continue

                raw_topic = metadata.get("topic")
                if not _nonempty_string(raw_topic):
                    rejection_counts["missing_topic"] += 1
                    continue
                if raw_topic.strip().lower() != normalized_domain:
                    rejection_counts["domain_mismatch"] += 1
                    continue

                required_fields = {
                    "id": row.get("uuid"),
                    "question": row.get("problem"),
                    "answer": row.get("expected_answer"),
                    "subtopic": metadata.get("subtopic"),
                }
                reasons = [
                    f"missing_{name}"
                    for name, value in required_fields.items()
                    if not _nonempty_string(value)
                ]
                if reasons:
                    rejection_counts.update(reasons)
                    continue

                sample = Sample(
                    id=required_fields["id"].strip(),
                    question=required_fields["question"],
                    answer=required_fields["answer"],
                    domain=normalized_domain,
                    subtopic=required_fields["subtopic"].strip(),
                    source="nemotron",
                )
                samples.append(sample)
                subtopics[sample.subtopic] += 1

        report = AuditReport(
            source="nemotron",
            domain=normalized_domain,
            total=total,
            accepted=len(samples),
            rejection_counts=dict(sorted(rejection_counts.items())),
            warning_counts={},
            accepted_distributions={"subtopic": dict(subtopics.most_common())},
        )
        return samples, report


class MatSciBenchAdapter:
    """Adapter for MatSciBench with explicit text-only quality filters."""

    supported_domains = frozenset({"materials"})

    def __init__(
        self,
        path: str | Path = DEFAULT_MATSCIBENCH_PATH,
        *,
        require_solution: bool = True,
        allow_images: bool = False,
        exclude_visual_references: bool = False,
    ) -> None:
        self.path = Path(path)
        self.require_solution = require_solution
        self.allow_images = allow_images
        self.exclude_visual_references = exclude_visual_references

    def load(self, domain: str = "materials") -> list[Sample]:
        samples, _ = self.load_with_report(domain)
        return samples

    def load_with_report(
        self,
        domain: str = "materials",
    ) -> tuple[list[Sample], AuditReport]:
        normalized_domain = domain.lower()
        if normalized_domain not in self.supported_domains:
            raise ValueError(
                f"MatSciBench does not support domain {domain!r}; "
                "supported: materials"
            )

        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise RuntimeError(
                "MatSciBench requires PyArrow. Activate the science-prompt "
                "Conda environment before loading this dataset."
            ) from error

        rows: Sequence[dict[str, Any]] = parquet.read_table(self.path).to_pylist()
        samples: list[Sample] = []
        rejection_counts: Counter[str] = Counter()
        warning_counts: Counter[str] = Counter()
        types: Counter[str] = Counter()
        difficulties: Counter[str] = Counter()
        subtopics: Counter[str] = Counter()

        for row in rows:
            reasons: list[str] = []
            if not _nonempty_string(row.get("qid")):
                reasons.append("missing_id")
            if not _nonempty_string(row.get("question")):
                reasons.append("missing_question")
            if not _nonempty_string(row.get("answer")):
                reasons.append("missing_answer")
            if not _nonempty_string(row.get("primary_category")):
                reasons.append("missing_subtopic")
            if self.require_solution and not _nonempty_string(row.get("solution")):
                reasons.append("missing_solution")
            if not self.allow_images and bool(row.get("image")):
                reasons.append("has_image")

            question = row.get("question")
            has_visual_reference = (
                isinstance(question, str)
                and VISUAL_REFERENCE_PATTERN.search(question) is not None
            )
            if has_visual_reference and self.exclude_visual_references:
                reasons.append("visual_reference")

            if reasons:
                rejection_counts.update(set(reasons))
                continue

            if has_visual_reference:
                warning_counts["visual_reference"] += 1

            solution = row.get("solution")
            sample = Sample(
                id=row["qid"].strip(),
                question=row["question"],
                answer=row["answer"],
                domain="materials",
                subtopic=row["primary_category"].strip(),
                source="matscibench",
                solution=solution if _nonempty_string(solution) else None,
            )
            samples.append(sample)
            types[str(row.get("type") or "<empty>")] += 1
            difficulties[str(row.get("difficulty_level") or "<empty>")] += 1
            subtopics[sample.subtopic] += 1

        report = AuditReport(
            source="matscibench",
            domain="materials",
            total=len(rows),
            accepted=len(samples),
            rejection_counts=dict(sorted(rejection_counts.items())),
            warning_counts=dict(sorted(warning_counts.items())),
            accepted_distributions={
                "type": dict(types.most_common()),
                "difficulty": dict(difficulties.most_common()),
                "subtopic": dict(subtopics.most_common()),
            },
        )
        return samples, report


def _print_report(report: AuditReport) -> None:
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


def _print_samples(
    samples: Sequence[Sample],
    *,
    count: int,
    seed: int | None,
) -> None:
    if count < 0:
        raise ValueError("show-samples must be non-negative")
    chosen = random.Random(seed).sample(samples, k=min(count, len(samples)))
    for index, sample in enumerate(chosen, start=1):
        print(f"\n=== accepted sample {index} ===")
        print(json.dumps(asdict(sample), ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit benchmark adapters and print accepted samples."
    )
    parser.add_argument(
        "--source",
        choices=("matscibench", "nemotron"),
        default="matscibench",
    )
    parser.add_argument("--domain", choices=SUPPORTED_DOMAINS)
    parser.add_argument("--show-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument(
        "--exclude-visual-references",
        action="store_true",
        help="Reject questions mentioning a figure, diagram, or table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source == "matscibench":
        domain = args.domain or "materials"
        adapter = MatSciBenchAdapter(
            exclude_visual_references=args.exclude_visual_references
        )
    else:
        if args.domain not in NemotronAdapter.supported_domains:
            raise SystemExit("Nemotron requires --domain biology or chemistry")
        domain = args.domain
        adapter = NemotronAdapter()

    samples, report = adapter.load_with_report(domain)
    _print_report(report)
    _print_samples(samples, count=args.show_samples, seed=args.seed)


if __name__ == "__main__":
    main()
