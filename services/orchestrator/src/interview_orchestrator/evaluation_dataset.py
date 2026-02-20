from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from interview_orchestrator.agent_state import AgentState, validate_agent_state
from interview_orchestrator.state_steps import score_aggregation_node


@dataclass(frozen=True)
class OfflineEvaluationSample:
    sample_id: str
    label: str
    state: AgentState


@dataclass(frozen=True)
class OfflineEvaluationResult:
    sample_id: str
    label: str
    final_score: float
    correctness_score: float
    performance_score: float
    readability_score: float
    security_score: float


def build_offline_dataset() -> list[OfflineEvaluationSample]:
    return [
        OfflineEvaluationSample(
            sample_id="correct_solution",
            label="Correct solution",
            state=_build_sample_state(
                task_id="offline-correct-1",
                runtime_samples_ms=[90, 110, 120, 80, 100],
                pylint_score=9.6,
                complexity_score=2.0,
                security_warning_count=0,
            ),
        ),
        OfflineEvaluationSample(
            sample_id="inefficient_solution",
            label="Inefficient solution",
            state=_build_sample_state(
                task_id="offline-inefficient-1",
                runtime_samples_ms=[2500, 2600, 2400, 2550, 2700],
                pylint_score=9.0,
                complexity_score=3.0,
                security_warning_count=0,
            ),
        ),
        OfflineEvaluationSample(
            sample_id="security_bad_solution",
            label="Security-bad solution",
            state=_build_sample_state(
                task_id="offline-security-1",
                runtime_samples_ms=[150, 180, 200, 170, 160],
                pylint_score=9.0,
                complexity_score=3.0,
                security_warning_count=5,
            ),
        ),
    ]


def evaluate_offline_dataset(
    samples: Sequence[OfflineEvaluationSample] | None = None,
) -> list[OfflineEvaluationResult]:
    dataset = list(samples) if samples is not None else build_offline_dataset()
    results: list[OfflineEvaluationResult] = []

    for sample in dataset:
        scored_state = score_aggregation_node(sample.state)
        score_breakdown_raw = scored_state.get("score_breakdown")
        if not isinstance(score_breakdown_raw, Mapping):
            raise ValueError(f"score_breakdown is missing for sample {sample.sample_id!r}.")

        results.append(
            OfflineEvaluationResult(
                sample_id=sample.sample_id,
                label=sample.label,
                final_score=_read_score(score_breakdown_raw, "aggregated_score"),
                correctness_score=_read_score(score_breakdown_raw, "correctness_score"),
                performance_score=_read_score(score_breakdown_raw, "performance_score"),
                readability_score=_read_score(score_breakdown_raw, "readability_score"),
                security_score=_read_score(score_breakdown_raw, "security_score"),
            )
        )

    return results


def check_scoring_consistency(results: Sequence[OfflineEvaluationResult]) -> bool:
    indexed = {result.sample_id: result for result in results}
    if len(indexed) != len(results):
        return False

    correct = indexed.get("correct_solution")
    inefficient = indexed.get("inefficient_solution")
    security_bad = indexed.get("security_bad_solution")
    if correct is None or inefficient is None or security_bad is None:
        return False

    same_correctness = _is_close(
        correct.correctness_score, inefficient.correctness_score
    ) and _is_close(correct.correctness_score, security_bad.correctness_score)
    return (
        same_correctness
        and correct.final_score > inefficient.final_score > security_bad.final_score
        and inefficient.performance_score < correct.performance_score
        and security_bad.security_score < correct.security_score
        and security_bad.security_score <= 20.0
    )


def run_offline_dataset_consistency_check() -> bool:
    return check_scoring_consistency(evaluate_offline_dataset())


def _build_sample_state(
    *,
    task_id: str,
    runtime_samples_ms: list[int],
    pylint_score: float,
    complexity_score: float,
    security_warning_count: int,
) -> AgentState:
    total = len(runtime_samples_ms)
    if total == 0:
        raise ValueError("runtime_samples_ms must contain at least one runtime value.")

    avg_runtime_ms = int(round(sum(runtime_samples_ms) / total))
    security_warnings = [
        {
            "test_id": f"B{101 + index}",
            "issue_text": "Potentially unsafe pattern.",
            "line_number": index + 1,
        }
        for index in range(max(0, security_warning_count))
    ]
    return validate_agent_state(
        {
            "user_id": "offline-eval",
            "task_id": task_id,
            "language": "python",
            "code": "print('offline-eval')",
            "metrics": {
                "runtime_ms": avg_runtime_ms,
                "memory_usage_kb": 1024,
                "exit_code": 0,
                "stdout": "ok\n",
                "stderr": "",
                "timed_out": False,
            },
            "test_results": {
                "total": total,
                "passed": total,
                "failed": 0,
                "first_failed_report": None,
                "case_results": [
                    {
                        "case_index": index,
                        "runtime_ms": runtime_ms,
                        "passed": True,
                    }
                    for index, runtime_ms in enumerate(runtime_samples_ms)
                ],
            },
            "static_analysis": {
                "language": "python",
                "pylint_score": pylint_score,
                "complexity_score": complexity_score,
                "security_warnings": security_warnings,
                "security_warnings_summary": f"Security warnings: {len(security_warnings)}",
                "pylint_warnings": [],
                "tool_errors": [],
            },
        }
    )


def _read_score(score_breakdown: Mapping[str, object], field_name: str) -> float:
    value = score_breakdown.get(field_name)
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric.")
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"{field_name} must be numeric.")


def _is_close(left: float, right: float, tolerance: float = 0.0001) -> bool:
    return abs(left - right) <= tolerance


__all__ = [
    "OfflineEvaluationSample",
    "OfflineEvaluationResult",
    "build_offline_dataset",
    "evaluate_offline_dataset",
    "check_scoring_consistency",
    "run_offline_dataset_consistency_check",
]
