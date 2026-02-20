from __future__ import annotations

from collections.abc import Mapping
from typing import NotRequired, TypedDict, cast

SUPPORTED_LANGUAGES: set[str] = {"python", "go", "java", "cpp"}


class ExecutionResultState(TypedDict):
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool


class MetricsState(TypedDict):
    runtime_ms: int
    memory_usage_kb: int | None
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool


class ObservabilityMetricsState(TypedDict):
    avg_runtime_ms: float
    fail_rate: float
    cost_per_submission: float


class TestResultsState(TypedDict):
    total: int
    passed: int
    failed: int
    first_failed_report: str | None
    case_results: list[dict[str, object]]


class StaticAnalysisState(TypedDict):
    language: str
    pylint_score: float | None
    complexity_score: float | None
    security_warnings: list[dict[str, object]]
    security_warnings_summary: str
    pylint_warnings: list[dict[str, object]]
    tool_errors: list[str]


class RetryBranchState(TypedDict):
    should_retry: bool
    retries_used: int
    retries_remaining: int
    max_retries: int
    reason: str


class AdaptiveDifficultyBranchState(TypedDict):
    action: str
    current_difficulty: int
    next_difficulty: int
    reason: str


class HintBranchState(TypedDict):
    should_show_hint: bool
    hints: list[str]


class BranchingState(TypedDict):
    retry: RetryBranchState
    adaptive_difficulty: AdaptiveDifficultyBranchState
    hint: HintBranchState
    next_node: str


class AgentState(TypedDict):
    user_id: str
    task_id: str
    language: str
    code: str
    task_prompt: NotRequired[str]
    task_category: NotRequired[str]
    task_difficulty: NotRequired[int]
    recommended_difficulty: NotRequired[int]
    retry_count: NotRequired[int]
    test_cases: NotRequired[str]
    stdin_data: NotRequired[str]
    main_class_name: NotRequired[str]
    execution_result: NotRequired[ExecutionResultState]
    test_results: NotRequired[TestResultsState]
    metrics: NotRequired[MetricsState]
    observability_metrics: NotRequired[ObservabilityMetricsState]
    static_analysis: NotRequired[StaticAnalysisState]
    llm_review: NotRequired[dict[str, object]]
    score_breakdown: NotRequired[dict[str, object]]
    skill_profile: NotRequired[dict[str, object]]
    branching: NotRequired[BranchingState]
    final_score: NotRequired[float]


class AgentStateValidationError(ValueError):
    pass


def validate_agent_state(state: Mapping[str, object]) -> AgentState:
    if not isinstance(state, Mapping):
        raise AgentStateValidationError("state must be a mapping.")

    normalized_state = dict(state)
    _validate_required_string(normalized_state, "user_id")
    _validate_required_string(normalized_state, "task_id")
    _validate_required_string(normalized_state, "language")
    _validate_required_string(normalized_state, "code")

    language = str(normalized_state["language"])
    if language not in SUPPORTED_LANGUAGES:
        raise AgentStateValidationError(
            f"state.language must be one of {sorted(SUPPORTED_LANGUAGES)}, got {language!r}."
        )

    _validate_optional_string(normalized_state, "test_cases")
    _validate_optional_string(normalized_state, "stdin_data")
    _validate_optional_string(normalized_state, "main_class_name")
    _validate_optional_string(normalized_state, "task_prompt")
    _validate_optional_string(normalized_state, "task_category")
    _validate_optional_positive_int(normalized_state, "task_difficulty")
    _validate_optional_positive_int(normalized_state, "recommended_difficulty")
    _validate_optional_non_negative_int(normalized_state, "retry_count")
    _validate_optional_dict(normalized_state, "llm_review")
    _validate_optional_dict(normalized_state, "score_breakdown")
    _validate_optional_dict(normalized_state, "skill_profile")

    if "final_score" in normalized_state and not isinstance(
        normalized_state["final_score"],
        int | float,
    ):
        raise AgentStateValidationError("state.final_score must be a number when present.")

    if "execution_result" in normalized_state:
        _validate_execution_result_state(normalized_state["execution_result"])
    if "metrics" in normalized_state:
        _validate_metrics_state(normalized_state["metrics"])
    if "observability_metrics" in normalized_state:
        _validate_observability_metrics_state(normalized_state["observability_metrics"])
    if "test_results" in normalized_state:
        _validate_test_results_state(normalized_state["test_results"])
    if "static_analysis" in normalized_state:
        _validate_static_analysis_state(normalized_state["static_analysis"])
    if "branching" in normalized_state:
        _validate_branching_state(normalized_state["branching"])

    return cast(AgentState, normalized_state)


def _validate_required_string(state: dict[str, object], key: str) -> None:
    value = state.get(key)
    if not isinstance(value, str) or value == "":
        raise AgentStateValidationError(f"state.{key} must be a non-empty string.")


def _validate_optional_string(state: dict[str, object], key: str) -> None:
    if key in state and not isinstance(state[key], str):
        raise AgentStateValidationError(f"state.{key} must be a string when present.")


def _validate_optional_dict(state: dict[str, object], key: str) -> None:
    if key in state and not isinstance(state[key], dict):
        raise AgentStateValidationError(f"state.{key} must be an object when present.")


def _validate_optional_positive_int(state: dict[str, object], key: str) -> None:
    if key not in state:
        return

    value = state[key]
    if not isinstance(value, int) or value < 1:
        raise AgentStateValidationError(f"state.{key} must be an integer >= 1 when present.")


def _validate_optional_non_negative_int(state: dict[str, object], key: str) -> None:
    if key not in state:
        return

    value = state[key]
    if not isinstance(value, int) or value < 0:
        raise AgentStateValidationError(f"state.{key} must be an integer >= 0 when present.")


def _validate_execution_result_state(value: object) -> None:
    if not isinstance(value, dict):
        raise AgentStateValidationError("state.execution_result must be an object.")

    _validate_required_field(value, "stdout", str, "state.execution_result.stdout")
    _validate_required_field(value, "stderr", str, "state.execution_result.stderr")
    _validate_required_field(value, "timed_out", bool, "state.execution_result.timed_out")
    _validate_int_or_none(value, "exit_code", "state.execution_result.exit_code")


def _validate_metrics_state(value: object) -> None:
    if not isinstance(value, dict):
        raise AgentStateValidationError("state.metrics must be an object.")

    _validate_required_field(value, "runtime_ms", int, "state.metrics.runtime_ms")
    _validate_int_or_none(value, "memory_usage_kb", "state.metrics.memory_usage_kb")
    _validate_int_or_none(value, "exit_code", "state.metrics.exit_code")
    _validate_required_field(value, "stdout", str, "state.metrics.stdout")
    _validate_required_field(value, "stderr", str, "state.metrics.stderr")
    _validate_required_field(value, "timed_out", bool, "state.metrics.timed_out")


def _validate_observability_metrics_state(value: object) -> None:
    if not isinstance(value, dict):
        raise AgentStateValidationError("state.observability_metrics must be an object.")

    avg_runtime_ms_raw = value.get("avg_runtime_ms")
    fail_rate_raw = value.get("fail_rate")
    cost_per_submission_raw = value.get("cost_per_submission")

    if isinstance(avg_runtime_ms_raw, bool) or not isinstance(avg_runtime_ms_raw, int | float):
        raise AgentStateValidationError(
            "state.observability_metrics.avg_runtime_ms must be number."
        )
    if isinstance(fail_rate_raw, bool) or not isinstance(fail_rate_raw, int | float):
        raise AgentStateValidationError("state.observability_metrics.fail_rate must be number.")
    if isinstance(cost_per_submission_raw, bool) or not isinstance(
        cost_per_submission_raw,
        int | float,
    ):
        raise AgentStateValidationError(
            "state.observability_metrics.cost_per_submission must be number."
        )

    avg_runtime_ms = float(avg_runtime_ms_raw)
    fail_rate = float(fail_rate_raw)
    cost_per_submission = float(cost_per_submission_raw)

    if avg_runtime_ms < 0.0:
        raise AgentStateValidationError("state.observability_metrics.avg_runtime_ms must be >= 0.")
    if not 0.0 <= fail_rate <= 1.0:
        raise AgentStateValidationError(
            "state.observability_metrics.fail_rate must be in range [0, 1]."
        )
    if cost_per_submission < 0.0:
        raise AgentStateValidationError(
            "state.observability_metrics.cost_per_submission must be >= 0."
        )


def _validate_test_results_state(value: object) -> None:
    if not isinstance(value, dict):
        raise AgentStateValidationError("state.test_results must be an object.")

    _validate_required_field(value, "total", int, "state.test_results.total")
    _validate_required_field(value, "passed", int, "state.test_results.passed")
    _validate_required_field(value, "failed", int, "state.test_results.failed")
    _validate_required_nullable_string(
        value,
        "first_failed_report",
        "state.test_results.first_failed_report",
    )

    case_results = value.get("case_results")
    if not isinstance(case_results, list):
        raise AgentStateValidationError("state.test_results.case_results must be a list.")
    if not all(isinstance(case_result, dict) for case_result in case_results):
        raise AgentStateValidationError("state.test_results.case_results items must be objects.")


def _validate_static_analysis_state(value: object) -> None:
    if not isinstance(value, dict):
        raise AgentStateValidationError("state.static_analysis must be an object.")

    _validate_required_field(value, "language", str, "state.static_analysis.language")
    _validate_float_or_none(value, "pylint_score", "state.static_analysis.pylint_score")
    _validate_float_or_none(value, "complexity_score", "state.static_analysis.complexity_score")
    _validate_required_field(
        value,
        "security_warnings_summary",
        str,
        "state.static_analysis.security_warnings_summary",
    )

    security_warnings = value.get("security_warnings")
    if not isinstance(security_warnings, list):
        raise AgentStateValidationError("state.static_analysis.security_warnings must be a list.")
    if not all(isinstance(warning, dict) for warning in security_warnings):
        raise AgentStateValidationError(
            "state.static_analysis.security_warnings items must be objects."
        )

    pylint_warnings = value.get("pylint_warnings")
    if not isinstance(pylint_warnings, list):
        raise AgentStateValidationError("state.static_analysis.pylint_warnings must be a list.")
    if not all(isinstance(warning, dict) for warning in pylint_warnings):
        raise AgentStateValidationError(
            "state.static_analysis.pylint_warnings items must be objects."
        )

    tool_errors = value.get("tool_errors")
    if not isinstance(tool_errors, list):
        raise AgentStateValidationError("state.static_analysis.tool_errors must be a list.")
    if not all(isinstance(tool_error, str) for tool_error in tool_errors):
        raise AgentStateValidationError("state.static_analysis.tool_errors items must be strings.")


def _validate_branching_state(value: object) -> None:
    if not isinstance(value, dict):
        raise AgentStateValidationError("state.branching must be an object.")

    retry = value.get("retry")
    if not isinstance(retry, dict):
        raise AgentStateValidationError("state.branching.retry must be an object.")
    _validate_required_field(retry, "should_retry", bool, "state.branching.retry.should_retry")
    _validate_required_field(retry, "retries_used", int, "state.branching.retry.retries_used")
    _validate_required_field(
        retry,
        "retries_remaining",
        int,
        "state.branching.retry.retries_remaining",
    )
    _validate_required_field(retry, "max_retries", int, "state.branching.retry.max_retries")
    _validate_required_field(retry, "reason", str, "state.branching.retry.reason")

    retries_used = retry["retries_used"]
    retries_remaining = retry["retries_remaining"]
    max_retries = retry["max_retries"]
    if retries_used < 0:
        raise AgentStateValidationError("state.branching.retry.retries_used must be >= 0.")
    if retries_remaining < 0:
        raise AgentStateValidationError("state.branching.retry.retries_remaining must be >= 0.")
    if max_retries < 0:
        raise AgentStateValidationError("state.branching.retry.max_retries must be >= 0.")
    if retries_used > max_retries:
        raise AgentStateValidationError(
            "state.branching.retry.retries_used must be <= state.branching.retry.max_retries."
        )
    if retries_remaining > max_retries:
        raise AgentStateValidationError(
            "state.branching.retry.retries_remaining must be <= state.branching.retry.max_retries."
        )

    adaptive_difficulty = value.get("adaptive_difficulty")
    if not isinstance(adaptive_difficulty, dict):
        raise AgentStateValidationError("state.branching.adaptive_difficulty must be an object.")

    _validate_required_field(
        adaptive_difficulty,
        "action",
        str,
        "state.branching.adaptive_difficulty.action",
    )
    _validate_required_field(
        adaptive_difficulty,
        "current_difficulty",
        int,
        "state.branching.adaptive_difficulty.current_difficulty",
    )
    _validate_required_field(
        adaptive_difficulty,
        "next_difficulty",
        int,
        "state.branching.adaptive_difficulty.next_difficulty",
    )
    _validate_required_field(
        adaptive_difficulty,
        "reason",
        str,
        "state.branching.adaptive_difficulty.reason",
    )
    if adaptive_difficulty["action"] not in {"increase", "decrease", "keep"}:
        raise AgentStateValidationError(
            "state.branching.adaptive_difficulty.action must be one of "
            "'increase', 'decrease', or 'keep'."
        )
    if adaptive_difficulty["current_difficulty"] < 1:
        raise AgentStateValidationError(
            "state.branching.adaptive_difficulty.current_difficulty must be >= 1."
        )
    if adaptive_difficulty["next_difficulty"] < 1:
        raise AgentStateValidationError(
            "state.branching.adaptive_difficulty.next_difficulty must be >= 1."
        )

    hint = value.get("hint")
    if not isinstance(hint, dict):
        raise AgentStateValidationError("state.branching.hint must be an object.")
    _validate_required_field(
        hint,
        "should_show_hint",
        bool,
        "state.branching.hint.should_show_hint",
    )
    hints = hint.get("hints")
    if not isinstance(hints, list):
        raise AgentStateValidationError("state.branching.hint.hints must be a list.")
    if not all(isinstance(hint_item, str) for hint_item in hints):
        raise AgentStateValidationError("state.branching.hint.hints items must be strings.")

    next_node = value.get("next_node")
    if next_node not in {"retry_submission", "retry_with_hint", "show_hint", "complete"}:
        raise AgentStateValidationError(
            "state.branching.next_node must be one of "
            "'retry_submission', 'retry_with_hint', 'show_hint', or 'complete'."
        )


def _validate_required_field(
    payload: dict[str, object],
    key: str,
    expected_type: type,
    field_name: str,
) -> None:
    value = payload.get(key)
    if not isinstance(value, expected_type):
        raise AgentStateValidationError(f"{field_name} must be {expected_type.__name__}.")


def _validate_required_nullable_string(
    payload: dict[str, object],
    key: str,
    field_name: str,
) -> None:
    value = payload.get(key)
    if value is not None and not isinstance(value, str):
        raise AgentStateValidationError(f"{field_name} must be string or null.")


def _validate_int_or_none(payload: dict[str, object], key: str, field_name: str) -> None:
    value = payload.get(key)
    if value is not None and not isinstance(value, int):
        raise AgentStateValidationError(f"{field_name} must be int or null.")


def _validate_float_or_none(payload: dict[str, object], key: str, field_name: str) -> None:
    value = payload.get(key)
    if value is not None and not isinstance(value, int | float):
        raise AgentStateValidationError(f"{field_name} must be number or null.")
