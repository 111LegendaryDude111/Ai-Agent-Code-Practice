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


class AgentState(TypedDict):
    user_id: str
    task_id: str
    language: str
    code: str
    task_prompt: NotRequired[str]
    task_category: NotRequired[str]
    task_difficulty: NotRequired[int]
    test_cases: NotRequired[str]
    stdin_data: NotRequired[str]
    main_class_name: NotRequired[str]
    execution_result: NotRequired[ExecutionResultState]
    test_results: NotRequired[TestResultsState]
    metrics: NotRequired[MetricsState]
    static_analysis: NotRequired[StaticAnalysisState]
    llm_review: NotRequired[dict[str, object]]
    score_breakdown: NotRequired[dict[str, object]]
    skill_profile: NotRequired[dict[str, object]]
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
    if "test_results" in normalized_state:
        _validate_test_results_state(normalized_state["test_results"])
    if "static_analysis" in normalized_state:
        _validate_static_analysis_state(normalized_state["static_analysis"])

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
