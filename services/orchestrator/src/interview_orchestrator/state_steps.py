from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypedDict, cast

from interview_orchestrator.agent_state import AgentState, validate_agent_state
from interview_orchestrator.sandbox_runner import (
    DockerSandboxRunner,
    SandboxExecutionResult,
    SandboxLimits,
)
from interview_orchestrator.static_analysis import (
    PythonStaticAnalysisResult,
    PythonStaticAnalyzer,
    format_security_warnings,
)
from interview_orchestrator.test_runner import (
    PredefinedTestCaseResult,
    PredefinedTestRunner,
    PredefinedTestRunResult,
)


class SandboxStepExecutor(Protocol):
    def execute(
        self,
        language: str,
        source_code: str,
        limits: SandboxLimits | None = None,
        main_class_name: str = "Main",
        stdin_data: str = "",
    ) -> SandboxExecutionResult: ...


class TestStepRunner(Protocol):
    def run_json(
        self,
        language: str,
        source_code: str,
        test_cases_json: str,
        limits: SandboxLimits | None = None,
        main_class_name: str = "Main",
    ) -> PredefinedTestRunResult: ...


class StaticAnalysisExecutor(Protocol):
    def analyze(
        self, source_code: str, filename: str = "main.py"
    ) -> PythonStaticAnalysisResult: ...


class _TaskTemplate(TypedDict):
    prompt: str
    test_cases_json: str
    category: str
    difficulty: int


class _ScoreBucket(TypedDict):
    attempts: int
    average_score: float
    best_score: float
    last_score: float


@dataclass(frozen=True)
class GeneratedTask:
    task_id: str
    prompt: str
    test_cases_json: str
    category: str = "general"
    difficulty: int = 1


@dataclass(frozen=True)
class ScoreAggregationWeights:
    test_weight: float = 0.5
    runtime_weight: float = 0.15
    static_weight: float = 0.15
    llm_weight: float = 0.2


class TaskGenerationExecutor(Protocol):
    def generate_task(
        self,
        user_id: str,
        language: str,
        task_id: str,
    ) -> GeneratedTask: ...


class LLMReviewExecutor(Protocol):
    def review(self, state: AgentState) -> dict[str, object]: ...


class ProfileUpdateExecutor(Protocol):
    def update_profile(
        self,
        user_id: str,
        language: str,
        category: str,
        final_score: float,
        current_profile: Mapping[str, object] | None = None,
    ) -> dict[str, object]: ...


class DeterministicTaskGenerator:
    _TASKS_BY_LANGUAGE: dict[str, _TaskTemplate] = {
        "python": {
            "prompt": (
                "Read two integers from stdin and print their sum. "
                "Use exact formatting with a trailing newline."
            ),
            "test_cases_json": (
                '[{"input":"1 2\\n","output":"3\\n"},{"input":"-10 7\\n","output":"-3\\n"}]'
            ),
            "category": "math",
            "difficulty": 1,
        },
        "go": {
            "prompt": "Read two integers from stdin and print their sum.",
            "test_cases_json": (
                '[{"input":"9 8\\n","output":"17\\n"},{"input":"0 0\\n","output":"0\\n"}]'
            ),
            "category": "math",
            "difficulty": 1,
        },
        "java": {
            "prompt": "Read two integers from stdin and print their sum.",
            "test_cases_json": (
                '[{"input":"4 5\\n","output":"9\\n"},{"input":"100 -1\\n","output":"99\\n"}]'
            ),
            "category": "math",
            "difficulty": 1,
        },
        "cpp": {
            "prompt": "Read two integers from stdin and print their sum.",
            "test_cases_json": (
                '[{"input":"3 7\\n","output":"10\\n"},{"input":"12 30\\n","output":"42\\n"}]'
            ),
            "category": "math",
            "difficulty": 1,
        },
    }

    def generate_task(
        self,
        user_id: str,
        language: str,
        task_id: str,
    ) -> GeneratedTask:
        template = self._TASKS_BY_LANGUAGE.get(language)
        if template is None:
            template = self._TASKS_BY_LANGUAGE["python"]

        prompt = template["prompt"]
        test_cases_json = template["test_cases_json"]
        category = template["category"]
        difficulty = template["difficulty"]

        return GeneratedTask(
            task_id=f"{task_id}:{user_id}:{language}",
            prompt=prompt,
            test_cases_json=test_cases_json,
            category=category,
            difficulty=max(1, difficulty),
        )


class HeuristicLLMReviewer:
    def review(self, state: AgentState) -> dict[str, object]:
        test_score = _compute_test_score(state)
        runtime_score = _compute_runtime_score(state)
        static_score = _compute_static_score(state)
        weighted_score = round(
            (test_score * 0.6) + (runtime_score * 0.2) + (static_score * 0.2),
            2,
        )

        strengths: list[str] = []
        issues: list[str] = []

        if test_score >= 80:
            strengths.append("Solution passes most test cases.")
        else:
            issues.append("Test coverage indicates logical issues in the solution.")

        execution_result = cast(dict[str, object], state.get("execution_result", {}))
        if execution_result.get("timed_out") is True:
            issues.append("Execution timed out.")
        elif execution_result.get("exit_code") == 0:
            strengths.append("Program exits successfully.")
        else:
            issues.append("Program exited with a non-zero code.")

        static_analysis = cast(dict[str, object], state.get("static_analysis", {}))
        security_warnings = static_analysis.get("security_warnings", [])
        if isinstance(security_warnings, list) and len(security_warnings) > 0:
            issues.append(f"Security warnings detected: {len(security_warnings)}.")
        else:
            strengths.append("No security warnings reported.")

        if len(strengths) == 0:
            strengths.append("Implementation has a valid submission structure.")
        if len(issues) == 0:
            issues.append("No critical issues detected.")

        summary = (
            "Heuristic review based on tests, runtime, and static analysis. "
            f"Estimated quality score: {weighted_score}/100."
        )

        return {
            "summary": summary,
            "strengths": strengths,
            "issues": issues,
            "score": weighted_score,
            "reviewer": "heuristic",
        }


class InMemoryProfileUpdater:
    def update_profile(
        self,
        user_id: str,
        language: str,
        category: str,
        final_score: float,
        current_profile: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        profile = _normalize_profile(current_profile)
        language_scores = _normalize_score_bucket_map(profile.get("language_scores"))
        category_scores = _normalize_score_bucket_map(profile.get("category_scores"))
        recent_scores = _normalize_recent_scores(profile.get("recent_scores"))

        normalized_language = language.strip().lower()
        normalized_category = category.strip().lower() or "general"
        rounded_score = round(final_score, 2)

        language_scores[normalized_language] = _apply_new_score_to_bucket(
            language_scores.get(normalized_language),
            rounded_score,
        )
        category_scores[normalized_category] = _apply_new_score_to_bucket(
            category_scores.get(normalized_category),
            rounded_score,
        )

        recent_scores.append(rounded_score)
        max_recent_scores = 20
        if len(recent_scores) > max_recent_scores:
            recent_scores = recent_scores[-max_recent_scores:]

        profile["version"] = _normalize_profile_version(profile.get("version"))
        profile["user_id"] = user_id
        profile["language_scores"] = language_scores
        profile["category_scores"] = category_scores
        profile["recent_scores"] = recent_scores
        profile["last_score"] = rounded_score

        return profile


def generate_task_node(
    state: AgentState,
    task_generator: TaskGenerationExecutor | None = None,
) -> AgentState:
    validated_state = validate_agent_state(state)
    generator = task_generator or DeterministicTaskGenerator()
    generated_task = generator.generate_task(
        user_id=validated_state["user_id"],
        language=validated_state["language"],
        task_id=validated_state["task_id"],
    )

    updated_state: dict[str, object] = dict(validated_state)
    if _is_blank_string(updated_state.get("test_cases")):
        updated_state["test_cases"] = generated_task.test_cases_json
    if _is_blank_string(updated_state.get("task_prompt")):
        updated_state["task_prompt"] = generated_task.prompt
    if _is_blank_string(updated_state.get("task_category")):
        updated_state["task_category"] = generated_task.category

    difficulty = updated_state.get("task_difficulty")
    if not isinstance(difficulty, int) or difficulty < 1:
        updated_state["task_difficulty"] = generated_task.difficulty

    return validate_agent_state(updated_state)


def execute_sandbox_node(
    state: AgentState,
    sandbox_executor: SandboxStepExecutor | None = None,
    limits: SandboxLimits | None = None,
) -> AgentState:
    return execute_sandbox_step(
        state=state,
        sandbox_executor=sandbox_executor,
        limits=limits,
    )


def run_tests_node(
    state: AgentState,
    test_runner: TestStepRunner | None = None,
    limits: SandboxLimits | None = None,
) -> AgentState:
    validated_state = validate_agent_state(state)
    test_cases_json = validated_state.get("test_cases")
    if not isinstance(test_cases_json, str) or test_cases_json == "":
        updated_state: dict[str, object] = dict(validated_state)
        updated_state["test_results"] = _empty_test_results_state()
        return validate_agent_state(updated_state)

    return run_predefined_tests_step(
        state=validated_state,
        test_runner=test_runner,
        limits=limits,
    )


def static_analysis_node(
    state: AgentState,
    static_analyzer: StaticAnalysisExecutor | None = None,
) -> AgentState:
    return run_python_static_analysis_step(
        state=state,
        static_analyzer=static_analyzer,
    )


def llm_review_node(
    state: AgentState,
    llm_reviewer: LLMReviewExecutor | None = None,
) -> AgentState:
    validated_state = validate_agent_state(state)
    reviewer = llm_reviewer or HeuristicLLMReviewer()
    review = reviewer.review(validated_state)
    if not isinstance(review, dict):
        raise ValueError("llm_reviewer must return a dictionary payload.")

    updated_state: dict[str, object] = dict(validated_state)
    updated_state["llm_review"] = review
    return validate_agent_state(updated_state)


def score_aggregation_node(
    state: AgentState,
    weights: ScoreAggregationWeights | None = None,
) -> AgentState:
    validated_state = validate_agent_state(state)
    normalized_weights = _normalize_weights(weights or ScoreAggregationWeights())

    test_score = _compute_test_score(validated_state)
    runtime_score = _compute_runtime_score(validated_state)
    static_score = _compute_static_score(validated_state)
    llm_score = _compute_llm_review_score(validated_state)

    final_score = round(
        (test_score * normalized_weights["test_weight"])
        + (runtime_score * normalized_weights["runtime_weight"])
        + (static_score * normalized_weights["static_weight"])
        + (llm_score * normalized_weights["llm_weight"]),
        2,
    )

    updated_state: dict[str, object] = dict(validated_state)
    updated_state["score_breakdown"] = {
        "test_score": round(test_score, 2),
        "runtime_score": round(runtime_score, 2),
        "static_score": round(static_score, 2),
        "llm_score": round(llm_score, 2),
        "weights": normalized_weights,
        "aggregated_score": final_score,
    }
    updated_state["final_score"] = final_score
    return validate_agent_state(updated_state)


def update_profile_node(
    state: AgentState,
    profile_updater: ProfileUpdateExecutor | None = None,
) -> AgentState:
    validated_state = validate_agent_state(state)
    score_value = validated_state.get("final_score")
    if isinstance(score_value, int | float):
        final_score = float(score_value)
        current_state = validated_state
    else:
        current_state = score_aggregation_node(validated_state)
        final_score = float(current_state["final_score"])

    current_profile = current_state.get("skill_profile")
    category = str(current_state.get("task_category") or "general")
    updater = profile_updater or InMemoryProfileUpdater()
    updated_profile = updater.update_profile(
        user_id=current_state["user_id"],
        language=current_state["language"],
        category=category,
        final_score=final_score,
        current_profile=(
            cast(dict[str, object], current_profile) if isinstance(current_profile, dict) else None
        ),
    )

    updated_state: dict[str, object] = dict(current_state)
    updated_state["skill_profile"] = updated_profile
    return validate_agent_state(updated_state)


def execute_sandbox_step(
    state: AgentState,
    sandbox_executor: SandboxStepExecutor | None = None,
    limits: SandboxLimits | None = None,
) -> AgentState:
    validated_state = validate_agent_state(state)
    executor = sandbox_executor or DockerSandboxRunner()
    main_class_name = validated_state.get("main_class_name", "Main")
    stdin_data = validated_state.get("stdin_data", "")

    execution = executor.execute(
        language=validated_state["language"],
        source_code=validated_state["code"],
        limits=limits,
        main_class_name=main_class_name,
        stdin_data=stdin_data,
    )

    runtime_ms = max(0, int(round(execution.duration_seconds * 1000)))
    updated_state: dict[str, object] = dict(validated_state)
    updated_state["execution_result"] = {
        "exit_code": execution.exit_code,
        "stdout": execution.stdout,
        "stderr": execution.stderr,
        "timed_out": execution.timed_out,
    }
    updated_state["metrics"] = {
        "runtime_ms": runtime_ms,
        "memory_usage_kb": execution.memory_usage_kb,
        "exit_code": execution.exit_code,
        "stdout": execution.stdout,
        "stderr": execution.stderr,
        "timed_out": execution.timed_out,
    }
    return validate_agent_state(updated_state)


def run_predefined_tests_step(
    state: AgentState,
    test_runner: TestStepRunner | None = None,
    limits: SandboxLimits | None = None,
) -> AgentState:
    validated_state = validate_agent_state(state)
    test_cases_json = validated_state.get("test_cases")
    if not isinstance(test_cases_json, str) or test_cases_json == "":
        raise ValueError("state.test_cases must be a non-empty JSON string to run tests.")

    runner = test_runner or PredefinedTestRunner()
    main_class_name = validated_state.get("main_class_name", "Main")
    test_results = runner.run_json(
        language=validated_state["language"],
        source_code=validated_state["code"],
        test_cases_json=test_cases_json,
        limits=limits,
        main_class_name=main_class_name,
    )

    updated_state: dict[str, object] = dict(validated_state)
    updated_state["test_results"] = {
        "total": test_results.total,
        "passed": test_results.passed,
        "failed": test_results.failed,
        "first_failed_report": test_results.first_failed_report,
        "case_results": [
            _serialize_test_case_result(case_result) for case_result in test_results.case_results
        ],
    }
    return validate_agent_state(updated_state)


def run_python_static_analysis_step(
    state: AgentState,
    static_analyzer: StaticAnalysisExecutor | None = None,
) -> AgentState:
    validated_state = validate_agent_state(state)
    language = validated_state["language"]
    updated_state: dict[str, object] = dict(validated_state)

    if language != "python":
        updated_state["static_analysis"] = {
            "language": language,
            "pylint_score": None,
            "complexity_score": None,
            "security_warnings": [],
            "security_warnings_summary": "Skipped: static analysis is only enabled for Python.",
            "pylint_warnings": [],
            "tool_errors": [],
        }
        return validate_agent_state(updated_state)

    analyzer = static_analyzer or PythonStaticAnalyzer()
    analysis_result = analyzer.analyze(validated_state["code"])
    updated_state["static_analysis"] = {
        "language": analysis_result.language,
        "pylint_score": analysis_result.pylint_score,
        "complexity_score": analysis_result.complexity_score,
        "security_warnings": [
            {
                "test_id": warning.test_id,
                "test_name": warning.test_name,
                "issue_severity": warning.issue_severity,
                "issue_confidence": warning.issue_confidence,
                "issue_text": warning.issue_text,
                "line_number": warning.line_number,
            }
            for warning in analysis_result.security_warnings
        ],
        "security_warnings_summary": format_security_warnings(analysis_result.security_warnings),
        "pylint_warnings": [
            {
                "message_id": warning.message_id,
                "symbol": warning.symbol,
                "message": warning.message,
                "path": warning.path,
                "line": warning.line,
                "column": warning.column,
            }
            for warning in analysis_result.pylint_warnings
        ],
        "tool_errors": list(analysis_result.tool_errors),
    }
    return validate_agent_state(updated_state)


def run_core_orchestration_steps(
    state: AgentState,
    sandbox_executor: SandboxStepExecutor | None = None,
    test_runner: TestStepRunner | None = None,
    static_analyzer: StaticAnalysisExecutor | None = None,
    limits: SandboxLimits | None = None,
) -> AgentState:
    current_state = execute_sandbox_step(
        state=state,
        sandbox_executor=sandbox_executor,
        limits=limits,
    )
    if "test_cases" in current_state:
        current_state = run_predefined_tests_step(
            state=current_state,
            test_runner=test_runner,
            limits=limits,
        )
    current_state = run_python_static_analysis_step(
        state=current_state,
        static_analyzer=static_analyzer,
    )
    return validate_agent_state(current_state)


def run_full_graph_cycle(
    state: AgentState,
    task_generator: TaskGenerationExecutor | None = None,
    sandbox_executor: SandboxStepExecutor | None = None,
    test_runner: TestStepRunner | None = None,
    static_analyzer: StaticAnalysisExecutor | None = None,
    llm_reviewer: LLMReviewExecutor | None = None,
    profile_updater: ProfileUpdateExecutor | None = None,
    weights: ScoreAggregationWeights | None = None,
    limits: SandboxLimits | None = None,
) -> AgentState:
    current_state = generate_task_node(state=state, task_generator=task_generator)
    current_state = execute_sandbox_node(
        state=current_state,
        sandbox_executor=sandbox_executor,
        limits=limits,
    )
    current_state = run_tests_node(
        state=current_state,
        test_runner=test_runner,
        limits=limits,
    )
    current_state = static_analysis_node(
        state=current_state,
        static_analyzer=static_analyzer,
    )
    current_state = llm_review_node(
        state=current_state,
        llm_reviewer=llm_reviewer,
    )
    current_state = score_aggregation_node(
        state=current_state,
        weights=weights,
    )
    current_state = update_profile_node(
        state=current_state,
        profile_updater=profile_updater,
    )
    return validate_agent_state(current_state)


def _serialize_test_case_result(case_result: PredefinedTestCaseResult) -> dict[str, object]:
    return {
        "case_index": case_result.case_index,
        "passed": case_result.passed,
        "input_data": case_result.input_data,
        "expected_output": case_result.expected_output,
        "actual_output": case_result.actual_output,
        "stderr": case_result.stderr,
        "exit_code": case_result.exit_code,
        "timed_out": case_result.timed_out,
        "runtime_ms": case_result.runtime_ms,
        "memory_usage_kb": case_result.memory_usage_kb,
        "description": case_result.description,
        "output_diff": case_result.output_diff,
        "failure_reason": case_result.failure_reason,
    }


def _empty_test_results_state() -> dict[str, object]:
    return {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "first_failed_report": None,
        "case_results": [],
    }


def _is_blank_string(value: object) -> bool:
    if not isinstance(value, str):
        return True
    return value.strip() == ""


def _compute_test_score(state: AgentState) -> float:
    raw_test_results = state.get("test_results")
    if isinstance(raw_test_results, dict):
        total = raw_test_results.get("total")
        passed = raw_test_results.get("passed")
        if isinstance(total, int) and total > 0 and isinstance(passed, int):
            bounded_passed = min(max(passed, 0), total)
            return (bounded_passed / total) * 100.0

    execution_result = state.get("execution_result")
    if not isinstance(execution_result, dict):
        return 0.0

    if execution_result.get("timed_out") is True:
        return 0.0

    exit_code = execution_result.get("exit_code")
    if exit_code == 0:
        return 100.0
    return 20.0


def _compute_runtime_score(state: AgentState) -> float:
    metrics = state.get("metrics")
    if not isinstance(metrics, dict):
        return 50.0

    if metrics.get("timed_out") is True:
        return 0.0

    exit_code = metrics.get("exit_code")
    if isinstance(exit_code, int) and exit_code != 0:
        return 25.0

    runtime_ms = metrics.get("runtime_ms")
    if not isinstance(runtime_ms, int):
        return 70.0
    if runtime_ms <= 500:
        return 100.0
    if runtime_ms <= 1000:
        return 90.0
    if runtime_ms <= 2000:
        return 75.0
    if runtime_ms <= 3000:
        return 60.0
    return 40.0


def _compute_static_score(state: AgentState) -> float:
    static_analysis = state.get("static_analysis")
    if not isinstance(static_analysis, dict):
        return 70.0

    language = static_analysis.get("language")
    if language != "python":
        return 80.0

    pylint_score = static_analysis.get("pylint_score")
    if isinstance(pylint_score, int | float):
        pylint_component = max(0.0, min(100.0, float(pylint_score) * 10.0))
    else:
        pylint_component = 75.0

    complexity_score = static_analysis.get("complexity_score")
    if isinstance(complexity_score, int | float):
        complexity_component = max(0.0, 100.0 - (max(0.0, float(complexity_score) - 5.0) * 10.0))
    else:
        complexity_component = 80.0

    security_warnings = static_analysis.get("security_warnings")
    warnings_count = len(security_warnings) if isinstance(security_warnings, list) else 0
    security_component = max(0.0, 100.0 - (warnings_count * 20.0))

    tool_errors = static_analysis.get("tool_errors")
    tool_error_count = len(tool_errors) if isinstance(tool_errors, list) else 0
    tool_error_penalty = min(30.0, tool_error_count * 5.0)

    score = (
        (pylint_component * 0.5)
        + (complexity_component * 0.25)
        + (security_component * 0.25)
        - tool_error_penalty
    )
    return max(0.0, min(100.0, score))


def _compute_llm_review_score(state: AgentState) -> float:
    llm_review = state.get("llm_review")
    if not isinstance(llm_review, dict):
        return _compute_test_score(state)

    raw_score = llm_review.get("score")
    if isinstance(raw_score, int | float):
        return max(0.0, min(100.0, float(raw_score)))
    return _compute_test_score(state)


def _normalize_weights(weights: ScoreAggregationWeights) -> dict[str, float]:
    raw_weights = {
        "test_weight": float(weights.test_weight),
        "runtime_weight": float(weights.runtime_weight),
        "static_weight": float(weights.static_weight),
        "llm_weight": float(weights.llm_weight),
    }
    total_weight = sum(raw_weights.values())
    if total_weight <= 0:
        raise ValueError("ScoreAggregationWeights must have a positive total weight.")

    normalized_weights: dict[str, float] = {}
    for key, value in raw_weights.items():
        if value < 0:
            raise ValueError(f"ScoreAggregationWeights.{key} must be >= 0.")
        normalized_weights[key] = value / total_weight

    return normalized_weights


def _normalize_profile(profile: Mapping[str, object] | None) -> dict[str, object]:
    normalized: dict[str, object] = {
        "version": 1,
        "language_scores": {},
        "category_scores": {},
        "recent_scores": [],
    }
    if profile is None:
        return normalized

    version = profile.get("version")
    if isinstance(version, int) and version > 0:
        normalized["version"] = version
    normalized["language_scores"] = _normalize_score_bucket_map(profile.get("language_scores"))
    normalized["category_scores"] = _normalize_score_bucket_map(profile.get("category_scores"))
    normalized["recent_scores"] = _normalize_recent_scores(profile.get("recent_scores"))
    return normalized


def _normalize_score_bucket_map(value: object) -> dict[str, _ScoreBucket]:
    if not isinstance(value, dict):
        return {}

    normalized: dict[str, _ScoreBucket] = {}
    for raw_key, raw_bucket in value.items():
        key = str(raw_key).strip().lower()
        if key == "":
            continue
        normalized[key] = _normalize_score_bucket(raw_bucket)
    return normalized


def _normalize_score_bucket(value: object) -> _ScoreBucket:
    if not isinstance(value, dict):
        return {
            "attempts": 0,
            "average_score": 0.0,
            "best_score": 0.0,
            "last_score": 0.0,
        }

    attempts_raw = value.get("attempts")
    attempts = attempts_raw if isinstance(attempts_raw, int) and attempts_raw > 0 else 0

    average_raw = value.get("average_score")
    average_score = float(average_raw) if isinstance(average_raw, int | float) else 0.0

    best_raw = value.get("best_score")
    best_score = float(best_raw) if isinstance(best_raw, int | float) else 0.0

    last_raw = value.get("last_score")
    last_score = float(last_raw) if isinstance(last_raw, int | float) else 0.0

    return {
        "attempts": attempts,
        "average_score": max(0.0, min(100.0, average_score)),
        "best_score": max(0.0, min(100.0, best_score)),
        "last_score": max(0.0, min(100.0, last_score)),
    }


def _apply_new_score_to_bucket(
    current_bucket: Mapping[str, object] | None,
    score: float,
) -> _ScoreBucket:
    normalized_bucket = _normalize_score_bucket(current_bucket)
    attempts = normalized_bucket["attempts"]
    average_score = normalized_bucket["average_score"]
    best_score = normalized_bucket["best_score"]

    next_attempts = attempts + 1
    next_average = ((average_score * attempts) + score) / next_attempts
    next_best = max(best_score, score)

    return {
        "attempts": next_attempts,
        "average_score": round(next_average, 2),
        "best_score": round(next_best, 2),
        "last_score": round(score, 2),
    }


def _normalize_recent_scores(value: object) -> list[float]:
    if not isinstance(value, list):
        return []

    normalized_scores: list[float] = []
    for raw_score in value:
        if isinstance(raw_score, int | float):
            normalized_scores.append(round(max(0.0, min(100.0, float(raw_score))), 2))
    return normalized_scores


def _normalize_profile_version(value: object) -> int:
    if isinstance(value, int) and value > 0:
        return value

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            parsed = int(stripped)
            if parsed > 0:
                return parsed

    return 1
