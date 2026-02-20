from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict, cast

from interview_common import get_settings
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

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


class _ComplexityMistakesSummary(TypedDict):
    total: int
    by_category: dict[str, int]
    last_score: float | None


@dataclass(frozen=True)
class GeneratedTask:
    task_id: str
    prompt: str
    test_cases_json: str
    category: str = "general"
    difficulty: int = 1


@dataclass(frozen=True)
class ScoreAggregationWeights:
    correctness_weight: float = 0.55
    performance_weight: float = 0.2
    readability_weight: float = 0.15
    security_weight: float = 0.1


@dataclass(frozen=True)
class BranchingPolicy:
    max_retries: int = 1
    high_score_threshold: float = 85.0
    low_score_threshold: float = 45.0
    score_streak_length: int = 3
    max_hints: int = 3


_DEFAULT_SCORE_AGGREGATION_WEIGHTS = ScoreAggregationWeights()
_REVIEW_SCORE_WEIGHTS: dict[str, float] = {
    "correctness": _DEFAULT_SCORE_AGGREGATION_WEIGHTS.correctness_weight,
    "performance": _DEFAULT_SCORE_AGGREGATION_WEIGHTS.performance_weight,
    "readability": _DEFAULT_SCORE_AGGREGATION_WEIGHTS.readability_weight,
    "security": _DEFAULT_SCORE_AGGREGATION_WEIGHTS.security_weight,
}
_COMPLEXITY_MISTAKE_THRESHOLD = 8.0
_PIPELINE_LOGGER = logging.getLogger("interview_orchestrator.pipeline")


class TaskGenerationExecutor(Protocol):
    def generate_task(
        self,
        user_id: str,
        language: str,
        task_id: str,
    ) -> GeneratedTask: ...


class LLMReviewExecutor(Protocol):
    def review(self, state: AgentState) -> dict[str, object]: ...


class StructuredReviewChain(Protocol):
    def invoke(self, input: dict[str, object]) -> object: ...


class ProfileUpdateExecutor(Protocol):
    def update_profile(
        self,
        user_id: str,
        language: str,
        category: str,
        final_score: float,
        current_profile: Mapping[str, object] | None = None,
    ) -> dict[str, object]: ...


class _PromptModelStructuredReviewChain:
    def __init__(
        self,
        prompt_template: Any,
        chat_model: Any,
        parser: JsonOutputParser,
    ) -> None:
        self._prompt_template = prompt_template
        self._chat_model = chat_model
        self._parser = parser

    def invoke(self, input: dict[str, object]) -> object:
        prompt_text = self._prompt_template.format(**input)
        response = self._chat_model.invoke(prompt_text)
        response_text = _coerce_llm_content(response.content)
        parsed_review = self._parser.parse(response_text)
        token_usage = _extract_llm_token_usage(response)
        return {
            "review_payload": parsed_review,
            "token_usage": token_usage,
        }


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
        score_template = _build_review_score_template(state)
        test_score = score_template["correctness"]
        weighted_score = score_template["final_score"]

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
            "improvement_suggestions": _build_improvement_suggestions(
                state=state,
                max_suggestions=3,
            ),
            "score_template": score_template,
            "score": weighted_score,
            "reviewer": "heuristic",
        }


class LangChainCloudLLMReviewer:
    _PROMPT_TEMPLATE = (
        "You are a senior software engineer reviewing a coding interview submission.\n"
        "Return ONLY valid JSON and follow the format instructions exactly.\n"
        "{format_instructions}\n\n"
        "Use this review template:\n"
        "{review_template_json}\n\n"
        "Use this deterministic score template exactly. Keep all numeric values unchanged:\n"
        "{score_template_json}\n\n"
        "Submission context:\n"
        "- Language: {language}\n"
        "- Task: {task_prompt}\n"
        "- Source code:\n{code}\n\n"
        "- Execution result JSON: {execution_result_json}\n"
        "- Test results JSON: {test_results_json}\n"
        "- Static analysis JSON: {static_analysis_json}\n"
        "- Metrics JSON: {metrics_json}\n\n"
        "Rules:\n"
        "- Keep strengths/issues concise and concrete.\n"
        "- improvement_suggestions must be 1-3 actionable sentences.\n"
        "- score must match score_template.final_score.\n"
        "- score must be from 0 to 100."
    )

    def __init__(
        self,
        api_key: str | None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_retries: int = 2,
        timeout_seconds: float = 20.0,
        base_url: str | None = None,
        fallback_reviewer: LLMReviewExecutor | None = None,
        chain_factory: Callable[[], StructuredReviewChain | None] | None = None,
    ) -> None:
        normalized_api_key = api_key.strip() if isinstance(api_key, str) else ""
        self._api_key = normalized_api_key or None
        self._model = model
        self._temperature = max(0.0, float(temperature))
        self._max_attempts = max(1, max_retries + 1)
        self._timeout_seconds = max(1.0, float(timeout_seconds))
        normalized_base_url = base_url.strip() if isinstance(base_url, str) else ""
        self._base_url = normalized_base_url or None
        self._fallback_reviewer = fallback_reviewer or HeuristicLLMReviewer()
        self._chain_factory = chain_factory

    def review(self, state: AgentState) -> dict[str, object]:
        score_template = _build_review_score_template(state)
        default_score = score_template["final_score"]
        default_improvement_suggestions = _build_improvement_suggestions(
            state=state,
            max_suggestions=3,
        )
        fallback_review = _normalize_llm_review_payload(
            self._fallback_reviewer.review(state),
            default_reviewer="heuristic",
            default_score=default_score,
            default_score_template=score_template,
            default_improvement_suggestions=default_improvement_suggestions,
        )

        chain = self._build_chain()
        if chain is None:
            _log_pipeline_event(
                logging.INFO,
                event="submission.llm_review.fallback",
                state=state,
                reviewer="heuristic",
                reason="cloud_chain_unavailable",
            )
            return fallback_review

        prompt_input = _build_llm_review_prompt_input(state)
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                raw_response = chain.invoke(prompt_input)
                raw_review, token_usage = _extract_chain_review_and_token_usage(raw_response)
                _log_pipeline_event(
                    logging.INFO,
                    event="submission.llm_review.token_usage",
                    state=state,
                    reviewer="langchain-cloud",
                    attempt=attempt,
                    token_usage=token_usage,
                )
                return _normalize_llm_review_payload(
                    raw_review,
                    default_reviewer="langchain-cloud",
                    default_score=default_score,
                    default_score_template=score_template,
                    default_improvement_suggestions=default_improvement_suggestions,
                )
            except Exception as error:  # noqa: BLE001 - reviewer should degrade gracefully
                last_error = error
                _log_pipeline_event(
                    logging.WARNING,
                    event="submission.llm_review.attempt_failed",
                    state=state,
                    reviewer="langchain-cloud",
                    attempt=attempt,
                    error_type=type(error).__name__,
                    error=str(error),
                )

        failed_review = dict(fallback_review)
        issues = _normalize_non_empty_string_list(
            failed_review.get("issues"),
            fallback=["No critical issues detected."],
        )
        fallback_issue = "Cloud LLM review failed after retries. Used heuristic fallback."
        if fallback_issue not in issues:
            issues.append(fallback_issue)
        failed_review["issues"] = issues
        failed_review["reviewer"] = "heuristic_fallback"
        if last_error is not None:
            failed_review["llm_error"] = str(last_error)
        _log_pipeline_event(
            logging.WARNING,
            event="submission.llm_review.fallback",
            state=state,
            reviewer="heuristic_fallback",
            reason="cloud_retries_exhausted",
            attempts=self._max_attempts,
            last_error_type=(type(last_error).__name__ if last_error is not None else ""),
        )
        return failed_review

    def _build_chain(self) -> StructuredReviewChain | None:
        if self._chain_factory is not None:
            return self._chain_factory()

        if self._api_key is None:
            return None

        chat_model = self._build_chat_model()
        if chat_model is None:
            return None

        parser = JsonOutputParser()
        prompt = PromptTemplate.from_template(self._PROMPT_TEMPLATE).partial(
            format_instructions=parser.get_format_instructions()
        )
        llm_model = cast(Any, chat_model)
        return cast(
            StructuredReviewChain,
            _PromptModelStructuredReviewChain(
                prompt_template=prompt,
                chat_model=llm_model,
                parser=parser,
            ),
        )

    def _build_chat_model(self) -> Any | None:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            return None

        model_kwargs: dict[str, object] = {
            "api_key": self._api_key,
            "model": self._model,
            "temperature": self._temperature,
            "max_retries": 0,
            "timeout": self._timeout_seconds,
        }
        if self._base_url is not None:
            model_kwargs["base_url"] = self._base_url

        return ChatOpenAI(**model_kwargs)


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
    _log_pipeline_event(
        logging.INFO,
        event="submission.generate_task.started",
        state=validated_state,
    )
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

    recommended_difficulty = updated_state.get("recommended_difficulty")
    if isinstance(recommended_difficulty, int) and recommended_difficulty >= 1:
        updated_state["task_difficulty"] = recommended_difficulty
    else:
        difficulty = updated_state.get("task_difficulty")
        if not isinstance(difficulty, int) or difficulty < 1:
            updated_state["task_difficulty"] = generated_task.difficulty

    _log_pipeline_event(
        logging.INFO,
        event="submission.generate_task.completed",
        state=updated_state,
        task_category=updated_state.get("task_category"),
        task_difficulty=updated_state.get("task_difficulty"),
        has_test_cases=isinstance(updated_state.get("test_cases"), str),
    )
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
        _log_pipeline_event(
            logging.INFO,
            event="submission.tests.skipped",
            state=updated_state,
            reason="missing_test_cases",
        )
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
    _log_pipeline_event(
        logging.INFO,
        event="submission.llm_review.started",
        state=validated_state,
    )
    reviewer = llm_reviewer or _build_default_llm_reviewer()
    raw_review = reviewer.review(validated_state)
    if not isinstance(raw_review, dict):
        raise ValueError("llm_reviewer must return a dictionary payload.")

    score_template = _build_review_score_template(validated_state)
    review = _normalize_llm_review_payload(
        raw_review,
        default_reviewer="llm_reviewer",
        default_score=score_template["final_score"],
        default_score_template=score_template,
        default_improvement_suggestions=_build_improvement_suggestions(
            state=validated_state,
            max_suggestions=3,
        ),
    )
    updated_state: dict[str, object] = dict(validated_state)
    updated_state["llm_review"] = review
    _log_pipeline_event(
        logging.INFO,
        event="submission.llm_review.completed",
        state=updated_state,
        reviewer=review.get("reviewer"),
        score=review.get("score"),
    )
    return validate_agent_state(updated_state)


def score_aggregation_node(
    state: AgentState,
    weights: ScoreAggregationWeights | None = None,
) -> AgentState:
    validated_state = validate_agent_state(state)
    normalized_weights = _normalize_weights(weights or _DEFAULT_SCORE_AGGREGATION_WEIGHTS)

    score_template = _build_review_score_template(validated_state)
    correctness_score = score_template["correctness"]
    performance_score = score_template["performance"]
    readability_score = score_template["readability"]
    security_score = score_template["security"]

    final_score = _clamp_review_score(
        (correctness_score * normalized_weights["correctness_weight"])
        + (performance_score * normalized_weights["performance_weight"])
        + (readability_score * normalized_weights["readability_weight"])
        + (security_score * normalized_weights["security_weight"])
    )

    updated_state: dict[str, object] = dict(validated_state)
    updated_state["score_breakdown"] = {
        "correctness_score": round(correctness_score, 2),
        "performance_score": round(performance_score, 2),
        "readability_score": round(readability_score, 2),
        "security_score": round(security_score, 2),
        "weights": normalized_weights,
        "aggregated_score": final_score,
    }
    updated_state["final_score"] = final_score
    _log_pipeline_event(
        logging.INFO,
        event="submission.score_aggregation.completed",
        state=updated_state,
        final_score=final_score,
        correctness_score=round(correctness_score, 2),
        performance_score=round(performance_score, 2),
        readability_score=round(readability_score, 2),
        security_score=round(security_score, 2),
    )
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
    has_failure = _has_submission_failure(current_state)
    complexity_score = _extract_complexity_score(current_state)
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
    tracked_profile = _apply_weakness_tracking(
        profile=updated_profile,
        category=category,
        has_failure=has_failure,
        complexity_score=complexity_score,
    )

    updated_state: dict[str, object] = dict(current_state)
    updated_state["skill_profile"] = tracked_profile
    _log_pipeline_event(
        logging.INFO,
        event="submission.profile_update.completed",
        state=updated_state,
        category=category,
        final_score=round(final_score, 2),
        recent_scores_count=len(_normalize_recent_scores(tracked_profile.get("recent_scores"))),
    )
    return validate_agent_state(updated_state)


def branching_logic_node(
    state: AgentState,
    policy: BranchingPolicy | None = None,
) -> AgentState:
    validated_state = validate_agent_state(state)
    normalized_policy = _normalize_branching_policy(policy or BranchingPolicy())
    has_failure = _has_submission_failure(validated_state)

    retry_count_raw = validated_state.get("retry_count")
    retry_count = (
        retry_count_raw if isinstance(retry_count_raw, int) and retry_count_raw >= 0 else 0
    )
    should_retry = has_failure and retry_count < normalized_policy.max_retries
    retries_used = retry_count + 1 if should_retry else retry_count
    if not has_failure:
        retries_used = 0
    retries_remaining = max(0, normalized_policy.max_retries - retries_used)

    if not has_failure:
        retry_reason = "No failure detected."
    elif should_retry:
        retry_reason = "Submission failed and retry budget is available."
    else:
        retry_reason = "Submission failed and retry budget is exhausted."

    current_difficulty_raw = validated_state.get("task_difficulty")
    current_difficulty = (
        current_difficulty_raw
        if isinstance(current_difficulty_raw, int) and current_difficulty_raw >= 1
        else 1
    )
    recent_scores = _extract_recent_scores(validated_state)
    has_high_score_streak = _has_score_streak(
        recent_scores=recent_scores,
        streak_length=normalized_policy.score_streak_length,
        min_score=normalized_policy.high_score_threshold,
    )
    has_low_score_streak = _has_score_streak(
        recent_scores=recent_scores,
        streak_length=normalized_policy.score_streak_length,
        max_score=normalized_policy.low_score_threshold,
    )
    has_repeated_failures = has_failure and retry_count >= normalized_policy.max_retries

    if has_high_score_streak:
        adaptive_action = "increase"
        next_difficulty = current_difficulty + 1
        adaptive_reason = (
            f"Last {normalized_policy.score_streak_length} scores are above "
            f"{normalized_policy.high_score_threshold:.1f}."
        )
    elif has_low_score_streak or has_repeated_failures:
        adaptive_action = "decrease"
        next_difficulty = max(1, current_difficulty - 1)
        adaptive_reason = (
            "Repeated failures detected."
            if has_repeated_failures
            else (
                f"Last {normalized_policy.score_streak_length} scores are below "
                f"{normalized_policy.low_score_threshold:.1f}."
            )
        )
    else:
        adaptive_action = "keep"
        next_difficulty = current_difficulty
        adaptive_reason = "Difficulty remains unchanged."

    hints = (
        _build_hints(
            state=validated_state,
            max_hints=normalized_policy.max_hints,
        )
        if has_failure
        else []
    )
    should_show_hint = len(hints) > 0

    next_node = "complete"
    if should_retry and should_show_hint:
        next_node = "retry_with_hint"
    elif should_retry:
        next_node = "retry_submission"
    elif should_show_hint:
        next_node = "show_hint"

    updated_state: dict[str, object] = dict(validated_state)
    updated_state["retry_count"] = retries_used
    updated_state["recommended_difficulty"] = next_difficulty
    updated_state["branching"] = {
        "retry": {
            "should_retry": should_retry,
            "retries_used": retries_used,
            "retries_remaining": retries_remaining,
            "max_retries": normalized_policy.max_retries,
            "reason": retry_reason,
        },
        "adaptive_difficulty": {
            "action": adaptive_action,
            "current_difficulty": current_difficulty,
            "next_difficulty": next_difficulty,
            "reason": adaptive_reason,
        },
        "hint": {
            "should_show_hint": should_show_hint,
            "hints": hints,
        },
        "next_node": next_node,
    }
    _log_pipeline_event(
        logging.INFO,
        event="submission.branching.completed",
        state=updated_state,
        should_retry=should_retry,
        retries_used=retries_used,
        recommended_difficulty=next_difficulty,
        adaptive_action=adaptive_action,
        next_node=next_node,
    )
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

    _log_pipeline_event(
        logging.INFO,
        event="submission.sandbox.started",
        state=validated_state,
        main_class_name=main_class_name,
    )
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
    _log_pipeline_event(
        logging.INFO,
        event="submission.sandbox.completed",
        state=updated_state,
        exit_code=execution.exit_code,
        timed_out=execution.timed_out,
        runtime_ms=runtime_ms,
        memory_usage_kb=execution.memory_usage_kb,
        stderr_excerpt=_truncate_for_log(execution.stderr, max_length=200),
    )
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
    _log_pipeline_event(
        logging.INFO,
        event="submission.tests.completed",
        state=updated_state,
        total=test_results.total,
        passed=test_results.passed,
        failed=test_results.failed,
    )
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
        _log_pipeline_event(
            logging.INFO,
            event="submission.static_analysis.skipped",
            state=updated_state,
            reason="language_not_supported",
        )
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
    _log_pipeline_event(
        logging.INFO,
        event="submission.static_analysis.completed",
        state=updated_state,
        pylint_score=analysis_result.pylint_score,
        complexity_score=analysis_result.complexity_score,
        security_warnings=len(analysis_result.security_warnings),
        pylint_warnings=len(analysis_result.pylint_warnings),
        tool_errors=len(analysis_result.tool_errors),
    )
    return validate_agent_state(updated_state)


def run_core_orchestration_steps(
    state: AgentState,
    sandbox_executor: SandboxStepExecutor | None = None,
    test_runner: TestStepRunner | None = None,
    static_analyzer: StaticAnalysisExecutor | None = None,
    limits: SandboxLimits | None = None,
) -> AgentState:
    validated_state = validate_agent_state(state)
    _log_pipeline_event(
        logging.INFO,
        event="submission.core_pipeline.started",
        state=validated_state,
    )

    try:
        current_state = execute_sandbox_step(
            state=validated_state,
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
        _log_pipeline_event(
            logging.INFO,
            event="submission.core_pipeline.completed",
            state=current_state,
        )
        return validate_agent_state(current_state)
    except Exception as error:
        _log_pipeline_event(
            logging.ERROR,
            event="submission.core_pipeline.failed",
            state=validated_state,
            error_type=type(error).__name__,
            error=str(error),
            exc_info=True,
        )
        raise


def run_full_graph_cycle(
    state: AgentState,
    task_generator: TaskGenerationExecutor | None = None,
    sandbox_executor: SandboxStepExecutor | None = None,
    test_runner: TestStepRunner | None = None,
    static_analyzer: StaticAnalysisExecutor | None = None,
    llm_reviewer: LLMReviewExecutor | None = None,
    profile_updater: ProfileUpdateExecutor | None = None,
    weights: ScoreAggregationWeights | None = None,
    branching_policy: BranchingPolicy | None = None,
    limits: SandboxLimits | None = None,
) -> AgentState:
    validated_state = validate_agent_state(state)
    _log_pipeline_event(
        logging.INFO,
        event="submission.pipeline.started",
        state=validated_state,
        graph_order=(
            "GenerateTask->ExecuteSandbox->RunTests->StaticAnalysis->"
            "LLMReview->ScoreAggregation->UpdateProfile->BranchingLogic"
        ),
    )

    try:
        current_state = generate_task_node(state=validated_state, task_generator=task_generator)
        _log_pipeline_event(
            logging.INFO,
            event="submission.pipeline.step_completed",
            state=current_state,
            step="GenerateTask",
        )
        current_state = execute_sandbox_node(
            state=current_state,
            sandbox_executor=sandbox_executor,
            limits=limits,
        )
        _log_pipeline_event(
            logging.INFO,
            event="submission.pipeline.step_completed",
            state=current_state,
            step="ExecuteSandbox",
        )
        current_state = run_tests_node(
            state=current_state,
            test_runner=test_runner,
            limits=limits,
        )
        _log_pipeline_event(
            logging.INFO,
            event="submission.pipeline.step_completed",
            state=current_state,
            step="RunTests",
        )
        current_state = static_analysis_node(
            state=current_state,
            static_analyzer=static_analyzer,
        )
        _log_pipeline_event(
            logging.INFO,
            event="submission.pipeline.step_completed",
            state=current_state,
            step="StaticAnalysis",
        )
        current_state = llm_review_node(
            state=current_state,
            llm_reviewer=llm_reviewer,
        )
        _log_pipeline_event(
            logging.INFO,
            event="submission.pipeline.step_completed",
            state=current_state,
            step="LLMReview",
        )
        current_state = score_aggregation_node(
            state=current_state,
            weights=weights,
        )
        _log_pipeline_event(
            logging.INFO,
            event="submission.pipeline.step_completed",
            state=current_state,
            step="ScoreAggregation",
        )
        current_state = update_profile_node(
            state=current_state,
            profile_updater=profile_updater,
        )
        _log_pipeline_event(
            logging.INFO,
            event="submission.pipeline.step_completed",
            state=current_state,
            step="UpdateProfile",
        )
        current_state = branching_logic_node(
            state=current_state,
            policy=branching_policy,
        )
        _log_pipeline_event(
            logging.INFO,
            event="submission.pipeline.step_completed",
            state=current_state,
            step="BranchingLogic",
        )
        _log_pipeline_event(
            logging.INFO,
            event="submission.pipeline.completed",
            state=current_state,
            final_score=current_state.get("final_score"),
            recommended_difficulty=current_state.get("recommended_difficulty"),
            next_node=cast(dict[str, object], current_state.get("branching", {})).get("next_node"),
        )
        return validate_agent_state(current_state)
    except Exception as error:
        _log_pipeline_event(
            logging.ERROR,
            event="submission.pipeline.failed",
            state=validated_state,
            error_type=type(error).__name__,
            error=str(error),
            exc_info=True,
        )
        raise


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


def _build_default_llm_reviewer() -> LLMReviewExecutor:
    settings = get_settings()
    return LangChainCloudLLMReviewer(
        api_key=settings.llm_api_key,
        fallback_reviewer=HeuristicLLMReviewer(),
    )


def _log_pipeline_event(
    level: int,
    *,
    event: str,
    state: Mapping[str, object],
    exc_info: bool = False,
    **fields: object,
) -> None:
    payload: dict[str, object] = {
        "event": event,
        "user_id": _extract_state_string(state, "user_id"),
        "task_id": _extract_state_string(state, "task_id"),
        "language": _extract_state_string(state, "language"),
    }
    payload.update(fields)
    _PIPELINE_LOGGER.log(
        level,
        json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str),
        exc_info=exc_info,
    )


def _extract_state_string(state: Mapping[str, object], key: str) -> str:
    value = state.get(key)
    if isinstance(value, str):
        return value
    return ""


def _truncate_for_log(value: object, max_length: int = 200) -> str:
    if max_length < 4:
        return ""
    if not isinstance(value, str):
        return ""

    normalized = value.replace("\n", "\\n")
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[: max_length - 3]}..."


def _build_llm_review_prompt_input(state: AgentState) -> dict[str, object]:
    task_prompt = state.get("task_prompt")
    normalized_task_prompt = (
        task_prompt
        if isinstance(task_prompt, str) and task_prompt.strip() != ""
        else "No task prompt provided."
    )
    score_template = _build_review_score_template(state)
    review_template = {
        "summary": "One concise paragraph with the main verdict.",
        "strengths": [
            "Concrete strength #1",
            "Concrete strength #2",
        ],
        "issues": [
            "Most important issue #1",
            "Most important issue #2",
        ],
        "improvement_suggestions": [
            "Actionable improvement #1",
            "Actionable improvement #2",
        ],
        "score_template": score_template,
        "score": score_template["final_score"],
        "reviewer": "langchain-cloud",
    }

    return {
        "language": state["language"],
        "task_prompt": normalized_task_prompt,
        "code": state["code"],
        "review_template_json": _json_dump_payload(review_template),
        "score_template_json": _json_dump_payload(score_template),
        "execution_result_json": _json_dump_payload(state.get("execution_result")),
        "test_results_json": _json_dump_payload(state.get("test_results")),
        "static_analysis_json": _json_dump_payload(state.get("static_analysis")),
        "metrics_json": _json_dump_payload(state.get("metrics")),
    }


def _extract_chain_review_and_token_usage(
    raw_response: object,
) -> tuple[object, dict[str, int] | None]:
    if not isinstance(raw_response, dict):
        return raw_response, None

    if "review_payload" in raw_response:
        return raw_response.get("review_payload"), _normalize_token_usage(
            raw_response.get("token_usage")
        )

    token_usage = _normalize_token_usage(raw_response.get("token_usage"))
    if token_usage is None:
        token_usage = _normalize_token_usage(raw_response.get("usage"))

    review_payload = dict(raw_response)
    review_payload.pop("token_usage", None)
    review_payload.pop("usage", None)
    return review_payload, token_usage


def _coerce_llm_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    chunks.append(text_value)
        if len(chunks) > 0:
            return "".join(chunks)
    return str(content)


def _extract_llm_token_usage(response: object) -> dict[str, int] | None:
    usage_metadata = getattr(response, "usage_metadata", None)
    normalized_usage = _normalize_token_usage(usage_metadata)
    if normalized_usage is not None:
        return normalized_usage

    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, Mapping):
        token_usage = response_metadata.get("token_usage")
        return _normalize_token_usage(token_usage)

    return None


def _normalize_token_usage(value: object) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None

    prompt_tokens = _coerce_non_negative_int(
        value.get("prompt_tokens") if "prompt_tokens" in value else value.get("input_tokens")
    )
    completion_tokens = _coerce_non_negative_int(
        value.get("completion_tokens")
        if "completion_tokens" in value
        else value.get("output_tokens")
    )
    total_tokens = _coerce_non_negative_int(value.get("total_tokens"))
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None

    return {
        "prompt_tokens": prompt_tokens or 0,
        "completion_tokens": completion_tokens or 0,
        "total_tokens": total_tokens or 0,
    }


def _coerce_non_negative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if value >= 0 and value.is_integer():
            return int(value)
        return None
    return None


def _normalize_llm_review_payload(
    payload: object,
    *,
    default_reviewer: str,
    default_score: float,
    default_score_template: Mapping[str, float],
    default_improvement_suggestions: list[str],
) -> dict[str, object]:
    payload_map = payload if isinstance(payload, dict) else {}

    summary_raw = payload_map.get("summary")
    summary = (
        summary_raw.strip()
        if isinstance(summary_raw, str) and summary_raw.strip() != ""
        else "Review generated from available pipeline signals."
    )

    strengths = _normalize_non_empty_string_list(
        payload_map.get("strengths"),
        fallback=["Submission structure is valid."],
    )
    issues = _normalize_non_empty_string_list(
        payload_map.get("issues"),
        fallback=["No critical issues detected."],
    )
    improvement_suggestions = _normalize_non_empty_string_list(
        payload_map.get("improvement_suggestions"),
        fallback=default_improvement_suggestions,
    )
    score_template = _normalize_score_template(
        payload_map.get("score_template"),
        default=default_score_template,
    )

    reviewer_raw = payload_map.get("reviewer")
    reviewer = (
        reviewer_raw.strip()
        if isinstance(reviewer_raw, str) and reviewer_raw.strip() != ""
        else default_reviewer
    )

    score = _normalize_review_score(payload_map.get("score"), default=default_score)
    if isinstance(payload_map.get("score_template"), dict):
        score = score_template["final_score"]

    return {
        "summary": summary,
        "strengths": strengths,
        "issues": issues,
        "improvement_suggestions": improvement_suggestions,
        "score_template": score_template,
        "score": score,
        "reviewer": reviewer,
    }


def _normalize_non_empty_string_list(value: object, fallback: list[str]) -> list[str]:
    if not isinstance(value, list):
        return list(fallback)

    normalized: list[str] = []
    for item in value:
        if isinstance(item, str):
            stripped_item = item.strip()
            if stripped_item != "":
                normalized.append(stripped_item)

    if len(normalized) == 0:
        return list(fallback)
    return normalized


def _normalize_review_score(value: object, default: float) -> float:
    if isinstance(value, int | float):
        return _clamp_review_score(float(value))
    return _clamp_review_score(float(default))


def _normalize_score_template(
    value: object,
    default: Mapping[str, float],
) -> dict[str, float]:
    normalized_default = {
        "correctness": _normalize_review_score(default.get("correctness"), default=0.0),
        "performance": _normalize_review_score(default.get("performance"), default=0.0),
        "readability": _normalize_review_score(default.get("readability"), default=0.0),
        "security": _normalize_review_score(default.get("security"), default=0.0),
        "final_score": _normalize_review_score(default.get("final_score"), default=0.0),
    }
    if not isinstance(value, dict):
        return normalized_default

    correctness = _normalize_review_score(
        value.get("correctness"),
        default=normalized_default["correctness"],
    )
    performance = _normalize_review_score(
        value.get("performance"),
        default=normalized_default["performance"],
    )
    readability = _normalize_review_score(
        value.get("readability"),
        default=normalized_default["readability"],
    )
    security = _normalize_review_score(
        value.get("security"),
        default=normalized_default["security"],
    )

    final_score_raw = value.get("final_score")
    if isinstance(final_score_raw, int | float):
        final_score = _normalize_review_score(
            final_score_raw,
            default=normalized_default["final_score"],
        )
    else:
        final_score = _compute_review_final_score(
            correctness=correctness,
            performance=performance,
            readability=readability,
            security=security,
        )

    return {
        "correctness": correctness,
        "performance": performance,
        "readability": readability,
        "security": security,
        "final_score": final_score,
    }


def _clamp_review_score(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 2)


def _compute_review_final_score(
    *,
    correctness: float,
    performance: float,
    readability: float,
    security: float,
) -> float:
    raw_final_score = (
        (correctness * _REVIEW_SCORE_WEIGHTS["correctness"])
        + (performance * _REVIEW_SCORE_WEIGHTS["performance"])
        + (readability * _REVIEW_SCORE_WEIGHTS["readability"])
        + (security * _REVIEW_SCORE_WEIGHTS["security"])
    )
    return _clamp_review_score(raw_final_score)


def _build_review_score_template(state: AgentState) -> dict[str, float]:
    correctness = _normalize_review_score(_compute_test_score(state), default=0.0)
    performance = _normalize_review_score(_compute_runtime_score(state), default=0.0)
    readability = _normalize_review_score(_compute_readability_score(state), default=0.0)
    security = _normalize_review_score(_compute_security_score(state), default=0.0)
    final_score = _compute_review_final_score(
        correctness=correctness,
        performance=performance,
        readability=readability,
        security=security,
    )

    return {
        "correctness": correctness,
        "performance": performance,
        "readability": readability,
        "security": security,
        "final_score": final_score,
    }


def _json_dump_payload(payload: object) -> str:
    if payload is None:
        return "{}"
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)


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


def _compute_readability_score(state: AgentState) -> float:
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

    tool_errors = static_analysis.get("tool_errors")
    tool_error_count = len(tool_errors) if isinstance(tool_errors, list) else 0
    readability_penalty = min(20.0, tool_error_count * 4.0)

    score = (pylint_component * 0.65) + (complexity_component * 0.35) - readability_penalty
    return max(0.0, min(100.0, score))


def _compute_security_score(state: AgentState) -> float:
    static_analysis = state.get("static_analysis")
    if not isinstance(static_analysis, dict):
        return 80.0

    language = static_analysis.get("language")
    if language != "python":
        return 85.0

    security_warnings = static_analysis.get("security_warnings")
    warnings_count = len(security_warnings) if isinstance(security_warnings, list) else 0

    tool_errors = static_analysis.get("tool_errors")
    tool_error_count = len(tool_errors) if isinstance(tool_errors, list) else 0

    score = 100.0 - (warnings_count * 20.0) - (tool_error_count * 5.0)
    return max(0.0, min(100.0, score))


def _compute_llm_review_score(state: AgentState) -> float:
    llm_review = state.get("llm_review")
    if not isinstance(llm_review, dict):
        return _compute_test_score(state)

    raw_score = llm_review.get("score")
    if isinstance(raw_score, int | float):
        return max(0.0, min(100.0, float(raw_score)))
    return _compute_test_score(state)


def _normalize_branching_policy(policy: BranchingPolicy) -> BranchingPolicy:
    if policy.max_retries < 0:
        raise ValueError("BranchingPolicy.max_retries must be >= 0.")
    if policy.score_streak_length < 1:
        raise ValueError("BranchingPolicy.score_streak_length must be >= 1.")
    if policy.max_hints < 0:
        raise ValueError("BranchingPolicy.max_hints must be >= 0.")
    if not 0.0 <= policy.high_score_threshold <= 100.0:
        raise ValueError("BranchingPolicy.high_score_threshold must be in range [0, 100].")
    if not 0.0 <= policy.low_score_threshold <= 100.0:
        raise ValueError("BranchingPolicy.low_score_threshold must be in range [0, 100].")
    return policy


def _has_submission_failure(state: AgentState) -> bool:
    test_results = state.get("test_results")
    if isinstance(test_results, dict):
        failed = test_results.get("failed")
        if isinstance(failed, int) and failed > 0:
            return True

    execution_result = state.get("execution_result")
    if isinstance(execution_result, dict):
        if execution_result.get("timed_out") is True:
            return True
        exit_code = execution_result.get("exit_code")
        if isinstance(exit_code, int) and exit_code != 0:
            return True

    return False


def _extract_recent_scores(state: AgentState) -> list[float]:
    profile = state.get("skill_profile")
    if not isinstance(profile, dict):
        return []
    return _normalize_recent_scores(profile.get("recent_scores"))


def _extract_complexity_score(state: AgentState) -> float | None:
    static_analysis = state.get("static_analysis")
    if not isinstance(static_analysis, dict):
        return None

    raw_complexity_score = static_analysis.get("complexity_score")
    if isinstance(raw_complexity_score, int | float):
        return max(0.0, float(raw_complexity_score))
    return None


def _has_score_streak(
    recent_scores: list[float],
    streak_length: int,
    min_score: float | None = None,
    max_score: float | None = None,
) -> bool:
    if len(recent_scores) < streak_length:
        return False

    streak = recent_scores[-streak_length:]
    for score in streak:
        if min_score is not None and score < min_score:
            return False
        if max_score is not None and score > max_score:
            return False
    return True


def _build_improvement_suggestions(
    state: AgentState,
    max_suggestions: int,
) -> list[str]:
    if max_suggestions <= 0:
        return []

    suggestions: list[str] = []

    test_results = state.get("test_results")
    if isinstance(test_results, dict):
        failed = test_results.get("failed")
        if isinstance(failed, int) and failed > 0:
            _append_hint(
                suggestions,
                "Fix failing test cases first, starting from the first failed report.",
                max_hints=max_suggestions,
            )

    execution_result = state.get("execution_result")
    if isinstance(execution_result, dict):
        if execution_result.get("timed_out") is True:
            _append_hint(
                suggestions,
                "Reduce algorithm complexity or remove unbounded loops to avoid timeouts.",
                max_hints=max_suggestions,
            )
        elif execution_result.get("exit_code") not in (None, 0):
            _append_hint(
                suggestions,
                "Handle runtime errors and invalid inputs to ensure a zero exit code.",
                max_hints=max_suggestions,
            )

    static_analysis = state.get("static_analysis")
    if isinstance(static_analysis, dict):
        security_warnings = static_analysis.get("security_warnings")
        if isinstance(security_warnings, list) and len(security_warnings) > 0:
            _append_hint(
                suggestions,
                "Address security warnings from static analysis before the next submission.",
                max_hints=max_suggestions,
            )

        complexity_score = static_analysis.get("complexity_score")
        if isinstance(complexity_score, int | float) and float(complexity_score) > 8.0:
            _append_hint(
                suggestions,
                "Refactor complex logic into smaller functions to improve readability.",
                max_hints=max_suggestions,
            )

    if len(suggestions) == 0:
        _append_hint(
            suggestions,
            "Keep the current approach and add edge-case tests to preserve stability.",
            max_hints=max_suggestions,
        )

    return suggestions


def _build_hints(
    state: AgentState,
    max_hints: int,
) -> list[str]:
    if max_hints <= 0:
        return []

    hints: list[str] = []

    test_results = state.get("test_results")
    if isinstance(test_results, dict):
        failed_report = test_results.get("first_failed_report")
        if isinstance(failed_report, str) and failed_report.strip() != "":
            _append_hint(
                hints,
                f"Review the first failed test: {failed_report.strip()}",
                max_hints=max_hints,
            )

    execution_result = state.get("execution_result")
    if isinstance(execution_result, dict):
        stderr = execution_result.get("stderr")
        if isinstance(stderr, str):
            stderr_hint = _first_non_blank_line(stderr)
            if stderr_hint is not None:
                _append_hint(hints, f"Execution stderr: {stderr_hint}", max_hints=max_hints)

    static_analysis = state.get("static_analysis")
    if isinstance(static_analysis, dict):
        warnings_summary = static_analysis.get("security_warnings_summary")
        if isinstance(warnings_summary, str):
            normalized_summary = warnings_summary.strip()
            if (
                normalized_summary != ""
                and not normalized_summary.startswith("No security warnings")
                and not normalized_summary.startswith("Skipped:")
            ):
                _append_hint(
                    hints,
                    f"Static analysis warning: {normalized_summary}",
                    max_hints=max_hints,
                )

    llm_review = state.get("llm_review")
    if isinstance(llm_review, dict):
        issues = llm_review.get("issues")
        if isinstance(issues, list):
            for issue in issues:
                if isinstance(issue, str) and issue.strip() != "":
                    _append_hint(hints, f"Review note: {issue.strip()}", max_hints=max_hints)
                    if len(hints) >= max_hints:
                        break

    return hints


def _append_hint(hints: list[str], candidate_hint: str, max_hints: int) -> None:
    normalized_hint = candidate_hint.strip()
    if normalized_hint == "" or len(hints) >= max_hints:
        return
    if normalized_hint in hints:
        return
    hints.append(normalized_hint)


def _first_non_blank_line(text: str) -> str | None:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line != "":
            return line
    return None


def _normalize_weights(weights: ScoreAggregationWeights) -> dict[str, float]:
    raw_weights = {
        "correctness_weight": float(weights.correctness_weight),
        "performance_weight": float(weights.performance_weight),
        "readability_weight": float(weights.readability_weight),
        "security_weight": float(weights.security_weight),
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


def _apply_weakness_tracking(
    profile: Mapping[str, object] | None,
    *,
    category: str,
    has_failure: bool,
    complexity_score: float | None,
) -> dict[str, object]:
    normalized_profile = dict(profile) if isinstance(profile, Mapping) else {}
    normalized_category = category.strip().lower() or "general"

    failed_categories = _normalize_counter_map(normalized_profile.get("failed_categories"))
    if has_failure:
        failed_categories[normalized_category] = failed_categories.get(normalized_category, 0) + 1

    complexity_mistakes = _normalize_complexity_mistakes(
        normalized_profile.get("complexity_mistakes")
    )
    if isinstance(complexity_score, int | float):
        normalized_complexity_score = round(max(0.0, float(complexity_score)), 2)
        complexity_mistakes["last_score"] = normalized_complexity_score
        if normalized_complexity_score > _COMPLEXITY_MISTAKE_THRESHOLD:
            complexity_mistakes["total"] = complexity_mistakes["total"] + 1
            by_category = complexity_mistakes["by_category"]
            by_category[normalized_category] = by_category.get(normalized_category, 0) + 1

    normalized_profile["failed_categories"] = failed_categories
    normalized_profile["complexity_mistakes"] = complexity_mistakes
    return normalized_profile


def _normalize_profile(profile: Mapping[str, object] | None) -> dict[str, object]:
    normalized: dict[str, object] = {
        "version": 1,
        "language_scores": {},
        "category_scores": {},
        "recent_scores": [],
        "failed_categories": {},
        "complexity_mistakes": {
            "total": 0,
            "by_category": {},
            "last_score": None,
        },
    }
    if profile is None:
        return normalized

    version = profile.get("version")
    if isinstance(version, int) and version > 0:
        normalized["version"] = version
    normalized["language_scores"] = _normalize_score_bucket_map(profile.get("language_scores"))
    normalized["category_scores"] = _normalize_score_bucket_map(profile.get("category_scores"))
    normalized["recent_scores"] = _normalize_recent_scores(profile.get("recent_scores"))
    normalized["failed_categories"] = _normalize_counter_map(profile.get("failed_categories"))
    normalized["complexity_mistakes"] = _normalize_complexity_mistakes(
        profile.get("complexity_mistakes")
    )
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


def _normalize_counter_map(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}

    normalized: dict[str, int] = {}
    for raw_key, raw_count in value.items():
        key = str(raw_key).strip().lower()
        if key == "":
            continue
        if isinstance(raw_count, int) and raw_count > 0:
            normalized[key] = raw_count
    return normalized


def _normalize_complexity_mistakes(value: object) -> _ComplexityMistakesSummary:
    normalized: _ComplexityMistakesSummary = {
        "total": 0,
        "by_category": {},
        "last_score": None,
    }
    if not isinstance(value, dict):
        return normalized

    total = value.get("total")
    if isinstance(total, int) and total > 0:
        normalized["total"] = total

    normalized["by_category"] = _normalize_counter_map(value.get("by_category"))

    last_score = value.get("last_score")
    if isinstance(last_score, int | float):
        normalized["last_score"] = round(max(0.0, float(last_score)), 2)

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
