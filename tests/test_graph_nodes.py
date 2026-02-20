from __future__ import annotations

import json
import unittest
from typing import cast

from interview_orchestrator.agent_state import AgentState
from interview_orchestrator.sandbox_runner import SandboxExecutionResult
from interview_orchestrator.state_steps import (
    BranchingPolicy,
    GeneratedTask,
    InMemoryLLMRateLimiter,
    LangChainCloudLLMReviewer,
    ScoreAggregationWeights,
    branching_logic_node,
    generate_task_node,
    llm_review_node,
    run_full_graph_cycle,
    run_tests_node,
    score_aggregation_node,
    update_profile_node,
)
from interview_orchestrator.static_analysis import (
    BanditWarning,
    PylintWarning,
    PythonStaticAnalysisResult,
    RadonComplexityBlock,
)
from interview_orchestrator.test_runner import (
    PredefinedTestCaseResult,
    PredefinedTestRunResult,
)


def _parse_json_log_record(record: str) -> dict[str, object]:
    parts = record.split(":", maxsplit=2)
    if len(parts) != 3:
        raise AssertionError(f"Unexpected log record format: {record!r}")
    return cast(dict[str, object], json.loads(parts[2]))


class StubTaskGenerator:
    def generate_task(self, user_id: str, language: str, task_id: str) -> GeneratedTask:
        return GeneratedTask(
            task_id=f"{task_id}:{user_id}:{language}",
            prompt="Add two numbers from stdin.",
            test_cases_json='[{"input":"1 2\\n","output":"3\\n"}]',
            category="math",
            difficulty=2,
        )


class StubSandboxExecutor:
    def execute(
        self,
        language: str,
        source_code: str,
        limits: object | None = None,
        main_class_name: str = "Main",
        stdin_data: str = "",
    ) -> SandboxExecutionResult:
        return SandboxExecutionResult(
            language=language,
            exit_code=0,
            stdout="3\n",
            stderr="",
            memory_usage_kb=1024,
            timed_out=False,
            duration_seconds=0.08,
            container_name="sandbox-test",
        )


class StubTestRunner:
    def run_json(
        self,
        language: str,
        source_code: str,
        test_cases_json: str,
        limits: object | None = None,
        main_class_name: str = "Main",
    ) -> PredefinedTestRunResult:
        case = PredefinedTestCaseResult(
            case_index=0,
            passed=True,
            input_data="1 2\n",
            expected_output="3\n",
            actual_output="3\n",
            stderr="",
            exit_code=0,
            timed_out=False,
            runtime_ms=15,
            memory_usage_kb=1024,
            description="sum",
            output_diff=None,
            failure_reason=None,
        )
        return PredefinedTestRunResult(
            total=1,
            passed=1,
            failed=0,
            case_results=[case],
            first_failed_case=None,
            first_failed_report=None,
        )


class StubStaticAnalyzer:
    def analyze(self, source_code: str, filename: str = "main.py") -> PythonStaticAnalysisResult:
        return PythonStaticAnalysisResult(
            language="python",
            pylint_score=9.1,
            pylint_warnings=[
                PylintWarning(
                    message_id="C0114",
                    symbol="missing-module-docstring",
                    message="Missing module docstring",
                    path="main.py",
                    line=1,
                    column=0,
                )
            ],
            complexity_score=2.0,
            complexity_blocks=[
                RadonComplexityBlock(
                    name="solve",
                    block_type="function",
                    complexity=2,
                    line=1,
                    endline=3,
                )
            ],
            security_warnings=[
                BanditWarning(
                    test_id="B101",
                    test_name="assert_used",
                    issue_severity="LOW",
                    issue_confidence="HIGH",
                    issue_text="Use of assert detected.",
                    line_number=2,
                )
            ],
            tool_errors=[],
        )


class StubLLMReviewer:
    def review(self, state: AgentState) -> dict[str, object]:
        return {
            "summary": "Structured review.",
            "strengths": ["Passes all tests."],
            "issues": [],
            "improvement_suggestions": ["Add more edge-case coverage."],
            "score_template": {
                "correctness": 90.0,
                "performance": 85.0,
                "readability": 80.0,
                "security": 95.0,
                "final_score": 88.0,
            },
            "score": 88.0,
            "reviewer": "stub",
        }


class CountingLLMReviewer:
    def __init__(self) -> None:
        self.calls = 0

    def review(self, state: AgentState) -> dict[str, object]:
        _ = state
        self.calls += 1
        return {
            "summary": "Counted review.",
            "strengths": ["Baseline feedback."],
            "issues": [],
            "improvement_suggestions": ["Keep tests deterministic."],
            "score_template": {
                "correctness": 80.0,
                "performance": 80.0,
                "readability": 80.0,
                "security": 80.0,
                "final_score": 80.0,
            },
            "score": 80.0,
            "reviewer": "counting-reviewer",
        }


class FailingLLMReviewer:
    def review(self, state: AgentState) -> dict[str, object]:
        _ = state
        raise RuntimeError("simulated llm outage")


class FailingSandboxExecutor:
    def execute(
        self,
        language: str,
        source_code: str,
        limits: object | None = None,
        main_class_name: str = "Main",
        stdin_data: str = "",
    ) -> SandboxExecutionResult:
        _ = (language, source_code, limits, main_class_name, stdin_data)
        raise RuntimeError("simulated sandbox crash")


class MalformedFallbackLLMReviewer:
    def review(self, state: AgentState) -> dict[str, object]:
        return {
            "summary": "  ",
            "strengths": [""],
            "issues": "not-a-list",
            "improvement_suggestions": "not-a-list",
            "score_template": {"correctness": "bad"},
            "score": "not-a-number",
        }


class FlakyStructuredReviewChain:
    def __init__(self, responses: list[object]) -> None:
        self._responses = responses
        self.calls = 0

    def invoke(self, payload: dict[str, object]) -> object:
        _ = payload
        self.calls += 1
        if self.calls > len(self._responses):
            raise RuntimeError("No response configured for this attempt.")

        response = self._responses[self.calls - 1]
        if isinstance(response, Exception):
            raise response
        return response


class TokenUsageStructuredReviewChain:
    def invoke(self, payload: dict[str, object]) -> object:
        _ = payload
        return {
            "review_payload": {
                "summary": "Structured cloud review.",
                "strengths": ["All tests passed."],
                "issues": ["Minor naming issue."],
                "score": 91.0,
            },
            "token_usage": {
                "input_tokens": 120,
                "output_tokens": 45,
                "total_tokens": 165,
            },
        }


class GraphNodesTests(unittest.TestCase):
    def test_generate_task_node_populates_missing_task_fields(self) -> None:
        updated_state = generate_task_node(
            state={
                "user_id": "42",
                "task_id": "task-1",
                "language": "python",
                "code": "print(sum(map(int, input().split())))",
            },
            task_generator=StubTaskGenerator(),
        )

        self.assertEqual(updated_state["task_prompt"], "Add two numbers from stdin.")
        self.assertEqual(updated_state["task_category"], "math")
        self.assertEqual(updated_state["task_difficulty"], 2)
        self.assertIn('"input":"1 2\\n"', updated_state["test_cases"])

    def test_generate_task_node_applies_recommended_difficulty(self) -> None:
        updated_state = generate_task_node(
            state={
                "user_id": "42",
                "task_id": "task-2",
                "language": "python",
                "code": "print(sum(map(int, input().split())))",
                "task_difficulty": 2,
                "recommended_difficulty": 3,
            },
            task_generator=StubTaskGenerator(),
        )

        self.assertEqual(updated_state["task_difficulty"], 3)

    def test_run_tests_node_writes_empty_payload_if_test_cases_missing(self) -> None:
        updated_state = run_tests_node(
            state={
                "user_id": "42",
                "task_id": "task-1",
                "language": "python",
                "code": "print('ok')",
            }
        )

        self.assertIn("test_results", updated_state)
        self.assertEqual(updated_state["test_results"]["total"], 0)
        self.assertEqual(updated_state["test_results"]["passed"], 0)
        self.assertEqual(updated_state["test_results"]["failed"], 0)

    def test_llm_review_node_builds_payload_with_heuristic_reviewer(self) -> None:
        updated_state = llm_review_node(
            state={
                "user_id": "42",
                "task_id": "task-1",
                "language": "python",
                "code": "print('ok')",
                "execution_result": {
                    "exit_code": 0,
                    "stdout": "ok\n",
                    "stderr": "",
                    "timed_out": False,
                },
                "test_results": {
                    "total": 2,
                    "passed": 2,
                    "failed": 0,
                    "first_failed_report": None,
                    "case_results": [],
                },
                "static_analysis": {
                    "language": "python",
                    "pylint_score": 8.5,
                    "complexity_score": 3.0,
                    "security_warnings": [],
                    "security_warnings_summary": "No security warnings.",
                    "pylint_warnings": [],
                    "tool_errors": [],
                },
            }
        )

        self.assertIn("llm_review", updated_state)
        self.assertIn("summary", updated_state["llm_review"])
        self.assertIn("score", updated_state["llm_review"])
        self.assertIn("score_template", updated_state["llm_review"])
        self.assertIn("improvement_suggestions", updated_state["llm_review"])
        llm_review = cast(dict[str, object], updated_state["llm_review"])
        llm_score = llm_review.get("score")
        self.assertIsInstance(llm_score, (int, float))
        if not isinstance(llm_score, int | float):
            self.fail("Expected numeric llm review score.")
        self.assertGreaterEqual(float(llm_score), 0.0)
        self.assertIsInstance(llm_review.get("score_template"), dict)
        self.assertIsInstance(llm_review.get("improvement_suggestions"), list)
        self.assertIn("observability_metrics", updated_state)
        observability_metrics_raw = updated_state.get("observability_metrics")
        self.assertIsInstance(observability_metrics_raw, dict)
        observability_metrics = cast(dict[str, object], observability_metrics_raw)
        self.assertEqual(observability_metrics["fail_rate"], 0.0)
        self.assertEqual(observability_metrics["cost_per_submission"], 0.0)

    def test_llm_review_node_computes_metrics_from_tests_and_token_usage(self) -> None:
        class TokenAwareReviewer:
            def review(self, state: AgentState) -> dict[str, object]:
                _ = state
                return {
                    "summary": "Cloud review with token usage.",
                    "strengths": ["Passes core logic."],
                    "issues": ["One edge case fails."],
                    "improvement_suggestions": ["Cover additional edge cases."],
                    "score_template": {
                        "correctness": 75.0,
                        "performance": 80.0,
                        "readability": 78.0,
                        "security": 90.0,
                        "final_score": 79.9,
                    },
                    "score": 79.9,
                    "reviewer": "token-aware",
                    "token_usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 25,
                        "total_tokens": 125,
                    },
                }

        updated_state = llm_review_node(
            state={
                "user_id": "42",
                "task_id": "task-metrics",
                "language": "python",
                "code": "print('ok')",
                "execution_result": {
                    "exit_code": 0,
                    "stdout": "ok\n",
                    "stderr": "",
                    "timed_out": False,
                },
                "metrics": {
                    "runtime_ms": 120,
                    "memory_usage_kb": 1024,
                    "exit_code": 0,
                    "stdout": "ok\n",
                    "stderr": "",
                    "timed_out": False,
                },
                "test_results": {
                    "total": 4,
                    "passed": 3,
                    "failed": 1,
                    "first_failed_report": "Case #4 failed.",
                    "case_results": [
                        {"runtime_ms": 40},
                        {"runtime_ms": 60},
                        {"runtime_ms": 50},
                        {"runtime_ms": 70},
                    ],
                },
            },
            llm_reviewer=TokenAwareReviewer(),
        )

        observability_metrics_raw = updated_state.get("observability_metrics")
        self.assertIsInstance(observability_metrics_raw, dict)
        observability_metrics = cast(dict[str, object], observability_metrics_raw)
        self.assertEqual(observability_metrics["avg_runtime_ms"], 55.0)
        self.assertEqual(observability_metrics["fail_rate"], 0.25)
        cost_per_submission = observability_metrics.get("cost_per_submission")
        self.assertIsInstance(cost_per_submission, (int, float))
        if not isinstance(cost_per_submission, int | float):
            self.fail("Expected numeric cost_per_submission.")
        self.assertAlmostEqual(float(cost_per_submission), 0.000041, places=8)

    def test_langchain_cloud_llm_reviewer_retries_until_success(self) -> None:
        chain = FlakyStructuredReviewChain(
            responses=[
                RuntimeError("temporary parsing failure"),
                {
                    "summary": "Structured cloud review.",
                    "strengths": ["All tests passed."],
                    "issues": ["Minor naming issue."],
                    "score": 91.0,
                },
            ]
        )

        reviewer = LangChainCloudLLMReviewer(
            api_key="test-key",
            max_retries=2,
            chain_factory=lambda: chain,
        )
        review = reviewer.review(
            {
                "user_id": "42",
                "task_id": "task-1",
                "language": "python",
                "code": "print('ok')",
            }
        )

        self.assertEqual(chain.calls, 2)
        self.assertEqual(review["reviewer"], "langchain-cloud")
        self.assertEqual(review["score"], 91.0)
        self.assertEqual(review["summary"], "Structured cloud review.")
        self.assertIn("score_template", review)
        self.assertIn("improvement_suggestions", review)

    def test_langchain_cloud_llm_reviewer_logs_token_usage(self) -> None:
        reviewer = LangChainCloudLLMReviewer(
            api_key="test-key",
            chain_factory=lambda: TokenUsageStructuredReviewChain(),
        )

        with self.assertLogs("interview_orchestrator.pipeline", level="INFO") as captured:
            review = reviewer.review(
                {
                    "user_id": "42",
                    "task_id": "task-1",
                    "language": "python",
                    "code": "print('ok')",
                }
            )

        self.assertEqual(review["reviewer"], "langchain-cloud")
        payloads = [_parse_json_log_record(record) for record in captured.output]
        token_usage_events = [
            payload
            for payload in payloads
            if payload.get("event") == "submission.llm_review.token_usage"
        ]
        self.assertEqual(len(token_usage_events), 1)
        token_usage = cast(dict[str, object], token_usage_events[0]["token_usage"])
        self.assertEqual(token_usage["prompt_tokens"], 120)
        self.assertEqual(token_usage["completion_tokens"], 45)
        self.assertEqual(token_usage["total_tokens"], 165)

    def test_langchain_cloud_llm_reviewer_returns_valid_json_after_failures(self) -> None:
        chain = FlakyStructuredReviewChain(
            responses=[
                RuntimeError("provider timeout"),
                RuntimeError("provider timeout"),
            ]
        )

        reviewer = LangChainCloudLLMReviewer(
            api_key="test-key",
            max_retries=1,
            fallback_reviewer=MalformedFallbackLLMReviewer(),
            chain_factory=lambda: chain,
        )
        review = reviewer.review(
            {
                "user_id": "42",
                "task_id": "task-1",
                "language": "python",
                "code": "print('ok')",
            }
        )

        self.assertEqual(chain.calls, 2)
        self.assertEqual(review["reviewer"], "heuristic_fallback")
        self.assertIsInstance(review["summary"], str)
        self.assertNotEqual(str(review["summary"]).strip(), "")
        self.assertIsInstance(review["strengths"], list)
        self.assertGreater(len(cast(list[object], review["strengths"])), 0)
        self.assertIsInstance(review["issues"], list)
        self.assertGreater(len(cast(list[object], review["issues"])), 0)
        self.assertIsInstance(review["improvement_suggestions"], list)
        self.assertGreater(len(cast(list[object], review["improvement_suggestions"])), 0)
        self.assertIsInstance(review["score_template"], dict)
        self.assertIsInstance(review["score"], (int, float))

    def test_llm_review_node_uses_score_template_final_score_when_present(self) -> None:
        class ScoreTemplateReviewer:
            def review(self, state: AgentState) -> dict[str, object]:
                _ = state
                return {
                    "summary": "Template-driven score.",
                    "strengths": ["Covers core requirements."],
                    "issues": ["Needs clearer naming."],
                    "improvement_suggestions": ["Rename variables for readability."],
                    "score_template": {
                        "correctness": 84.0,
                        "performance": 80.0,
                        "readability": 70.0,
                        "security": 90.0,
                        "final_score": 82.5,
                    },
                    "score": 10.0,
                    "reviewer": "template-reviewer",
                }

        updated_state = llm_review_node(
            state={
                "user_id": "42",
                "task_id": "task-1",
                "language": "python",
                "code": "print('ok')",
            },
            llm_reviewer=ScoreTemplateReviewer(),
        )

        llm_review = cast(dict[str, object], updated_state["llm_review"])
        self.assertEqual(llm_review["score"], 82.5)
        score_template = cast(dict[str, object], llm_review["score_template"])
        self.assertEqual(score_template["final_score"], 82.5)

    def test_llm_review_node_falls_back_when_reviewer_raises(self) -> None:
        updated_state = llm_review_node(
            state={
                "user_id": "42",
                "task_id": "task-llm-failure",
                "language": "python",
                "code": "print('ok')",
            },
            llm_reviewer=FailingLLMReviewer(),
            llm_rate_limiter=InMemoryLLMRateLimiter(max_calls=5, window_seconds=120),
        )

        review = cast(dict[str, object], updated_state["llm_review"])
        self.assertEqual(review["reviewer"], "heuristic_fallback")
        self.assertIn("llm_error", review)
        issues = cast(list[str], review["issues"])
        self.assertTrue(any("fallback" in issue.lower() for issue in issues))
        self.assertIn("observability_metrics", updated_state)

    def test_llm_review_node_blocks_second_call_when_rate_limit_is_exceeded(self) -> None:
        reviewer = CountingLLMReviewer()
        limiter = InMemoryLLMRateLimiter(max_calls=1, window_seconds=120)
        state: AgentState = {
            "user_id": "rate-limit-user",
            "task_id": "task-rate-limit",
            "language": "python",
            "code": "print('ok')",
        }

        first_state = llm_review_node(
            state=state,
            llm_reviewer=reviewer,
            llm_rate_limiter=limiter,
        )
        second_state = llm_review_node(
            state=state,
            llm_reviewer=reviewer,
            llm_rate_limiter=limiter,
        )

        self.assertEqual(reviewer.calls, 1)
        first_review = cast(dict[str, object], first_state["llm_review"])
        second_review = cast(dict[str, object], second_state["llm_review"])
        self.assertEqual(first_review["reviewer"], "counting-reviewer")
        self.assertEqual(second_review["reviewer"], "heuristic_rate_limited")
        issues = cast(list[str], second_review["issues"])
        self.assertTrue(any("rate limit" in issue.lower() for issue in issues))
        self.assertIn("observability_metrics", second_state)

    def test_llm_review_node_rate_limit_is_tracked_per_user(self) -> None:
        reviewer = CountingLLMReviewer()
        limiter = InMemoryLLMRateLimiter(max_calls=1, window_seconds=120)

        _ = llm_review_node(
            state={
                "user_id": "rate-user-a",
                "task_id": "task-1",
                "language": "python",
                "code": "print('ok')",
            },
            llm_reviewer=reviewer,
            llm_rate_limiter=limiter,
        )
        _ = llm_review_node(
            state={
                "user_id": "rate-user-b",
                "task_id": "task-2",
                "language": "python",
                "code": "print('ok')",
            },
            llm_reviewer=reviewer,
            llm_rate_limiter=limiter,
        )

        self.assertEqual(reviewer.calls, 2)

    def test_score_aggregation_node_writes_weighted_dimension_breakdown(self) -> None:
        state: AgentState = {
            "user_id": "42",
            "task_id": "task-1",
            "language": "python",
            "code": "print('ok')",
            "metrics": {
                "runtime_ms": 120,
                "memory_usage_kb": 1024,
                "exit_code": 0,
                "stdout": "ok\n",
                "stderr": "",
                "timed_out": False,
            },
            "test_results": {
                "total": 4,
                "passed": 3,
                "failed": 1,
                "first_failed_report": "Mismatch",
                "case_results": [],
            },
            "static_analysis": {
                "language": "python",
                "pylint_score": 9.0,
                "complexity_score": 3.0,
                "security_warnings": [],
                "security_warnings_summary": "No security warnings.",
                "pylint_warnings": [],
                "tool_errors": [],
            },
            "llm_review": {"score": 80.0},
        }

        updated_state = score_aggregation_node(state=state)
        score_breakdown = cast(dict[str, object], updated_state["score_breakdown"])

        self.assertEqual(updated_state["final_score"], 85.28)
        self.assertEqual(score_breakdown["correctness_score"], 75.0)
        self.assertEqual(score_breakdown["performance_score"], 100.0)
        self.assertEqual(score_breakdown["readability_score"], 93.5)
        self.assertEqual(score_breakdown["security_score"], 100.0)

        weights = cast(dict[str, float], score_breakdown["weights"])
        self.assertEqual(weights["correctness_weight"], 0.55)
        self.assertEqual(weights["performance_weight"], 0.2)
        self.assertEqual(weights["readability_weight"], 0.15)
        self.assertEqual(weights["security_weight"], 0.1)

    def test_score_aggregation_node_is_reproducible_for_same_signals(self) -> None:
        base_state: AgentState = {
            "user_id": "42",
            "task_id": "task-1",
            "language": "python",
            "code": "print('ok')",
            "metrics": {
                "runtime_ms": 120,
                "memory_usage_kb": 1024,
                "exit_code": 0,
                "stdout": "ok\n",
                "stderr": "",
                "timed_out": False,
            },
            "test_results": {
                "total": 4,
                "passed": 3,
                "failed": 1,
                "first_failed_report": "Mismatch",
                "case_results": [],
            },
            "static_analysis": {
                "language": "python",
                "pylint_score": 9.0,
                "complexity_score": 3.0,
                "security_warnings": [],
                "security_warnings_summary": "No security warnings.",
                "pylint_warnings": [],
                "tool_errors": [],
            },
            "llm_review": {"score": 5.0},
        }
        state_with_different_review = cast(AgentState, dict(base_state))
        state_with_different_review["llm_review"] = {"score": 99.0}

        first = score_aggregation_node(state=base_state)
        second = score_aggregation_node(state=state_with_different_review)

        self.assertEqual(first["final_score"], second["final_score"])
        self.assertEqual(first["score_breakdown"], second["score_breakdown"])

    def test_score_aggregation_node_supports_custom_dimension_weights(self) -> None:
        updated_state = score_aggregation_node(
            state={
                "user_id": "42",
                "task_id": "task-1",
                "language": "python",
                "code": "print('ok')",
                "metrics": {
                    "runtime_ms": 120,
                    "memory_usage_kb": 1024,
                    "exit_code": 0,
                    "stdout": "ok\n",
                    "stderr": "",
                    "timed_out": False,
                },
                "test_results": {
                    "total": 4,
                    "passed": 3,
                    "failed": 1,
                    "first_failed_report": "Mismatch",
                    "case_results": [],
                },
                "static_analysis": {
                    "language": "python",
                    "pylint_score": 9.0,
                    "complexity_score": 3.0,
                    "security_warnings": [],
                    "security_warnings_summary": "No security warnings.",
                    "pylint_warnings": [],
                    "tool_errors": [],
                },
            },
            weights=ScoreAggregationWeights(
                correctness_weight=1.0,
                performance_weight=0.0,
                readability_weight=0.0,
                security_weight=0.0,
            ),
        )

        self.assertEqual(updated_state["final_score"], 75.0)

    def test_update_profile_node_updates_language_scores(self) -> None:
        updated_state = update_profile_node(
            state={
                "user_id": "42",
                "task_id": "task-1",
                "language": "python",
                "code": "print('ok')",
                "task_category": "arrays",
                "final_score": 90.0,
            }
        )

        skill_profile = updated_state["skill_profile"]
        profile_map = cast(dict[str, object], skill_profile)
        language_scores = cast(dict[str, object], profile_map["language_scores"])
        python_stats = cast(dict[str, object], language_scores["python"])

        self.assertEqual(python_stats["attempts"], 1)
        self.assertEqual(python_stats["last_score"], 90.0)
        self.assertEqual(python_stats["best_score"], 90.0)
        self.assertEqual(profile_map["failed_categories"], {})
        self.assertEqual(
            profile_map["complexity_mistakes"],
            {"total": 0, "by_category": {}, "last_score": None},
        )

    def test_update_profile_node_tracks_failed_category_and_complexity_mistakes(self) -> None:
        first_state = update_profile_node(
            state={
                "user_id": "42",
                "task_id": "task-1",
                "language": "python",
                "code": "print('first')",
                "task_category": "arrays",
                "final_score": 40.0,
                "test_results": {
                    "total": 2,
                    "passed": 0,
                    "failed": 2,
                    "first_failed_report": "Output mismatch.",
                    "case_results": [],
                },
                "static_analysis": {
                    "language": "python",
                    "pylint_score": 6.0,
                    "complexity_score": 9.1,
                    "security_warnings": [],
                    "security_warnings_summary": "No security warnings.",
                    "pylint_warnings": [],
                    "tool_errors": [],
                },
            }
        )

        second_state = update_profile_node(
            state={
                "user_id": "42",
                "task_id": "task-2",
                "language": "python",
                "code": "print('second')",
                "task_category": "arrays",
                "final_score": 45.0,
                "test_results": {
                    "total": 3,
                    "passed": 1,
                    "failed": 2,
                    "first_failed_report": "Edge case failed.",
                    "case_results": [],
                },
                "static_analysis": {
                    "language": "python",
                    "pylint_score": 7.0,
                    "complexity_score": 9.6,
                    "security_warnings": [],
                    "security_warnings_summary": "No security warnings.",
                    "pylint_warnings": [],
                    "tool_errors": [],
                },
                "skill_profile": cast(dict[str, object], first_state["skill_profile"]),
            }
        )

        profile_map = cast(dict[str, object], second_state["skill_profile"])
        failed_categories = cast(dict[str, int], profile_map["failed_categories"])
        complexity_mistakes = cast(dict[str, object], profile_map["complexity_mistakes"])
        by_category = cast(dict[str, int], complexity_mistakes["by_category"])

        self.assertEqual(failed_categories["arrays"], 2)
        self.assertEqual(complexity_mistakes["total"], 2)
        self.assertEqual(by_category["arrays"], 2)
        self.assertEqual(complexity_mistakes["last_score"], 9.6)

    def test_branching_logic_node_routes_retry_with_hint_on_failure(self) -> None:
        updated_state = branching_logic_node(
            state={
                "user_id": "42",
                "task_id": "task-1",
                "language": "python",
                "code": "print('boom')",
                "task_difficulty": 2,
                "retry_count": 0,
                "execution_result": {
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": "NameError: boom",
                    "timed_out": False,
                },
                "test_results": {
                    "total": 2,
                    "passed": 0,
                    "failed": 2,
                    "first_failed_report": "Test case #1 failed. Reason: Output mismatch.",
                    "case_results": [],
                },
                "llm_review": {
                    "issues": ["Incorrect output formatting."],
                },
                "static_analysis": {
                    "language": "python",
                    "pylint_score": 4.0,
                    "complexity_score": 7.0,
                    "security_warnings": [],
                    "security_warnings_summary": "No security warnings.",
                    "pylint_warnings": [],
                    "tool_errors": [],
                },
            },
            policy=BranchingPolicy(max_retries=1, max_hints=3),
        )

        self.assertEqual(updated_state["retry_count"], 1)
        self.assertEqual(updated_state["recommended_difficulty"], 2)
        branching = cast(dict[str, object], updated_state["branching"])
        self.assertEqual(branching["next_node"], "retry_with_hint")
        retry = cast(dict[str, object], branching["retry"])
        self.assertEqual(retry["should_retry"], True)
        self.assertEqual(retry["retries_used"], 1)
        hint = cast(dict[str, object], branching["hint"])
        self.assertEqual(hint["should_show_hint"], True)
        hints = cast(list[str], hint["hints"])
        self.assertGreater(len(hints), 0)

    def test_branching_logic_node_increases_difficulty_on_high_score_streak(self) -> None:
        updated_state = branching_logic_node(
            state={
                "user_id": "42",
                "task_id": "task-1",
                "language": "python",
                "code": "print('ok')",
                "task_difficulty": 2,
                "retry_count": 1,
                "execution_result": {
                    "exit_code": 0,
                    "stdout": "ok\n",
                    "stderr": "",
                    "timed_out": False,
                },
                "test_results": {
                    "total": 2,
                    "passed": 2,
                    "failed": 0,
                    "first_failed_report": None,
                    "case_results": [],
                },
                "skill_profile": {
                    "recent_scores": [90.0, 92.0, 95.0],
                },
            }
        )

        branching = cast(dict[str, object], updated_state["branching"])
        adaptive = cast(dict[str, object], branching["adaptive_difficulty"])
        self.assertEqual(adaptive["action"], "increase")
        self.assertEqual(adaptive["current_difficulty"], 2)
        self.assertEqual(adaptive["next_difficulty"], 3)
        self.assertEqual(updated_state["recommended_difficulty"], 3)
        self.assertEqual(updated_state["retry_count"], 0)
        self.assertEqual(branching["next_node"], "complete")

    def test_run_full_graph_cycle_completes_without_failures(self) -> None:
        final_state = run_full_graph_cycle(
            state={
                "user_id": "42",
                "task_id": "task-1",
                "language": "python",
                "code": "print(sum(map(int, input().split())))",
            },
            task_generator=StubTaskGenerator(),
            sandbox_executor=StubSandboxExecutor(),
            test_runner=StubTestRunner(),
            static_analyzer=StubStaticAnalyzer(),
            llm_reviewer=StubLLMReviewer(),
        )

        self.assertIn("task_prompt", final_state)
        self.assertIn("execution_result", final_state)
        self.assertIn("test_results", final_state)
        self.assertIn("static_analysis", final_state)
        self.assertIn("llm_review", final_state)
        self.assertIn("score_breakdown", final_state)
        self.assertIn("final_score", final_state)
        self.assertIn("skill_profile", final_state)
        self.assertIn("branching", final_state)
        self.assertIn("recommended_difficulty", final_state)
        self.assertIn("observability_metrics", final_state)

    def test_run_full_graph_cycle_emits_structured_submission_logs(self) -> None:
        with self.assertLogs("interview_orchestrator.pipeline", level="INFO") as captured:
            _ = run_full_graph_cycle(
                state={
                    "user_id": "42",
                    "task_id": "task-structured-logs",
                    "language": "python",
                    "code": "print(sum(map(int, input().split())))",
                },
                task_generator=StubTaskGenerator(),
                sandbox_executor=StubSandboxExecutor(),
                test_runner=StubTestRunner(),
                static_analyzer=StubStaticAnalyzer(),
                llm_reviewer=StubLLMReviewer(),
            )

        payloads = [_parse_json_log_record(record) for record in captured.output]
        events = [str(payload.get("event")) for payload in payloads]
        self.assertIn("submission.pipeline.started", events)
        self.assertIn("submission.pipeline.completed", events)
        self.assertIn("submission.metrics.computed", events)
        self.assertGreaterEqual(events.count("submission.pipeline.step_completed"), 8)
        completed_payload = next(
            payload
            for payload in payloads
            if payload.get("event") == "submission.pipeline.completed"
        )
        self.assertIn("avg_runtime_ms", completed_payload)
        self.assertIn("fail_rate", completed_payload)
        self.assertIn("cost_per_submission", completed_payload)

    def test_run_full_graph_cycle_recovers_from_partial_failures(self) -> None:
        final_state = run_full_graph_cycle(
            state={
                "user_id": "42",
                "task_id": "task-partial-failures",
                "language": "python",
                "code": "print(sum(map(int, input().split())))",
            },
            task_generator=StubTaskGenerator(),
            sandbox_executor=FailingSandboxExecutor(),
            test_runner=StubTestRunner(),
            static_analyzer=StubStaticAnalyzer(),
            llm_reviewer=FailingLLMReviewer(),
            llm_rate_limiter=InMemoryLLMRateLimiter(max_calls=10, window_seconds=120),
        )

        execution_result = cast(dict[str, object], final_state["execution_result"])
        self.assertIsNone(execution_result["exit_code"])
        self.assertIn("Sandbox execution failed", str(execution_result["stderr"]))
        llm_review = cast(dict[str, object], final_state["llm_review"])
        self.assertEqual(llm_review["reviewer"], "heuristic_fallback")
        self.assertIn("final_score", final_state)


if __name__ == "__main__":
    unittest.main()
