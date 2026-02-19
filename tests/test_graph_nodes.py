from __future__ import annotations

import unittest

from interview_orchestrator.sandbox_runner import SandboxExecutionResult
from interview_orchestrator.state_steps import (
    GeneratedTask,
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
    def review(self, state: dict[str, object]) -> dict[str, object]:
        return {
            "summary": "Structured review.",
            "strengths": ["Passes all tests."],
            "issues": [],
            "score": 88.0,
            "reviewer": "stub",
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
        self.assertGreaterEqual(float(updated_state["llm_review"]["score"]), 0.0)

    def test_score_aggregation_node_writes_final_score_and_breakdown(self) -> None:
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
                "llm_review": {"score": 80.0},
            }
        )

        self.assertIn("final_score", updated_state)
        self.assertIn("score_breakdown", updated_state)
        self.assertGreaterEqual(float(updated_state["final_score"]), 0.0)
        self.assertLessEqual(float(updated_state["final_score"]), 100.0)

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
        language_scores = skill_profile["language_scores"]
        python_stats = language_scores["python"]

        self.assertEqual(python_stats["attempts"], 1)
        self.assertEqual(python_stats["last_score"], 90.0)
        self.assertEqual(python_stats["best_score"], 90.0)

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


if __name__ == "__main__":
    unittest.main()
