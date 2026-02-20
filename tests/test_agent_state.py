from __future__ import annotations

import unittest

from interview_orchestrator.agent_state import (
    AgentStateValidationError,
    validate_agent_state,
)
from interview_orchestrator.sandbox_runner import SandboxExecutionResult
from interview_orchestrator.state_steps import (
    execute_sandbox_step,
    run_core_orchestration_steps,
    run_predefined_tests_step,
    run_python_static_analysis_step,
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


class StubSandboxExecutor:
    def __init__(self, result: SandboxExecutionResult) -> None:
        self._result = result
        self.calls: list[dict[str, object]] = []

    def execute(
        self,
        language: str,
        source_code: str,
        limits: object | None = None,
        main_class_name: str = "Main",
        stdin_data: str = "",
    ) -> SandboxExecutionResult:
        self.calls.append(
            {
                "language": language,
                "source_code": source_code,
                "limits": limits,
                "main_class_name": main_class_name,
                "stdin_data": stdin_data,
            }
        )
        return self._result


class StubTestRunner:
    def __init__(self, result: PredefinedTestRunResult) -> None:
        self._result = result
        self.calls: list[dict[str, object]] = []

    def run_json(
        self,
        language: str,
        source_code: str,
        test_cases_json: str,
        limits: object | None = None,
        main_class_name: str = "Main",
    ) -> PredefinedTestRunResult:
        self.calls.append(
            {
                "language": language,
                "source_code": source_code,
                "test_cases_json": test_cases_json,
                "limits": limits,
                "main_class_name": main_class_name,
            }
        )
        return self._result


class StubStaticAnalyzer:
    def __init__(self, result: PythonStaticAnalysisResult) -> None:
        self._result = result
        self.calls: list[dict[str, object]] = []

    def analyze(self, source_code: str, filename: str = "main.py") -> PythonStaticAnalysisResult:
        self.calls.append({"source_code": source_code, "filename": filename})
        return self._result


def _sandbox_result() -> SandboxExecutionResult:
    return SandboxExecutionResult(
        language="python",
        exit_code=0,
        stdout="ok\n",
        stderr="",
        memory_usage_kb=2048,
        timed_out=False,
        duration_seconds=0.12,
        container_name="sandbox-test",
    )


def _test_run_result() -> PredefinedTestRunResult:
    case_result = PredefinedTestCaseResult(
        case_index=0,
        passed=False,
        input_data="1\n2\n",
        expected_output="3\n",
        actual_output="4\n",
        stderr="",
        exit_code=0,
        timed_out=False,
        runtime_ms=20,
        memory_usage_kb=1024,
        description="sum",
        output_diff="--- expected\n+++ actual\n@@\n-3\n+4",
        failure_reason="Output mismatch.",
    )
    return PredefinedTestRunResult(
        total=1,
        passed=0,
        failed=1,
        case_results=[case_result],
        first_failed_case=case_result,
        first_failed_report="Test case #1 failed. Reason: Output mismatch.",
    )


def _python_static_analysis_result() -> PythonStaticAnalysisResult:
    return PythonStaticAnalysisResult(
        language="python",
        pylint_score=8.5,
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
        complexity_score=3.0,
        complexity_blocks=[
            RadonComplexityBlock(
                name="solve",
                block_type="function",
                complexity=3,
                line=1,
                endline=4,
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


class AgentStateTests(unittest.TestCase):
    def test_validate_agent_state_accepts_minimal_payload(self) -> None:
        state = validate_agent_state(
            {
                "user_id": "42",
                "task_id": "task-1",
                "language": "python",
                "code": "print('ok')",
            }
        )
        self.assertEqual(state["user_id"], "42")
        self.assertEqual(state["language"], "python")

    def test_validate_agent_state_rejects_missing_required_fields(self) -> None:
        with self.assertRaises(AgentStateValidationError):
            validate_agent_state(
                {
                    "user_id": "42",
                    "task_id": "task-1",
                    "language": "python",
                }
            )

    def test_validate_agent_state_rejects_unsupported_language(self) -> None:
        with self.assertRaises(AgentStateValidationError):
            validate_agent_state(
                {
                    "user_id": "42",
                    "task_id": "task-1",
                    "language": "rust",
                    "code": "fn main() {}",
                }
            )

    def test_validate_agent_state_accepts_observability_metrics(self) -> None:
        state = validate_agent_state(
            {
                "user_id": "42",
                "task_id": "task-1",
                "language": "python",
                "code": "print('ok')",
                "observability_metrics": {
                    "avg_runtime_ms": 35.5,
                    "fail_rate": 0.25,
                    "cost_per_submission": 0.000031,
                },
            }
        )
        self.assertIn("observability_metrics", state)
        metrics = state["observability_metrics"]
        self.assertEqual(metrics["avg_runtime_ms"], 35.5)
        self.assertEqual(metrics["fail_rate"], 0.25)
        self.assertEqual(metrics["cost_per_submission"], 0.000031)

    def test_validate_agent_state_rejects_invalid_observability_fail_rate(self) -> None:
        with self.assertRaises(AgentStateValidationError):
            validate_agent_state(
                {
                    "user_id": "42",
                    "task_id": "task-1",
                    "language": "python",
                    "code": "print('ok')",
                    "observability_metrics": {
                        "avg_runtime_ms": 10.0,
                        "fail_rate": 1.2,
                        "cost_per_submission": 0.0,
                    },
                }
            )

    def test_validate_agent_state_accepts_branching_payload(self) -> None:
        state = validate_agent_state(
            {
                "user_id": "42",
                "task_id": "task-1",
                "language": "python",
                "code": "print('ok')",
                "retry_count": 1,
                "recommended_difficulty": 2,
                "branching": {
                    "retry": {
                        "should_retry": False,
                        "retries_used": 1,
                        "retries_remaining": 0,
                        "max_retries": 1,
                        "reason": "Retry limit reached.",
                    },
                    "adaptive_difficulty": {
                        "action": "decrease",
                        "current_difficulty": 2,
                        "next_difficulty": 1,
                        "reason": "Repeated failures detected.",
                    },
                    "hint": {
                        "should_show_hint": True,
                        "hints": ["Check first failing test case."],
                    },
                    "next_node": "show_hint",
                },
            }
        )

        self.assertIn("branching", state)
        self.assertEqual(state["recommended_difficulty"], 2)

    def test_validate_agent_state_rejects_invalid_branching_action(self) -> None:
        with self.assertRaises(AgentStateValidationError):
            validate_agent_state(
                {
                    "user_id": "42",
                    "task_id": "task-1",
                    "language": "python",
                    "code": "print('ok')",
                    "branching": {
                        "retry": {
                            "should_retry": False,
                            "retries_used": 0,
                            "retries_remaining": 0,
                            "max_retries": 1,
                            "reason": "ok",
                        },
                        "adaptive_difficulty": {
                            "action": "up",
                            "current_difficulty": 1,
                            "next_difficulty": 2,
                            "reason": "ok",
                        },
                        "hint": {
                            "should_show_hint": False,
                            "hints": [],
                        },
                        "next_node": "complete",
                    },
                }
            )

    def test_execute_sandbox_step_writes_execution_result_and_metrics(self) -> None:
        sandbox = StubSandboxExecutor(_sandbox_result())
        updated_state = execute_sandbox_step(
            state={
                "user_id": "7",
                "task_id": "task-a",
                "language": "python",
                "code": "print('ok')",
                "stdin_data": "input\n",
            },
            sandbox_executor=sandbox,
        )

        self.assertIn("execution_result", updated_state)
        self.assertIn("metrics", updated_state)
        self.assertEqual(updated_state["execution_result"]["stdout"], "ok\n")
        self.assertEqual(updated_state["metrics"]["runtime_ms"], 120)
        self.assertEqual(sandbox.calls[0]["stdin_data"], "input\n")

    def test_run_predefined_tests_step_requires_test_cases(self) -> None:
        with self.assertRaises(ValueError):
            run_predefined_tests_step(
                state={
                    "user_id": "7",
                    "task_id": "task-a",
                    "language": "python",
                    "code": "print('ok')",
                },
            )

    def test_run_predefined_tests_step_writes_test_results(self) -> None:
        test_runner = StubTestRunner(_test_run_result())
        updated_state = run_predefined_tests_step(
            state={
                "user_id": "7",
                "task_id": "task-a",
                "language": "python",
                "code": "print('ok')",
                "test_cases": '[{"input":"1\\n2\\n","output":"3\\n"}]',
            },
            test_runner=test_runner,
        )

        self.assertIn("test_results", updated_state)
        self.assertEqual(updated_state["test_results"]["failed"], 1)
        self.assertEqual(
            updated_state["test_results"]["first_failed_report"],
            "Test case #1 failed. Reason: Output mismatch.",
        )
        self.assertEqual(test_runner.calls[0]["language"], "python")

    def test_run_python_static_analysis_step_writes_static_analysis(self) -> None:
        analyzer = StubStaticAnalyzer(_python_static_analysis_result())
        updated_state = run_python_static_analysis_step(
            state={
                "user_id": "7",
                "task_id": "task-a",
                "language": "python",
                "code": "assert True",
            },
            static_analyzer=analyzer,
        )

        self.assertIn("static_analysis", updated_state)
        self.assertEqual(updated_state["static_analysis"]["complexity_score"], 3.0)
        self.assertEqual(
            updated_state["static_analysis"]["security_warnings"][0]["test_id"],
            "B101",
        )
        self.assertIn(
            "Security warnings: 1",
            updated_state["static_analysis"]["security_warnings_summary"],
        )

    def test_run_python_static_analysis_step_skips_non_python_language(self) -> None:
        updated_state = run_python_static_analysis_step(
            state={
                "user_id": "7",
                "task_id": "task-a",
                "language": "go",
                "code": "package main",
            },
        )

        self.assertEqual(updated_state["static_analysis"]["language"], "go")
        self.assertEqual(
            updated_state["static_analysis"]["security_warnings_summary"],
            "Skipped: static analysis is only enabled for Python.",
        )

    def test_run_core_orchestration_steps_uses_single_agent_state(self) -> None:
        sandbox = StubSandboxExecutor(_sandbox_result())
        test_runner = StubTestRunner(_test_run_result())
        analyzer = StubStaticAnalyzer(_python_static_analysis_result())

        final_state = run_core_orchestration_steps(
            state={
                "user_id": "7",
                "task_id": "task-a",
                "language": "python",
                "code": "assert True",
                "test_cases": '[{"input":"1\\n2\\n","output":"3\\n"}]',
            },
            sandbox_executor=sandbox,
            test_runner=test_runner,
            static_analyzer=analyzer,
        )

        self.assertIn("execution_result", final_state)
        self.assertIn("metrics", final_state)
        self.assertIn("test_results", final_state)
        self.assertIn("static_analysis", final_state)
        self.assertEqual(final_state["user_id"], "7")
        self.assertEqual(final_state["task_id"], "task-a")


if __name__ == "__main__":
    unittest.main()
