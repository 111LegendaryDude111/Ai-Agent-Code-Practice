from __future__ import annotations

import unittest

from interview_orchestrator.sandbox_runner import SandboxExecutionResult, SandboxLimits
from interview_orchestrator.test_runner import (
    PredefinedTestCase,
    PredefinedTestRunner,
    format_first_failed_test_report,
    parse_test_cases_json,
    parse_test_cases_payload,
)


class StubSandboxExecutor:
    def __init__(self, results: list[SandboxExecutionResult]) -> None:
        self._results = list(results)
        self.calls: list[dict[str, object]] = []

    def execute(
        self,
        language: str,
        source_code: str,
        limits: SandboxLimits | None = None,
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

        if len(self._results) == 0:
            raise AssertionError("No stubbed sandbox results left.")
        return self._results.pop(0)


def _result(
    *,
    stdout: str,
    stderr: str = "",
    exit_code: int | None = 0,
    timed_out: bool = False,
    duration_seconds: float = 0.02,
    memory_usage_kb: int | None = 1024,
) -> SandboxExecutionResult:
    return SandboxExecutionResult(
        language="python",
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        memory_usage_kb=memory_usage_kb,
        timed_out=timed_out,
        duration_seconds=duration_seconds,
        container_name="test-container",
    )


class PredefinedTestRunnerTests(unittest.TestCase):
    def test_parse_test_cases_json_supports_output_aliases(self) -> None:
        test_cases = parse_test_cases_json(
            """
            [
              {"input":"1\\n2\\n","output":"3\\n"},
              {"input":"","expected_output":"0\\n","description":"edge-empty-input"}
            ]
            """
        )

        self.assertEqual(len(test_cases), 2)
        self.assertEqual(test_cases[0].input_data, "1\n2\n")
        self.assertEqual(test_cases[0].expected_output, "3\n")
        self.assertEqual(test_cases[1].description, "edge-empty-input")

    def test_parse_test_cases_payload_supports_wrapped_object(self) -> None:
        payload = {"test_cases": [{"input": "a", "output": "b"}]}
        test_cases = parse_test_cases_payload(payload)

        self.assertEqual(len(test_cases), 1)
        self.assertEqual(test_cases[0].input_data, "a")
        self.assertEqual(test_cases[0].expected_output, "b")

    def test_parse_test_cases_rejects_invalid_payloads(self) -> None:
        with self.assertRaises(ValueError):
            parse_test_cases_json("not-json")

        with self.assertRaises(ValueError):
            parse_test_cases_json("[]")

        with self.assertRaises(ValueError):
            parse_test_cases_payload({"unexpected": []})

        with self.assertRaises(ValueError):
            parse_test_cases_json('[{"input":"x"}]')

        with self.assertRaises(ValueError):
            parse_test_cases_json('[{"input":1,"output":"x"}]')

    def test_run_counts_pass_fail_correctly(self) -> None:
        stub = StubSandboxExecutor(
            [
                _result(stdout="3\n", exit_code=0),
                _result(stdout="not-four\n", exit_code=0),
                _result(stdout="ignored\n", exit_code=1, stderr="runtime error"),
            ]
        )
        runner = PredefinedTestRunner(sandbox_executor=stub)

        result = runner.run(
            language="python",
            source_code="print(input())",
            test_cases=[
                PredefinedTestCase(input_data="1\n2\n", expected_output="3"),
                PredefinedTestCase(input_data="2\n2\n", expected_output="4"),
                PredefinedTestCase(input_data="broken\n", expected_output="ok"),
            ],
        )

        self.assertEqual(result.total, 3)
        self.assertEqual(result.passed, 1)
        self.assertEqual(result.failed, 2)
        self.assertTrue(result.case_results[0].passed)
        self.assertFalse(result.case_results[1].passed)
        self.assertFalse(result.case_results[2].passed)
        self.assertIsNotNone(result.first_failed_case)
        if result.first_failed_case is None:
            self.fail("Expected first failed case.")
        self.assertEqual(result.first_failed_case.case_index, 1)
        self.assertEqual(result.first_failed_case.failure_reason, "Output mismatch.")
        self.assertIsNotNone(result.first_failed_case.output_diff)
        if result.first_failed_case.output_diff is None:
            self.fail("Expected output diff.")
        self.assertIn("--- expected", result.first_failed_case.output_diff)
        self.assertIn("+++ actual", result.first_failed_case.output_diff)
        self.assertIn("-4", result.first_failed_case.output_diff)
        self.assertIn("+not-four", result.first_failed_case.output_diff)
        self.assertIsNotNone(result.first_failed_report)
        if result.first_failed_report is None:
            self.fail("Expected first failed report text.")
        self.assertIn("Reason: Output mismatch.", result.first_failed_report)

    def test_run_handles_edge_case_whitespace_normalization(self) -> None:
        stub = StubSandboxExecutor(
            [
                _result(stdout="42   \r\n\r\n", exit_code=0),
                _result(stdout="\n", exit_code=0),
            ]
        )
        runner = PredefinedTestRunner(sandbox_executor=stub)

        result = runner.run(
            language="python",
            source_code="pass",
            test_cases=[
                PredefinedTestCase(input_data="", expected_output="42"),
                PredefinedTestCase(input_data="", expected_output=""),
            ],
        )

        self.assertEqual(result.passed, 2)
        self.assertEqual(result.failed, 0)
        self.assertIsNone(result.first_failed_case)
        self.assertIsNone(result.first_failed_report)

    def test_run_marks_timeout_as_failure(self) -> None:
        stub = StubSandboxExecutor(
            [
                _result(
                    stdout="partial",
                    stderr="timeout",
                    exit_code=None,
                    timed_out=True,
                    duration_seconds=0.5,
                )
            ]
        )
        runner = PredefinedTestRunner(sandbox_executor=stub)

        result = runner.run(
            language="python",
            source_code="while True: pass",
            test_cases=[PredefinedTestCase(input_data="", expected_output="done")],
        )

        self.assertEqual(result.passed, 0)
        self.assertEqual(result.failed, 1)
        self.assertTrue(result.case_results[0].timed_out)
        self.assertGreaterEqual(result.case_results[0].runtime_ms, 500)
        self.assertIsNotNone(result.first_failed_case)
        if result.first_failed_case is None:
            self.fail("Expected first failed case for timeout.")
        self.assertEqual(result.first_failed_case.failure_reason, "Execution timed out.")

    def test_first_failed_case_prefers_runtime_error_over_later_mismatch(self) -> None:
        stub = StubSandboxExecutor(
            [
                _result(stdout="", exit_code=2, stderr="Traceback"),
                _result(stdout="wrong", exit_code=0),
            ]
        )
        runner = PredefinedTestRunner(sandbox_executor=stub)

        result = runner.run(
            language="python",
            source_code="pass",
            test_cases=[
                PredefinedTestCase(input_data="", expected_output="ok", description="runtime"),
                PredefinedTestCase(input_data="", expected_output="expected", description="diff"),
            ],
        )

        self.assertEqual(result.failed, 2)
        self.assertIsNotNone(result.first_failed_case)
        if result.first_failed_case is None:
            self.fail("Expected first failed case.")
        self.assertEqual(result.first_failed_case.case_index, 0)
        self.assertEqual(result.first_failed_case.failure_reason, "Program exited with code 2.")
        self.assertIsNotNone(result.first_failed_report)
        if result.first_failed_report is None:
            self.fail("Expected first failure report.")
        self.assertIn("stderr:", result.first_failed_report)
        self.assertIn("Traceback", result.first_failed_report)

    def test_run_rejects_empty_test_case_list(self) -> None:
        runner = PredefinedTestRunner(sandbox_executor=StubSandboxExecutor([]))

        with self.assertRaises(ValueError):
            runner.run(language="python", source_code="pass", test_cases=[])

    def test_run_json_forwards_stdin_to_executor(self) -> None:
        stub = StubSandboxExecutor(
            [
                _result(stdout="ok\n"),
                _result(stdout="done\n"),
            ]
        )
        runner = PredefinedTestRunner(sandbox_executor=stub)

        result = runner.run_json(
            language="python",
            source_code="code",
            test_cases_json='[{"input":"A","output":"ok"},{"input":"B","output":"done"}]',
        )

        self.assertEqual(result.total, 2)
        self.assertEqual(stub.calls[0]["stdin_data"], "A")
        self.assertEqual(stub.calls[1]["stdin_data"], "B")

    def test_format_first_failed_test_report_returns_same_text(self) -> None:
        stub = StubSandboxExecutor([_result(stdout="bad", exit_code=0)])
        runner = PredefinedTestRunner(sandbox_executor=stub)
        result = runner.run(
            language="python",
            source_code="pass",
            test_cases=[PredefinedTestCase(input_data="", expected_output="good")],
        )

        formatted = format_first_failed_test_report(result)
        self.assertEqual(formatted, result.first_failed_report)
        self.assertIsNotNone(formatted)
        if formatted is None:
            self.fail("Expected formatted failure report.")
        self.assertIn("Test case #1 failed.", formatted)


if __name__ == "__main__":
    unittest.main()
