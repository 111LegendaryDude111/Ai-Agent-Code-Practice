from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from interview_orchestrator.sandbox_runner import (
    DockerSandboxRunner,
    SandboxExecutionResult,
    SandboxLimits,
)


@dataclass(frozen=True)
class PredefinedTestCase:
    input_data: str
    expected_output: str
    description: str | None = None


@dataclass(frozen=True)
class PredefinedTestCaseResult:
    case_index: int
    passed: bool
    input_data: str
    expected_output: str
    actual_output: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    runtime_ms: int
    memory_usage_kb: int | None
    description: str | None = None
    output_diff: str | None = None
    failure_reason: str | None = None


@dataclass(frozen=True)
class PredefinedTestRunResult:
    total: int
    passed: int
    failed: int
    case_results: list[PredefinedTestCaseResult]
    first_failed_case: PredefinedTestCaseResult | None = None
    first_failed_report: str | None = None


class SandboxExecutor(Protocol):
    def execute(
        self,
        language: str,
        source_code: str,
        limits: SandboxLimits | None = None,
        main_class_name: str = "Main",
        stdin_data: str = "",
    ) -> SandboxExecutionResult:
        ...


class PredefinedTestRunner:
    def __init__(self, sandbox_executor: SandboxExecutor | None = None) -> None:
        self._sandbox_executor = sandbox_executor or DockerSandboxRunner()

    def run_json(
        self,
        language: str,
        source_code: str,
        test_cases_json: str,
        limits: SandboxLimits | None = None,
        main_class_name: str = "Main",
    ) -> PredefinedTestRunResult:
        test_cases = parse_test_cases_json(test_cases_json)
        return self.run(
            language=language,
            source_code=source_code,
            test_cases=test_cases,
            limits=limits,
            main_class_name=main_class_name,
        )

    def run(
        self,
        language: str,
        source_code: str,
        test_cases: Sequence[PredefinedTestCase],
        limits: SandboxLimits | None = None,
        main_class_name: str = "Main",
    ) -> PredefinedTestRunResult:
        if len(test_cases) == 0:
            raise ValueError("test_cases must not be empty.")

        case_results: list[PredefinedTestCaseResult] = []
        for case_index, test_case in enumerate(test_cases):
            execution = self._sandbox_executor.execute(
                language=language,
                source_code=source_code,
                limits=limits,
                main_class_name=main_class_name,
                stdin_data=test_case.input_data,
            )

            output_matches = _normalize_output(execution.stdout) == _normalize_output(
                test_case.expected_output
            )
            passed = output_matches and not execution.timed_out and execution.exit_code == 0
            runtime_ms = max(0, int(round(execution.duration_seconds * 1000)))
            output_diff = _build_output_diff(
                expected_output=test_case.expected_output,
                actual_output=execution.stdout,
            )
            failure_reason = _build_failure_reason(
                output_matches=output_matches,
                execution=execution,
            )

            case_results.append(
                PredefinedTestCaseResult(
                    case_index=case_index,
                    passed=passed,
                    input_data=test_case.input_data,
                    expected_output=test_case.expected_output,
                    actual_output=execution.stdout,
                    stderr=execution.stderr,
                    exit_code=execution.exit_code,
                    timed_out=execution.timed_out,
                    runtime_ms=runtime_ms,
                    memory_usage_kb=execution.memory_usage_kb,
                    description=test_case.description,
                    output_diff=output_diff,
                    failure_reason=None if passed else failure_reason,
                )
            )

        passed_count = sum(1 for result in case_results if result.passed)
        total = len(case_results)
        first_failed_case = next((result for result in case_results if not result.passed), None)
        return PredefinedTestRunResult(
            total=total,
            passed=passed_count,
            failed=total - passed_count,
            case_results=case_results,
            first_failed_case=first_failed_case,
            first_failed_report=format_case_failure_report(first_failed_case),
        )


def parse_test_cases_json(test_cases_json: str) -> list[PredefinedTestCase]:
    try:
        payload = json.loads(test_cases_json)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid test_cases JSON.") from exc
    return parse_test_cases_payload(payload)


def parse_test_cases_payload(payload: object) -> list[PredefinedTestCase]:
    raw_cases: object
    if isinstance(payload, dict):
        if "test_cases" not in payload:
            raise ValueError("JSON object must contain 'test_cases'.")
        raw_cases = payload["test_cases"]
    else:
        raw_cases = payload

    if not isinstance(raw_cases, list):
        raise ValueError("test_cases must be a list.")
    if len(raw_cases) == 0:
        raise ValueError("test_cases must not be empty.")

    parsed_cases: list[PredefinedTestCase] = []
    for case_index, raw_case in enumerate(raw_cases):
        parsed_cases.append(_parse_test_case(raw_case, case_index))
    return parsed_cases


def _parse_test_case(raw_case: object, case_index: int) -> PredefinedTestCase:
    if not isinstance(raw_case, dict):
        raise ValueError(f"test case at index {case_index} must be an object.")

    input_data = raw_case.get("input")
    if not isinstance(input_data, str):
        raise ValueError(f"test case at index {case_index} must contain string field 'input'.")

    expected_output = _extract_expected_output(raw_case, case_index)

    description_value = raw_case.get("description")
    if description_value is not None and not isinstance(description_value, str):
        raise ValueError(
            f"test case at index {case_index} field 'description' must be a string when present."
        )

    return PredefinedTestCase(
        input_data=input_data,
        expected_output=expected_output,
        description=description_value,
    )


def _extract_expected_output(raw_case: dict[str, Any], case_index: int) -> str:
    expected_output: object | None = raw_case.get("output")
    if expected_output is None:
        expected_output = raw_case.get("expected_output")

    if not isinstance(expected_output, str):
        raise ValueError(
            f"test case at index {case_index} must contain string field "
            "'output' (or alias 'expected_output')."
        )
    return expected_output


def _normalize_output(output: str) -> str:
    normalized_lines = output.replace("\r\n", "\n").split("\n")
    stripped_lines = [line.rstrip() for line in normalized_lines]

    while stripped_lines and stripped_lines[-1] == "":
        stripped_lines.pop()

    return "\n".join(stripped_lines)


def _build_output_diff(expected_output: str, actual_output: str) -> str | None:
    normalized_expected = _normalize_output(expected_output)
    normalized_actual = _normalize_output(actual_output)
    if normalized_expected == normalized_actual:
        return None

    diff_lines = list(
        difflib.unified_diff(
            normalized_expected.split("\n"),
            normalized_actual.split("\n"),
            fromfile="expected",
            tofile="actual",
            lineterm="",
        )
    )
    if len(diff_lines) == 0:
        return None
    return "\n".join(diff_lines)


def _build_failure_reason(
    output_matches: bool,
    execution: SandboxExecutionResult,
) -> str:
    if execution.timed_out:
        return "Execution timed out."
    if execution.exit_code is None:
        return "Execution failed before completion."
    if execution.exit_code != 0:
        return f"Program exited with code {execution.exit_code}."
    if not output_matches:
        return "Output mismatch."
    return "Unknown failure."


def format_case_failure_report(case_result: PredefinedTestCaseResult | None) -> str | None:
    if case_result is None:
        return None

    lines = [f"Test case #{case_result.case_index + 1} failed."]
    if case_result.description is not None and case_result.description != "":
        lines.append(f"Description: {case_result.description}")
    if case_result.failure_reason is not None:
        lines.append(f"Reason: {case_result.failure_reason}")

    lines.append("Input:")
    lines.append(case_result.input_data if case_result.input_data != "" else "<empty>")
    lines.append("Expected output:")
    lines.append(case_result.expected_output if case_result.expected_output != "" else "<empty>")
    lines.append("Actual output:")
    lines.append(case_result.actual_output if case_result.actual_output != "" else "<empty>")

    if case_result.output_diff is not None:
        lines.append("Diff:")
        lines.append(case_result.output_diff)
    if case_result.stderr != "":
        lines.append("stderr:")
        lines.append(case_result.stderr)

    if case_result.exit_code is not None:
        lines.append(f"Exit code: {case_result.exit_code}")
    lines.append(f"Runtime: {case_result.runtime_ms} ms")
    if case_result.memory_usage_kb is not None:
        lines.append(f"Memory: {case_result.memory_usage_kb} KB")

    return "\n".join(lines)


def format_first_failed_test_report(test_run_result: PredefinedTestRunResult) -> str | None:
    if test_run_result.first_failed_report is not None:
        return test_run_result.first_failed_report
    return format_case_failure_report(test_run_result.first_failed_case)
