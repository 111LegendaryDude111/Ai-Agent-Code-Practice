from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

PYLINT_SCORE_PATTERN = re.compile(r"rated at (-?\d+(?:\.\d+)?)/10")


@dataclass(frozen=True)
class PylintWarning:
    message_id: str
    symbol: str
    message: str
    path: str
    line: int
    column: int


@dataclass(frozen=True)
class RadonComplexityBlock:
    name: str
    block_type: str
    complexity: int
    line: int
    endline: int | None


@dataclass(frozen=True)
class BanditWarning:
    test_id: str
    test_name: str
    issue_severity: str
    issue_confidence: str
    issue_text: str
    line_number: int


@dataclass(frozen=True)
class PythonStaticAnalysisResult:
    language: str
    pylint_score: float | None
    pylint_warnings: list[PylintWarning]
    complexity_score: float | None
    complexity_blocks: list[RadonComplexityBlock]
    security_warnings: list[BanditWarning]
    tool_errors: list[str]


@dataclass(frozen=True)
class _CommandResult:
    stdout: str
    stderr: str
    return_code: int | None
    invocation_error: str | None


class PythonStaticAnalyzer:
    def __init__(self, python_binary: str = "python3") -> None:
        self._python_binary = python_binary

    def analyze(self, source_code: str, filename: str = "main.py") -> PythonStaticAnalysisResult:
        with tempfile.TemporaryDirectory(prefix="python_static_analysis_") as workspace_dir:
            source_path = Path(workspace_dir) / filename
            source_path.write_text(source_code, encoding="utf-8")

            pylint_result = self._run_tool(
                [
                    self._python_binary,
                    "-m",
                    "pylint",
                    "--output-format=json2",
                    "--score=y",
                    str(source_path),
                ]
            )
            radon_result = self._run_tool(
                [
                    self._python_binary,
                    "-m",
                    "radon",
                    "cc",
                    "-j",
                    str(source_path),
                ]
            )
            bandit_result = self._run_tool(
                [
                    self._python_binary,
                    "-m",
                    "bandit",
                    "-q",
                    "-f",
                    "json",
                    str(source_path),
                ]
            )

        tool_errors: list[str] = []
        pylint_score, pylint_warnings = _parse_pylint_result(
            pylint_result,
            source_path=str(source_path),
            tool_errors=tool_errors,
        )
        complexity_score, complexity_blocks = _parse_radon_result(
            radon_result,
            source_path=str(source_path),
            tool_errors=tool_errors,
        )
        security_warnings = _parse_bandit_result(
            bandit_result,
            source_path=str(source_path),
            tool_errors=tool_errors,
        )

        return PythonStaticAnalysisResult(
            language="python",
            pylint_score=pylint_score,
            pylint_warnings=pylint_warnings,
            complexity_score=complexity_score,
            complexity_blocks=complexity_blocks,
            security_warnings=security_warnings,
            tool_errors=tool_errors,
        )

    def _run_tool(self, command: list[str]) -> _CommandResult:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
            return _CommandResult(
                stdout=completed.stdout,
                stderr=completed.stderr,
                return_code=completed.returncode,
                invocation_error=None,
            )
        except FileNotFoundError as exc:
            return _CommandResult(
                stdout="",
                stderr="",
                return_code=None,
                invocation_error=str(exc),
            )


def format_security_warnings(
    security_warnings: list[BanditWarning],
    max_items: int = 3,
) -> str:
    if len(security_warnings) == 0:
        return "No security warnings."

    bounded_max_items = max(1, max_items)
    lines = [f"Security warnings: {len(security_warnings)}"]
    for warning in security_warnings[:bounded_max_items]:
        lines.append(
            f"- {warning.test_id} [{warning.issue_severity}/{warning.issue_confidence}] "
            f"line {warning.line_number}: {warning.issue_text}"
        )

    remaining = len(security_warnings) - bounded_max_items
    if remaining > 0:
        lines.append(f"... and {remaining} more warning(s).")

    return "\n".join(lines)


def _parse_pylint_result(
    command_result: _CommandResult,
    source_path: str,
    tool_errors: list[str],
) -> tuple[float | None, list[PylintWarning]]:
    if command_result.invocation_error is not None:
        tool_errors.append(f"pylint failed to start: {command_result.invocation_error}")
        return None, []

    payload = _load_json_payload(command_result.stdout)
    if payload is None:
        if command_result.return_code not in (None, 0):
            tool_errors.append(
                _build_tool_parse_error("pylint", source_path, command_result.stderr)
            )
        return _extract_pylint_score(None, command_result), []

    raw_messages: object
    raw_score: object = None
    if isinstance(payload, dict):
        raw_messages = payload.get("messages", [])
        statistics = payload.get("statistics")
        if isinstance(statistics, dict):
            raw_score = statistics.get("score")
    elif isinstance(payload, list):
        raw_messages = payload
    else:
        tool_errors.append(_build_tool_parse_error("pylint", source_path, command_result.stderr))
        return _extract_pylint_score(None, command_result), []

    if not isinstance(raw_messages, list):
        tool_errors.append(_build_tool_parse_error("pylint", source_path, command_result.stderr))
        return _extract_pylint_score(raw_score, command_result), []

    warnings: list[PylintWarning] = []
    for entry in raw_messages:
        if not isinstance(entry, dict):
            continue
        warnings.append(
            PylintWarning(
                message_id=str(entry.get("messageId") or entry.get("message-id") or ""),
                symbol=str(entry.get("symbol") or ""),
                message=str(entry.get("message") or ""),
                path=str(entry.get("path") or ""),
                line=int(entry.get("line") or 0),
                column=int(entry.get("column") or 0),
            )
        )

    return _extract_pylint_score(raw_score, command_result), warnings


def _parse_radon_result(
    command_result: _CommandResult,
    source_path: str,
    tool_errors: list[str],
) -> tuple[float | None, list[RadonComplexityBlock]]:
    if command_result.invocation_error is not None:
        tool_errors.append(f"radon failed to start: {command_result.invocation_error}")
        return None, []

    payload = _load_json_payload(command_result.stdout)
    if not isinstance(payload, dict):
        if command_result.return_code not in (None, 0):
            tool_errors.append(_build_tool_parse_error("radon", source_path, command_result.stderr))
        return None, []

    raw_blocks: list[object] = []
    for value in payload.values():
        if isinstance(value, list):
            raw_blocks.extend(value)

    blocks: list[RadonComplexityBlock] = []
    for raw_block in raw_blocks:
        if not isinstance(raw_block, dict):
            continue
        blocks.append(
            RadonComplexityBlock(
                name=str(raw_block.get("name") or "<anonymous>"),
                block_type=str(raw_block.get("type") or "block"),
                complexity=int(raw_block.get("complexity") or 0),
                line=int(raw_block.get("lineno") or 0),
                endline=(
                    int(raw_block["endline"]) if raw_block.get("endline") is not None else None
                ),
            )
        )

    if len(blocks) == 0:
        return 0.0, []

    average_complexity = sum(block.complexity for block in blocks) / len(blocks)
    return round(average_complexity, 2), blocks


def _parse_bandit_result(
    command_result: _CommandResult,
    source_path: str,
    tool_errors: list[str],
) -> list[BanditWarning]:
    if command_result.invocation_error is not None:
        tool_errors.append(f"bandit failed to start: {command_result.invocation_error}")
        return []

    payload = _load_json_payload(command_result.stdout)
    if not isinstance(payload, dict):
        if command_result.return_code not in (None, 0, 1) or command_result.stderr.strip() != "":
            tool_errors.append(
                _build_tool_parse_error("bandit", source_path, command_result.stderr)
            )
        return []

    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        return []

    warnings: list[BanditWarning] = []
    for result in raw_results:
        if not isinstance(result, dict):
            continue
        warnings.append(
            BanditWarning(
                test_id=str(result.get("test_id") or ""),
                test_name=str(result.get("test_name") or ""),
                issue_severity=str(result.get("issue_severity") or ""),
                issue_confidence=str(result.get("issue_confidence") or ""),
                issue_text=str(result.get("issue_text") or ""),
                line_number=int(result.get("line_number") or 0),
            )
        )

    return warnings


def _load_json_payload(text: str) -> object | None:
    stripped = text.strip()
    if stripped == "":
        return None

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _extract_pylint_score(raw_score: object, command_result: _CommandResult) -> float | None:
    if isinstance(raw_score, int | float):
        return round(float(raw_score), 2)

    score_match = PYLINT_SCORE_PATTERN.search(command_result.stdout)
    if score_match is None:
        score_match = PYLINT_SCORE_PATTERN.search(command_result.stderr)
    if score_match is None:
        return None

    try:
        return round(float(score_match.group(1)), 2)
    except ValueError:
        return None


def _build_tool_parse_error(tool_name: str, source_path: str, stderr: str) -> str:
    error_hint = stderr.strip()
    if error_hint == "":
        error_hint = f"no parseable output for {source_path}"
    return f"{tool_name} failed: {error_hint}"
