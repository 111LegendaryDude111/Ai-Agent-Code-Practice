from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from uuid import uuid4

MEMORY_USAGE_MARKER_PREFIX = "__METRIC_MAX_RSS_KB__:"


DEFAULT_IMAGE_BY_LANGUAGE: dict[str, str] = {
    "python": "interview-assistant/sandbox-python:local",
    "go": "interview-assistant/sandbox-go:local",
    "java": "interview-assistant/sandbox-java:local",
    "cpp": "interview-assistant/sandbox-cpp:local",
}

DEFAULT_SOURCE_FILE_BY_LANGUAGE: dict[str, str] = {
    "python": "main.py",
    "go": "main.go",
    "java": "Main.java",
    "cpp": "main.cpp",
}


@dataclass(frozen=True)
class SandboxLimits:
    cpu_limit: float = 1.0
    memory_limit_mb: int = 256
    timeout_seconds: float = 3.0
    pids_limit: int = 64
    tmpfs_size_mb: int = 64


@dataclass(frozen=True)
class SandboxExecutionResult:
    language: str
    exit_code: int | None
    stdout: str
    stderr: str
    memory_usage_kb: int | None
    timed_out: bool
    duration_seconds: float
    container_name: str


class DockerSandboxRunner:
    def __init__(
        self,
        image_by_language: Mapping[str, str] | None = None,
        docker_binary: str = "docker",
    ) -> None:
        self._docker_binary = docker_binary
        self._image_by_language = dict(DEFAULT_IMAGE_BY_LANGUAGE)
        if image_by_language is not None:
            self._image_by_language.update(image_by_language)

    def execute(
        self,
        language: str,
        source_code: str,
        limits: SandboxLimits | None = None,
        main_class_name: str = "Main",
        stdin_data: str = "",
    ) -> SandboxExecutionResult:
        if language not in DEFAULT_SOURCE_FILE_BY_LANGUAGE:
            raise ValueError(
                f"Unsupported language '{language}'. "
                f"Supported languages: {sorted(DEFAULT_SOURCE_FILE_BY_LANGUAGE)}."
            )

        normalized_limits = limits or SandboxLimits()
        _validate_limits(normalized_limits)

        source_file_name = DEFAULT_SOURCE_FILE_BY_LANGUAGE[language]
        container_name = f"interview-sandbox-{language}-{uuid4().hex[:12]}"

        with tempfile.TemporaryDirectory(prefix=f"sandbox_{language}_") as workspace_dir:
            source_path = Path(workspace_dir) / source_file_name
            source_path.write_text(source_code, encoding="utf-8")

            command = self._build_docker_command(
                language=language,
                workspace_dir=workspace_dir,
                source_file_name=source_file_name,
                limits=normalized_limits,
                container_name=container_name,
                main_class_name=main_class_name,
            )

            started_at = time.monotonic()
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    input=stdin_data,
                    timeout=normalized_limits.timeout_seconds,
                    check=False,
                )
                memory_usage_kb, sanitized_stderr = _extract_memory_usage_kb(completed.stderr)
                return SandboxExecutionResult(
                    language=language,
                    exit_code=completed.returncode,
                    stdout=completed.stdout,
                    stderr=sanitized_stderr,
                    memory_usage_kb=memory_usage_kb,
                    timed_out=False,
                    duration_seconds=time.monotonic() - started_at,
                    container_name=container_name,
                )
            except subprocess.TimeoutExpired as exc:
                timeout_message = (
                    f"Execution timed out after {normalized_limits.timeout_seconds:.2f}s."
                )
                raw_stderr = _coerce_stream(exc.stderr)
                memory_usage_kb, stderr = _extract_memory_usage_kb(raw_stderr)
                if stderr:
                    stderr = f"{stderr}\n{timeout_message}"
                else:
                    stderr = timeout_message

                return SandboxExecutionResult(
                    language=language,
                    exit_code=None,
                    stdout=_coerce_stream(exc.output),
                    stderr=stderr,
                    memory_usage_kb=memory_usage_kb,
                    timed_out=True,
                    duration_seconds=time.monotonic() - started_at,
                    container_name=container_name,
                )
            except FileNotFoundError:
                return SandboxExecutionResult(
                    language=language,
                    exit_code=None,
                    stdout="",
                    stderr="Docker CLI was not found in PATH.",
                    memory_usage_kb=None,
                    timed_out=False,
                    duration_seconds=time.monotonic() - started_at,
                    container_name=container_name,
                )
            finally:
                self._cleanup_container(container_name)

    def _build_docker_command(
        self,
        language: str,
        workspace_dir: str,
        source_file_name: str,
        limits: SandboxLimits,
        container_name: str,
        main_class_name: str,
    ) -> list[str]:
        image = self._image_by_language.get(language)
        if image is None:
            raise ValueError(
                f"Container image is not configured for language '{language}'."
            )

        mount = f"{Path(workspace_dir).resolve()}:/workspace:ro"
        command: list[str] = [
            self._docker_binary,
            "run",
            "--name",
            container_name,
            "--rm",
            "--network",
            "none",
            "--cpus",
            str(limits.cpu_limit),
            "--memory",
            f"{limits.memory_limit_mb}m",
            "--pids-limit",
            str(limits.pids_limit),
            "--read-only",
            "--tmpfs",
            f"/tmp:rw,noexec,nosuid,size={limits.tmpfs_size_mb}m",
            "--cap-drop",
            "ALL",
            "-v",
            mount,
            image,
            f"/workspace/{source_file_name}",
        ]

        if language == "java":
            command.append(main_class_name)

        return command

    def _cleanup_container(self, container_name: str) -> None:
        try:
            subprocess.run(
                [self._docker_binary, "rm", "-f", container_name],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return


def _coerce_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _extract_memory_usage_kb(stderr: str) -> tuple[int | None, str]:
    if stderr == "":
        return None, ""

    memory_usage_kb: int | None = None
    kept_lines: list[str] = []
    for line in stderr.splitlines():
        if line.startswith(MEMORY_USAGE_MARKER_PREFIX):
            value = line.removeprefix(MEMORY_USAGE_MARKER_PREFIX).strip()
            if value.isdigit():
                memory_usage_kb = int(value)
            continue
        kept_lines.append(line)

    normalized_stderr = "\n".join(kept_lines)
    if stderr.endswith("\n") and normalized_stderr != "":
        normalized_stderr = f"{normalized_stderr}\n"
    return memory_usage_kb, normalized_stderr


def _validate_limits(limits: SandboxLimits) -> None:
    if limits.cpu_limit <= 0:
        raise ValueError("cpu_limit must be > 0.")
    if limits.memory_limit_mb <= 0:
        raise ValueError("memory_limit_mb must be > 0.")
    if limits.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0.")
    if limits.pids_limit <= 0:
        raise ValueError("pids_limit must be > 0.")
    if limits.tmpfs_size_mb <= 0:
        raise ValueError("tmpfs_size_mb must be > 0.")
