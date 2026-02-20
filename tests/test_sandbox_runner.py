from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from interview_orchestrator.sandbox_runner import DockerSandboxRunner, SandboxLimits


def _parse_json_log_record(record: str) -> dict[str, object]:
    parts = record.split(":", maxsplit=2)
    if len(parts) != 3:
        raise AssertionError(f"Unexpected log record format: {record!r}")
    return json.loads(parts[2])


class DockerSandboxRunnerTests(unittest.TestCase):
    def test_build_docker_command_has_required_security_flags(self) -> None:
        runner = DockerSandboxRunner()
        limits = SandboxLimits(
            cpu_limit=0.5,
            memory_limit_mb=128,
            timeout_seconds=2.0,
            pids_limit=32,
            tmpfs_size_mb=32,
        )

        command = runner._build_docker_command(
            language="python",
            workspace_dir="/tmp/sandbox-work",
            source_file_name="main.py",
            limits=limits,
            container_name="sandbox-test",
            main_class_name="Main",
        )

        self.assertEqual(command[0:2], ["docker", "run"])
        self.assertIn("--rm", command)
        self.assertIn("--network", command)
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertIn("--cpus", command)
        self.assertEqual(command[command.index("--cpus") + 1], "0.5")
        self.assertIn("--memory", command)
        self.assertEqual(command[command.index("--memory") + 1], "128m")
        self.assertIn("--pids-limit", command)
        self.assertEqual(command[command.index("--pids-limit") + 1], "32")
        self.assertIn("--read-only", command)
        self.assertIn("--tmpfs", command)
        self.assertIn("--cap-drop", command)
        self.assertEqual(command[command.index("--cap-drop") + 1], "ALL")
        self.assertNotIn("--privileged", command)

        mount_index = command.index("-v") + 1
        expected_mount = f"{Path('/tmp/sandbox-work').resolve()}:/workspace:ro"
        self.assertEqual(command[mount_index], expected_mount)

    @patch("interview_orchestrator.sandbox_runner.subprocess.run")
    def test_execute_success_returns_output_and_forces_cleanup(
        self,
        subprocess_run_mock: Mock,
    ) -> None:
        subprocess_run_mock.side_effect = [
            subprocess.CompletedProcess(
                args=["docker", "run"],
                returncode=0,
                stdout="ok\n",
                stderr="warn\n__METRIC_MAX_RSS_KB__:2048\n",
            ),
            subprocess.CompletedProcess(
                args=["docker", "rm", "-f", "container"],
                returncode=0,
                stdout="",
                stderr="",
            ),
        ]

        runner = DockerSandboxRunner()
        result = runner.execute(
            language="python",
            source_code="print('ok')",
            limits=SandboxLimits(timeout_seconds=1.0),
        )

        self.assertFalse(result.timed_out)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout, "ok\n")
        self.assertEqual(result.stderr, "warn\n")
        self.assertEqual(result.memory_usage_kb, 2048)

        run_command = subprocess_run_mock.call_args_list[0].args[0]
        cleanup_command = subprocess_run_mock.call_args_list[-1].args[0]
        container_name = run_command[run_command.index("--name") + 1]
        self.assertEqual(cleanup_command[0:3], ["docker", "rm", "-f"])
        self.assertEqual(cleanup_command[3], container_name)

    @patch("interview_orchestrator.sandbox_runner.subprocess.run")
    def test_execute_timeout_marks_result_and_forces_cleanup(
        self,
        subprocess_run_mock: Mock,
    ) -> None:
        subprocess_run_mock.side_effect = [
            subprocess.TimeoutExpired(
                cmd=["docker", "run"],
                timeout=0.01,
                output="partial stdout",
                stderr="partial stderr\n__METRIC_MAX_RSS_KB__:512\n",
            ),
            subprocess.CompletedProcess(
                args=["docker", "rm", "-f", "container"],
                returncode=0,
                stdout="",
                stderr="",
            ),
        ]

        runner = DockerSandboxRunner()
        result = runner.execute(
            language="python",
            source_code="while True:\n    pass\n",
            limits=SandboxLimits(timeout_seconds=0.01),
        )

        self.assertTrue(result.timed_out)
        self.assertIsNone(result.exit_code)
        self.assertIn("partial stdout", result.stdout)
        self.assertIn("timed out", result.stderr.lower())
        self.assertEqual(result.memory_usage_kb, 512)

        run_command = subprocess_run_mock.call_args_list[0].args[0]
        cleanup_command = subprocess_run_mock.call_args_list[-1].args[0]
        container_name = run_command[run_command.index("--name") + 1]
        self.assertEqual(cleanup_command[3], container_name)

    @patch("interview_orchestrator.sandbox_runner.subprocess.run")
    def test_execute_logs_sandbox_crash_when_docker_cli_is_missing(
        self,
        subprocess_run_mock: Mock,
    ) -> None:
        subprocess_run_mock.side_effect = [
            FileNotFoundError(),
            subprocess.CompletedProcess(
                args=["docker", "rm", "-f", "container"],
                returncode=0,
                stdout="",
                stderr="",
            ),
        ]

        runner = DockerSandboxRunner()
        with self.assertLogs("interview_orchestrator.sandbox", level="ERROR") as captured:
            result = runner.execute(
                language="python",
                source_code="print('ok')",
                limits=SandboxLimits(timeout_seconds=1.0),
            )

        self.assertIsNone(result.exit_code)
        self.assertEqual(result.stderr, "Docker CLI was not found in PATH.")
        payloads = [_parse_json_log_record(record) for record in captured.output]
        has_crash_event = any(
            payload.get("event") == "sandbox.execution.crashed" for payload in payloads
        )
        has_missing_docker_type = any(
            payload.get("crash_type") == "docker_cli_missing" for payload in payloads
        )
        self.assertTrue(has_crash_event)
        self.assertTrue(has_missing_docker_type)

    @patch("interview_orchestrator.sandbox_runner.subprocess.run")
    def test_execute_uses_single_read_only_mount_from_tempdir(
        self,
        subprocess_run_mock: Mock,
    ) -> None:
        subprocess_run_mock.side_effect = [
            subprocess.CompletedProcess(
                args=["docker", "run"],
                returncode=0,
                stdout="ok",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["docker", "rm", "-f", "container"],
                returncode=0,
                stdout="",
                stderr="",
            ),
        ]

        runner = DockerSandboxRunner()
        runner.execute(
            language="go",
            source_code="package main\nfunc main(){}",
            limits=SandboxLimits(timeout_seconds=1.0),
        )

        run_command = subprocess_run_mock.call_args_list[0].args[0]
        self.assertEqual(run_command.count("-v"), 1)
        mount = run_command[run_command.index("-v") + 1]
        self.assertTrue(mount.endswith(":/workspace:ro"))

        host_mount_path = mount.removesuffix(":/workspace:ro")
        resolved_mount_path = Path(host_mount_path).resolve()
        resolved_temp_root = Path(tempfile.gettempdir()).resolve()
        self.assertTrue(str(resolved_mount_path).startswith(str(resolved_temp_root)))

    def test_execute_rejects_unsupported_language(self) -> None:
        runner = DockerSandboxRunner()

        with self.assertRaises(ValueError):
            runner.execute(language="rust", source_code="fn main() {}")

    def test_limits_validation_rejects_non_positive_values(self) -> None:
        runner = DockerSandboxRunner()

        with self.assertRaises(ValueError):
            runner.execute(
                language="python",
                source_code="print('ok')",
                limits=SandboxLimits(cpu_limit=0),
            )


if __name__ == "__main__":
    unittest.main()
