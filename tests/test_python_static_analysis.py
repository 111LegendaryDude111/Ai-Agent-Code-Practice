from __future__ import annotations

import json
import subprocess
import unittest
from unittest.mock import Mock, patch

from interview_orchestrator.static_analysis import (
    PythonStaticAnalyzer,
    format_security_warnings,
)


class PythonStaticAnalyzerTests(unittest.TestCase):
    @patch("interview_orchestrator.static_analysis.subprocess.run")
    def test_analyze_parses_pylint_radon_and_bandit(
        self,
        subprocess_run_mock: Mock,
    ) -> None:
        pylint_stdout = json.dumps(
            {
                "messages": [
                    {
                        "messageId": "C0114",
                        "symbol": "missing-module-docstring",
                        "message": "Missing module docstring",
                        "path": "main.py",
                        "line": 1,
                        "column": 0,
                    }
                ],
                "statistics": {"score": 8.75},
            }
        )
        radon_stdout = json.dumps(
            {
                "main.py": [
                    {
                        "type": "function",
                        "name": "solve",
                        "lineno": 1,
                        "endline": 3,
                        "complexity": 4,
                    }
                ]
            }
        )
        bandit_stdout = json.dumps(
            {
                "results": [
                    {
                        "test_id": "B101",
                        "test_name": "assert_used",
                        "issue_severity": "LOW",
                        "issue_confidence": "HIGH",
                        "issue_text": "Use of assert detected.",
                        "line_number": 2,
                    }
                ]
            }
        )
        subprocess_run_mock.side_effect = [
            subprocess.CompletedProcess(
                args=["python3", "-m", "pylint"],
                returncode=16,
                stdout=pylint_stdout,
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["python3", "-m", "radon"],
                returncode=0,
                stdout=radon_stdout,
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["python3", "-m", "bandit"],
                returncode=1,
                stdout=bandit_stdout,
                stderr="",
            ),
        ]

        analyzer = PythonStaticAnalyzer()
        result = analyzer.analyze("def solve():\n    assert True\n")

        self.assertEqual(result.language, "python")
        self.assertEqual(result.pylint_score, 8.75)
        self.assertEqual(result.complexity_score, 4.0)
        self.assertEqual(len(result.pylint_warnings), 1)
        self.assertEqual(result.pylint_warnings[0].message_id, "C0114")
        self.assertEqual(len(result.complexity_blocks), 1)
        self.assertEqual(result.complexity_blocks[0].complexity, 4)
        self.assertEqual(len(result.security_warnings), 1)
        self.assertEqual(result.security_warnings[0].test_id, "B101")
        self.assertEqual(result.tool_errors, [])

        formatted = format_security_warnings(result.security_warnings)
        self.assertIn("Security warnings: 1", formatted)
        self.assertIn("B101", formatted)
        self.assertIn("Use of assert detected.", formatted)

    @patch("interview_orchestrator.static_analysis.subprocess.run")
    def test_analyze_returns_zero_complexity_for_empty_radon_blocks(
        self,
        subprocess_run_mock: Mock,
    ) -> None:
        subprocess_run_mock.side_effect = [
            subprocess.CompletedProcess(
                args=["python3", "-m", "pylint"],
                returncode=0,
                stdout=json.dumps({"messages": [], "statistics": {"score": 10.0}}),
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["python3", "-m", "radon"],
                returncode=0,
                stdout=json.dumps({"main.py": []}),
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["python3", "-m", "bandit"],
                returncode=0,
                stdout=json.dumps({"results": []}),
                stderr="",
            ),
        ]

        analyzer = PythonStaticAnalyzer()
        result = analyzer.analyze("print('ok')\n")

        self.assertEqual(result.complexity_score, 0.0)
        self.assertEqual(result.security_warnings, [])
        self.assertEqual(
            format_security_warnings(result.security_warnings),
            "No security warnings.",
        )

    @patch("interview_orchestrator.static_analysis.subprocess.run")
    def test_analyze_collects_tool_errors_when_outputs_are_not_parseable(
        self,
        subprocess_run_mock: Mock,
    ) -> None:
        subprocess_run_mock.side_effect = [
            subprocess.CompletedProcess(
                args=["python3", "-m", "pylint"],
                returncode=2,
                stdout="",
                stderr="No module named pylint",
            ),
            subprocess.CompletedProcess(
                args=["python3", "-m", "radon"],
                returncode=2,
                stdout="",
                stderr="No module named radon",
            ),
            subprocess.CompletedProcess(
                args=["python3", "-m", "bandit"],
                returncode=2,
                stdout="",
                stderr="No module named bandit",
            ),
        ]

        analyzer = PythonStaticAnalyzer()
        result = analyzer.analyze("print('ok')\n")

        self.assertIsNone(result.pylint_score)
        self.assertIsNone(result.complexity_score)
        self.assertEqual(result.security_warnings, [])
        self.assertEqual(len(result.tool_errors), 3)
        self.assertIn("pylint failed", result.tool_errors[0])
        self.assertIn("radon failed", result.tool_errors[1])
        self.assertIn("bandit failed", result.tool_errors[2])


if __name__ == "__main__":
    unittest.main()
