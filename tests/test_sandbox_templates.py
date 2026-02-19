from __future__ import annotations

import unittest
from pathlib import Path


class SandboxTemplatesTests(unittest.TestCase):
    def test_sandbox_templates_exist_for_all_languages(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        expected_languages = ("python", "go", "java", "cpp")

        for language in expected_languages:
            dockerfile = project_root / "sandbox" / language / "Dockerfile"
            run_script = project_root / "sandbox" / language / "run.sh"
            self.assertTrue(dockerfile.exists(), f"Missing Dockerfile for {language}")
            self.assertTrue(run_script.exists(), f"Missing run.sh for {language}")

    def test_dockerfiles_use_non_root_user_and_entrypoint(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        expected_languages = ("python", "go", "java", "cpp")

        for language in expected_languages:
            dockerfile = (project_root / "sandbox" / language / "Dockerfile").read_text(
                encoding="utf-8"
            )
            self.assertIn("install -y --no-install-recommends time", dockerfile)
            self.assertIn("USER sandbox", dockerfile)
            self.assertIn('ENTRYPOINT ["/usr/local/bin/run.sh"]', dockerfile)

    def test_run_scripts_have_expected_defaults(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        expected_defaults = {
            "python": "/workspace/main.py",
            "go": "/workspace/main.go",
            "java": "/workspace/Main.java",
            "cpp": "/workspace/main.cpp",
        }

        for language, expected_path in expected_defaults.items():
            run_script = (project_root / "sandbox" / language / "run.sh").read_text(
                encoding="utf-8"
            )
            self.assertTrue(run_script.startswith("#!/usr/bin/env sh"))
            self.assertIn("set -eu", run_script)
            self.assertIn(expected_path, run_script)
            self.assertIn("__METRIC_MAX_RSS_KB__:%M", run_script)


if __name__ == "__main__":
    unittest.main()
