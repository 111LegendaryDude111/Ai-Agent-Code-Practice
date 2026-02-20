from __future__ import annotations

import io
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from interview_bot.main import main as run_bot_bootstrap
from interview_common.settings import get_settings
from interview_orchestrator.main import main as run_orchestrator_bootstrap


class BootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_env = dict(os.environ)
        get_settings.cache_clear()

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._original_env)
        get_settings.cache_clear()

    def test_bot_bootstrap_requires_token(self) -> None:
        os.environ.pop("BOT_TOKEN", None)

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            run_bot_bootstrap()

        output = buffer.getvalue()
        self.assertIn("BOT_TOKEN is not set", output)

    def test_orchestrator_bootstrap_prints_selected_environment(self) -> None:
        os.environ["APP_ENV"] = "test"
        os.environ["LOG_LEVEL"] = "DEBUG"

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            run_orchestrator_bootstrap()

        output = buffer.getvalue()
        self.assertIn("env=test", output)
        self.assertIn("log_level=DEBUG", output)

    def test_settings_reads_values_from_dotenv(self) -> None:
        os.environ.pop("APP_ENV", None)
        os.environ.pop("LOG_LEVEL", None)
        os.environ.pop("BOT_TOKEN", None)

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            env_file = tmp_path / ".env"
            env_file.write_text(
                "APP_ENV=qa\n"
                "LOG_LEVEL=WARNING\n"
                "BOT_TOKEN=secret-token\n"
                "SUBMISSION_RATE_LIMIT_COUNT=7\n"
                "SUBMISSION_RATE_LIMIT_WINDOW_SECONDS=45\n"
                "LLM_RATE_LIMIT_COUNT=4\n"
                "LLM_RATE_LIMIT_WINDOW_SECONDS=90\n",
                encoding="utf-8",
            )

            current_workdir = Path.cwd()
            try:
                os.chdir(tmp_path)
                get_settings.cache_clear()
                settings = get_settings()
            finally:
                os.chdir(current_workdir)

        self.assertEqual(settings.app_env, "qa")
        self.assertEqual(settings.log_level, "WARNING")
        self.assertEqual(settings.bot_token, "secret-token")
        self.assertEqual(settings.submission_rate_limit_count, 7)
        self.assertEqual(settings.submission_rate_limit_window_seconds, 45)
        self.assertEqual(settings.llm_rate_limit_count, 4)
        self.assertEqual(settings.llm_rate_limit_window_seconds, 90)


if __name__ == "__main__":
    unittest.main()
