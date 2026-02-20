from __future__ import annotations

import unittest
from typing import cast
from unittest.mock import AsyncMock, patch

from interview_bot.main import (
    _TELEGRAM_MESSAGE_TARGET_LENGTH,
    _build_submission_agent_state,
    _build_submission_result_message,
    _process_saved_submission,
    _run_full_graph_cycle_for_submission,
)
from interview_bot.user_repository import UserRepository
from interview_orchestrator.agent_state import AgentState


class _StubRepository:
    def __init__(
        self,
        *,
        submission_id: int | None,
        skill_profile: dict[str, object] | None = None,
    ) -> None:
        self._submission_id = submission_id
        self._skill_profile = skill_profile
        self.save_calls: list[tuple[int, str, int | None]] = []
        self.metrics_calls: list[dict[str, object]] = []
        self.static_analysis_calls: list[dict[str, object]] = []
        self.orchestration_calls: list[dict[str, object]] = []

    async def save_submission_with_id(
        self,
        telegram_id: int,
        code: str,
        task_id: int | None = None,
    ) -> int | None:
        self.save_calls.append((telegram_id, code, task_id))
        return self._submission_id

    async def get_skill_profile(self, telegram_id: int) -> dict[str, object] | None:
        _ = telegram_id
        return self._skill_profile

    async def save_submission_metrics(
        self,
        submission_id: int,
        runtime_ms: int,
        memory_usage_kb: int | None,
        exit_code: int | None,
        stdout: str,
        stderr: str,
        timed_out: bool,
    ) -> bool:
        self.metrics_calls.append(
            {
                "submission_id": submission_id,
                "runtime_ms": runtime_ms,
                "memory_usage_kb": memory_usage_kb,
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
                "timed_out": timed_out,
            }
        )
        return True

    async def save_submission_static_analysis(
        self,
        submission_id: int,
        language: str,
        pylint_score: float | None,
        complexity_score: float | None,
        security_warnings: list[dict[str, object]],
        pylint_warnings: list[dict[str, object]],
    ) -> bool:
        self.static_analysis_calls.append(
            {
                "submission_id": submission_id,
                "language": language,
                "pylint_score": pylint_score,
                "complexity_score": complexity_score,
                "security_warnings": security_warnings,
                "pylint_warnings": pylint_warnings,
            }
        )
        return True

    async def save_submission_orchestration_artifacts(
        self,
        submission_id: int,
        llm_review: dict[str, object],
        score_breakdown: dict[str, object],
        final_score: float | None,
        branching: dict[str, object],
        recommended_difficulty: int | None,
    ) -> bool:
        self.orchestration_calls.append(
            {
                "submission_id": submission_id,
                "llm_review": llm_review,
                "score_breakdown": score_breakdown,
                "final_score": final_score,
                "branching": branching,
                "recommended_difficulty": recommended_difficulty,
            }
        )
        return True


class BotOrchestrationFlowTests(unittest.IsolatedAsyncioTestCase):
    def test_build_submission_agent_state_contains_required_fields(self) -> None:
        state = _build_submission_agent_state(
            telegram_user_id=42,
            submission_id=7,
            preferred_language="python",
            submission_text="print('ok')",
            skill_profile={"recent_scores": [88.0]},
        )

        self.assertEqual(state["user_id"], "42")
        self.assertEqual(state["task_id"], "submission-7")
        self.assertEqual(state["language"], "python")
        self.assertEqual(state["code"], "print('ok')")
        self.assertEqual(state["retry_count"], 0)
        self.assertIn("skill_profile", state)

    def test_build_submission_agent_state_normalizes_unknown_language(self) -> None:
        state = _build_submission_agent_state(
            telegram_user_id=9,
            submission_id=3,
            preferred_language="rust",
            submission_text="fn main() {}",
            skill_profile=None,
        )

        self.assertEqual(state["language"], "python")

    def test_build_submission_result_message_for_success_includes_summary(self) -> None:
        result_state = cast(
            AgentState,
            {
                "user_id": "42",
                "task_id": "submission-7",
                "language": "python",
                "code": "print('ok')",
                "final_score": 87.5,
                "llm_review": {"reviewer": "heuristic_rate_limited"},
                "branching": {
                    "retry": {
                        "should_retry": False,
                        "retries_used": 0,
                        "retries_remaining": 1,
                        "max_retries": 1,
                        "reason": "No failure detected.",
                    },
                    "adaptive_difficulty": {
                        "action": "keep",
                        "current_difficulty": 1,
                        "next_difficulty": 1,
                        "reason": "Difficulty remains unchanged.",
                    },
                    "hint": {
                        "should_show_hint": True,
                        "hints": ["Покройте граничные случаи входа."],
                    },
                    "next_node": "show_hint",
                },
            },
        )

        message = _build_submission_result_message(
            language_label="Python",
            submission_id=7,
            state=result_state,
            orchestration_error=None,
        )

        self.assertIn("Результат: решение прошло проверку.", message)
        self.assertIn("Итоговый score: 87.50/100.", message)
        self.assertIn("Следующее действие: изучите подсказки перед следующей попыткой.", message)
        self.assertIn("Рекомендуемая сложность: 1.", message)
        self.assertIn("Короткие подсказки:", message)
        self.assertIn("1. Покройте граничные случаи входа.", message)
        self.assertIn("Примечание: часть проверки выполнена в fallback-режиме.", message)

    def test_build_submission_result_message_for_mismatch_includes_case_context(self) -> None:
        result_state = cast(
            AgentState,
            {
                "user_id": "99",
                "task_id": "submission-99",
                "language": "python",
                "code": "print('bad')",
                "final_score": 42.0,
                "execution_result": {
                    "exit_code": 0,
                    "stdout": "2\n",
                    "stderr": "",
                    "timed_out": False,
                },
                "test_results": {
                    "total": 2,
                    "passed": 1,
                    "failed": 1,
                    "first_failed_report": "Test case #1 failed. Reason: Output mismatch.",
                    "case_results": [
                        {
                            "case_index": 0,
                            "passed": False,
                            "input_data": "1 1\n",
                            "expected_output": "2\n",
                            "actual_output": "3\n",
                            "stderr": "",
                            "exit_code": 0,
                            "timed_out": False,
                            "runtime_ms": 11,
                            "memory_usage_kb": 1024,
                            "description": "sum",
                            "output_diff": "--- expected\n+++ actual\n@@\n-2\n+3",
                            "failure_reason": "Output mismatch.",
                        }
                    ],
                },
                "llm_review": {
                    "improvement_suggestions": ["Сверьте формат вывода с условием."],
                    "reviewer": "heuristic",
                },
                "branching": {
                    "retry": {
                        "should_retry": True,
                        "retries_used": 1,
                        "retries_remaining": 0,
                        "max_retries": 1,
                        "reason": "Submission failed and retry budget is available.",
                    },
                    "adaptive_difficulty": {
                        "action": "decrease",
                        "current_difficulty": 2,
                        "next_difficulty": 1,
                        "reason": "Repeated failures detected.",
                    },
                    "hint": {
                        "should_show_hint": True,
                        "hints": ["Проверьте крайние случаи и перевод строки в конце ответа."],
                    },
                    "next_node": "retry_with_hint",
                },
                "recommended_difficulty": 1,
            },
        )

        message = _build_submission_result_message(
            language_label="Python",
            submission_id=99,
            state=result_state,
            orchestration_error=None,
        )

        self.assertIn("Результат: решение не прошло проверку.", message)
        self.assertIn("Причина: Вывод не совпадает с ожидаемым результатом.", message)
        self.assertIn("Первый упавший тест: #1.", message)
        self.assertIn("Ожидалось: 2", message)
        self.assertIn("Получено: 3", message)
        self.assertIn("Diff:", message)
        self.assertIn("--- expected", message)
        self.assertIn("Текущий score: 42.00/100.", message)
        self.assertIn(
            "Следующее действие: исправьте решение, учитывая подсказки, и отправьте заново.",
            message,
        )
        self.assertIn("Рекомендуемая сложность: 1.", message)
        self.assertIn("Сверьте формат вывода с условием.", message)
        self.assertLessEqual(len(message), _TELEGRAM_MESSAGE_TARGET_LENGTH)

    def test_build_submission_result_message_limits_output_for_telegram(self) -> None:
        oversized_payload = "x" * 10000
        result_state = cast(
            AgentState,
            {
                "user_id": "15",
                "task_id": "submission-15",
                "language": "python",
                "code": "print('bad')",
                "execution_result": {
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": oversized_payload,
                    "timed_out": False,
                },
                "test_results": {
                    "total": 1,
                    "passed": 0,
                    "failed": 1,
                    "first_failed_report": oversized_payload,
                    "case_results": [
                        {
                            "case_index": 0,
                            "passed": False,
                            "input_data": oversized_payload,
                            "expected_output": oversized_payload,
                            "actual_output": oversized_payload,
                            "stderr": oversized_payload,
                            "exit_code": 1,
                            "timed_out": False,
                            "runtime_ms": 100,
                            "memory_usage_kb": 1024,
                            "description": "too big",
                            "output_diff": oversized_payload,
                            "failure_reason": "Program exited with code 1.",
                        }
                    ],
                },
                "branching": {
                    "retry": {
                        "should_retry": True,
                        "retries_used": 1,
                        "retries_remaining": 0,
                        "max_retries": 1,
                        "reason": "Submission failed and retry budget is available.",
                    },
                    "adaptive_difficulty": {
                        "action": "decrease",
                        "current_difficulty": 1,
                        "next_difficulty": 1,
                        "reason": "Repeated failures detected.",
                    },
                    "hint": {
                        "should_show_hint": True,
                        "hints": [oversized_payload],
                    },
                    "next_node": "retry_submission",
                },
            },
        )

        message = _build_submission_result_message(
            language_label="Python",
            submission_id=15,
            state=result_state,
            orchestration_error=None,
        )

        self.assertLessEqual(len(message), _TELEGRAM_MESSAGE_TARGET_LENGTH)

    async def test_run_full_graph_cycle_for_submission_returns_fallback_on_exception(self) -> None:
        initial_state = _build_submission_agent_state(
            telegram_user_id=1,
            submission_id=10,
            preferred_language="python",
            submission_text="print('ok')",
            skill_profile=None,
        )

        with patch("interview_bot.main.run_full_graph_cycle", side_effect=RuntimeError("boom")):
            processed_state, orchestration_error = await _run_full_graph_cycle_for_submission(
                initial_state
            )

        self.assertIsNone(processed_state)
        self.assertEqual(orchestration_error, "RuntimeError: boom")

    async def test_process_saved_submission_runs_orchestration_after_save(self) -> None:
        repository = _StubRepository(
            submission_id=33,
            skill_profile={"recent_scores": [75.0]},
        )
        processed_state = cast(
            AgentState,
            {
                "user_id": "12",
                "task_id": "submission-33",
                "language": "python",
                "code": "print('ok')",
                "metrics": {
                    "runtime_ms": 120,
                    "memory_usage_kb": 2048,
                    "exit_code": 0,
                    "stdout": "ok\n",
                    "stderr": "",
                    "timed_out": False,
                },
                "static_analysis": {
                    "language": "python",
                    "pylint_score": 9.1,
                    "complexity_score": 2.0,
                    "security_warnings": [{"test_id": "B101"}],
                    "security_warnings_summary": "B101",
                    "pylint_warnings": [{"message_id": "C0114"}],
                    "tool_errors": [],
                },
                "llm_review": {
                    "summary": "ok",
                    "strengths": ["tests passed"],
                    "issues": [],
                    "improvement_suggestions": ["keep it up"],
                    "score_template": {
                        "correctness": 90.0,
                        "performance": 92.0,
                        "readability": 89.0,
                        "security": 94.0,
                        "final_score": 91.25,
                    },
                    "score": 91.25,
                    "reviewer": "heuristic",
                },
                "score_breakdown": {
                    "correctness_score": 90.0,
                    "performance_score": 92.0,
                    "readability_score": 89.0,
                    "security_score": 94.0,
                    "aggregated_score": 91.25,
                },
                "final_score": 91.25,
                "recommended_difficulty": 2,
                "branching": {
                    "retry": {
                        "should_retry": False,
                        "retries_used": 0,
                        "retries_remaining": 1,
                        "max_retries": 1,
                        "reason": "No failure detected.",
                    },
                    "adaptive_difficulty": {
                        "action": "increase",
                        "current_difficulty": 1,
                        "next_difficulty": 2,
                        "reason": "High score streak detected.",
                    },
                    "hint": {
                        "should_show_hint": False,
                        "hints": [],
                    },
                    "next_node": "complete",
                },
            },
        )

        with patch(
            "interview_bot.main._run_full_graph_cycle_for_submission",
            new=AsyncMock(return_value=(processed_state, None)),
        ) as orchestration_mock:
            is_saved, response_text = await _process_saved_submission(
                user_repository=cast(UserRepository, repository),
                telegram_user_id=12,
                preferred_language="python",
                submission_text="print('ok')",
            )

        self.assertTrue(is_saved)
        self.assertEqual(len(repository.save_calls), 1)
        self.assertEqual(repository.save_calls[0], (12, "print('ok')", None))
        self.assertEqual(orchestration_mock.await_count, 1)
        self.assertIn("Результат: решение прошло проверку.", response_text)
        self.assertIn("Итоговый score: 91.25/100.", response_text)
        self.assertIn("Следующее действие: можно переходить к следующей задаче.", response_text)
        self.assertIn("Рекомендуемая сложность: 2.", response_text)
        self.assertEqual(len(repository.metrics_calls), 1)
        self.assertEqual(repository.metrics_calls[0]["runtime_ms"], 120)
        self.assertEqual(len(repository.static_analysis_calls), 1)
        self.assertEqual(repository.static_analysis_calls[0]["language"], "python")
        self.assertEqual(len(repository.orchestration_calls), 1)
        self.assertEqual(repository.orchestration_calls[0]["final_score"], 91.25)
        self.assertEqual(repository.orchestration_calls[0]["recommended_difficulty"], 2)

    async def test_process_saved_submission_returns_fallback_message_when_orchestrator_fails(
        self,
    ) -> None:
        repository = _StubRepository(submission_id=5)
        with patch(
            "interview_bot.main._run_full_graph_cycle_for_submission",
            new=AsyncMock(return_value=(None, "RuntimeError: boom")),
        ):
            is_saved, response_text = await _process_saved_submission(
                user_repository=cast(UserRepository, repository),
                telegram_user_id=1,
                preferred_language="python",
                submission_text="print('ok')",
            )

        self.assertTrue(is_saved)
        self.assertIn("Проверка временно недоступна", response_text)
        self.assertIn("Причина: RuntimeError: boom", response_text)
        self.assertIn("Следующее действие: исправьте решение и отправьте заново.", response_text)
        self.assertEqual(len(repository.metrics_calls), 0)
        self.assertEqual(len(repository.static_analysis_calls), 0)
        self.assertEqual(len(repository.orchestration_calls), 0)

    async def test_process_saved_submission_returns_save_error_when_insert_failed(self) -> None:
        repository = _StubRepository(submission_id=None)
        is_saved, response_text = await _process_saved_submission(
            user_repository=cast(UserRepository, repository),
            telegram_user_id=1,
            preferred_language="python",
            submission_text="print('ok')",
        )

        self.assertFalse(is_saved)
        self.assertEqual(
            response_text,
            "Не удалось сохранить решение. Выберите язык и повторите попытку.",
        )


if __name__ == "__main__":
    unittest.main()
