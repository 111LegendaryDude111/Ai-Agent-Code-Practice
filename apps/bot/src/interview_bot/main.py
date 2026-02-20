from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any

from interview_common import get_settings
from interview_orchestrator.agent_state import AgentState, validate_agent_state
from interview_orchestrator.state_steps import run_full_graph_cycle

from interview_bot.user_repository import UserRepository, build_user_repository

LANG_CALLBACK_PREFIX = "lang:"
SUPPORTED_LANGUAGES: tuple[tuple[str, str], ...] = (
    ("python", "Python"),
    ("go", "Go"),
    ("java", "Java"),
    ("cpp", "C++"),
)
LANGUAGE_LABELS = dict(SUPPORTED_LANGUAGES)
MAX_SUBMISSION_LENGTH = 4000
_ORCHESTRATION_LOGGER = logging.getLogger("interview_bot.orchestration")
_ORCHESTRATION_NEXT_STEP_LABELS: dict[str, str] = {
    "complete": "можно переходить к следующей задаче.",
    "retry_submission": "исправьте решение и отправьте заново.",
    "retry_with_hint": "исправьте решение, учитывая подсказки, и отправьте заново.",
    "show_hint": "изучите подсказки перед следующей попыткой.",
}
_TELEGRAM_MESSAGE_TARGET_LENGTH = 3800
_TELEGRAM_SNIPPET_MAX_LENGTH = 220


def _import_aiogram() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        from aiogram import Bot, Dispatcher, F, Router
        from aiogram.filters import CommandStart
        from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
    except ImportError as exc:
        raise RuntimeError(
            "aiogram is not installed. Run `make install` to install bot dependencies."
        ) from exc

    return Bot, Dispatcher, Router, CommandStart, F, InlineKeyboardButton, InlineKeyboardMarkup


def _language_keyboard() -> Any:
    _, _, _, _, _, InlineKeyboardButton, InlineKeyboardMarkup = _import_aiogram()
    buttons = [
        [
            InlineKeyboardButton(
                text=language_label,
                callback_data=f"{LANG_CALLBACK_PREFIX}{language_code}",
            )
        ]
        for language_code, language_label in SUPPORTED_LANGUAGES
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def _validate_submission_text(text: str | None) -> tuple[str | None, str | None]:
    if text is None:
        return None, "Отправьте код в текстовом сообщении."

    normalized_text = text.strip()
    if normalized_text == "":
        return (
            None,
            "Пустой ввод. Отправьте непустой фрагмент кода.",
        )

    if len(normalized_text) > MAX_SUBMISSION_LENGTH:
        return (
            None,
            f"Слишком большой фрагмент кода. Максимум {MAX_SUBMISSION_LENGTH} символов.",
        )

    return normalized_text, None


def _build_submission_agent_state(
    *,
    telegram_user_id: int,
    submission_id: int,
    preferred_language: str,
    submission_text: str,
    skill_profile: dict[str, object] | None,
) -> AgentState:
    normalized_language = (
        preferred_language if preferred_language in LANGUAGE_LABELS else SUPPORTED_LANGUAGES[0][0]
    )
    raw_state: dict[str, object] = {
        "user_id": str(telegram_user_id),
        "task_id": f"submission-{submission_id}",
        "language": normalized_language,
        "code": submission_text,
        "retry_count": 0,
    }
    if isinstance(skill_profile, dict):
        raw_state["skill_profile"] = skill_profile
    return validate_agent_state(raw_state)


async def _run_full_graph_cycle_for_submission(
    state: AgentState,
) -> tuple[AgentState | None, str | None]:
    try:
        result_state = await asyncio.to_thread(run_full_graph_cycle, state)
    except Exception as error:  # noqa: BLE001 - bot must gracefully recover
        _ORCHESTRATION_LOGGER.exception(
            "submission.orchestration.failed",
            extra={
                "user_id": state["user_id"],
                "task_id": state["task_id"],
                "language": state["language"],
            },
        )
        return None, f"{type(error).__name__}: {error}"

    return result_state, None


def _extract_final_score(state: Mapping[str, object]) -> float:
    final_score = state.get("final_score")
    if isinstance(final_score, int | float) and not isinstance(final_score, bool):
        return round(float(final_score), 2)

    llm_review = state.get("llm_review")
    if isinstance(llm_review, Mapping):
        review_score = llm_review.get("score")
        if isinstance(review_score, int | float) and not isinstance(review_score, bool):
            return round(float(review_score), 2)

    return 0.0


def _extract_next_step(state: Mapping[str, object]) -> str:
    branching = state.get("branching")
    if isinstance(branching, Mapping):
        next_node = branching.get("next_node")
        if isinstance(next_node, str) and next_node != "":
            return next_node
    return "retry_submission"


def _extract_recommended_difficulty(state: Mapping[str, object]) -> int | None:
    recommended_difficulty = _extract_int_value(state.get("recommended_difficulty"))
    if recommended_difficulty is not None and recommended_difficulty >= 1:
        return recommended_difficulty

    branching = state.get("branching")
    if not isinstance(branching, Mapping):
        return None

    adaptive_difficulty = branching.get("adaptive_difficulty")
    if not isinstance(adaptive_difficulty, Mapping):
        return None

    next_difficulty = _extract_int_value(adaptive_difficulty.get("next_difficulty"))
    if next_difficulty is not None and next_difficulty >= 1:
        return next_difficulty
    return None


def _extract_actionable_hints(
    state: Mapping[str, object],
    *,
    max_hints: int = 2,
) -> list[str]:
    if max_hints <= 0:
        return []

    hints: list[str] = []
    seen: set[str] = set()

    branching = state.get("branching")
    if isinstance(branching, Mapping):
        hint_state = branching.get("hint")
        if isinstance(hint_state, Mapping):
            branch_hints = hint_state.get("hints")
            if isinstance(branch_hints, list):
                for branch_hint in branch_hints:
                    _append_actionable_hint(
                        hints,
                        seen,
                        branch_hint,
                        max_hints=max_hints,
                    )

    llm_review = state.get("llm_review")
    if isinstance(llm_review, Mapping):
        improvement_suggestions = llm_review.get("improvement_suggestions")
        if isinstance(improvement_suggestions, list):
            for suggestion in improvement_suggestions:
                _append_actionable_hint(
                    hints,
                    seen,
                    suggestion,
                    max_hints=max_hints,
                )

    return hints


def _append_actionable_hint(
    hints: list[str],
    seen: set[str],
    value: object,
    *,
    max_hints: int,
) -> None:
    if len(hints) >= max_hints or not isinstance(value, str):
        return

    normalized = value.strip()
    if normalized == "" or normalized in seen:
        return

    seen.add(normalized)
    hints.append(_truncate_text(normalized, max_length=_TELEGRAM_SNIPPET_MAX_LENGTH))


def _extract_test_results(state: Mapping[str, object]) -> Mapping[str, object] | None:
    test_results = state.get("test_results")
    if isinstance(test_results, Mapping):
        return test_results
    return None


def _extract_first_failed_case(state: Mapping[str, object]) -> Mapping[str, object] | None:
    test_results = _extract_test_results(state)
    if test_results is None:
        return None

    case_results = test_results.get("case_results")
    if not isinstance(case_results, list):
        return None

    for case_result in case_results:
        if not isinstance(case_result, Mapping):
            continue
        if _extract_bool_value(case_result.get("passed")) is False:
            return case_result

    return None


def _extract_first_failed_report(state: Mapping[str, object]) -> str | None:
    test_results = _extract_test_results(state)
    if test_results is None:
        return None

    first_failed_report = test_results.get("first_failed_report")
    if isinstance(first_failed_report, str) and first_failed_report.strip() != "":
        return first_failed_report.strip()
    return None


def _extract_submission_stderr(
    state: Mapping[str, object],
    *,
    first_failed_case: Mapping[str, object] | None,
) -> str | None:
    candidates: list[object] = []
    if first_failed_case is not None:
        candidates.append(first_failed_case.get("stderr"))

    execution_result = state.get("execution_result")
    if isinstance(execution_result, Mapping):
        candidates.append(execution_result.get("stderr"))

    metrics = state.get("metrics")
    if isinstance(metrics, Mapping):
        candidates.append(metrics.get("stderr"))

    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        for raw_line in candidate.splitlines():
            line = raw_line.strip()
            if line != "":
                return line

    return None


def _extract_exit_code(
    state: Mapping[str, object],
    *,
    first_failed_case: Mapping[str, object] | None,
) -> int | None:
    if first_failed_case is not None:
        case_exit_code = _extract_int_value(first_failed_case.get("exit_code"))
        if case_exit_code is not None:
            return case_exit_code

    execution_result = state.get("execution_result")
    if not isinstance(execution_result, Mapping):
        return None
    return _extract_int_value(execution_result.get("exit_code"))


def _has_submission_failure(state: Mapping[str, object]) -> bool:
    test_results = _extract_test_results(state)
    if test_results is not None:
        failed = _extract_int_value(test_results.get("failed"))
        if failed is not None and failed > 0:
            return True

    execution_result = state.get("execution_result")
    if isinstance(execution_result, Mapping):
        if _extract_bool_value(execution_result.get("timed_out")) is True:
            return True

        exit_code = _extract_int_value(execution_result.get("exit_code"))
        if exit_code is not None and exit_code != 0:
            return True

    return False


def _translate_failure_reason(reason: str) -> tuple[str, str]:
    normalized_reason = reason.strip()
    lowered_reason = normalized_reason.lower()

    if "output mismatch" in lowered_reason:
        return "output_mismatch", "Вывод не совпадает с ожидаемым результатом."
    if "timed out" in lowered_reason:
        return "timeout", "Превышен лимит времени выполнения (timeout)."
    if "failed before completion" in lowered_reason:
        return "runtime_error", "Исполнение прервалось до завершения в sandbox."
    if "program exited with code" in lowered_reason:
        exit_code_token = lowered_reason.removeprefix("program exited with code").strip(" .")
        if exit_code_token.lstrip("-").isdigit():
            exit_code = int(exit_code_token)
            if exit_code < 0:
                return "runtime_error", f"Процесс завершился сигналом {abs(exit_code)}."
            return "runtime_error", f"Программа завершилась с кодом {exit_code}."
        return "runtime_error", "Программа завершилась с ошибкой выполнения."

    return "runtime_error", _truncate_text(normalized_reason, max_length=180)


def _classify_failure(
    *,
    state: Mapping[str, object],
    first_failed_case: Mapping[str, object] | None,
) -> tuple[str, str]:
    if first_failed_case is not None:
        if _extract_bool_value(first_failed_case.get("timed_out")) is True:
            return "timeout", "Превышен лимит времени выполнения (timeout)."

        failure_reason = first_failed_case.get("failure_reason")
        if isinstance(failure_reason, str) and failure_reason.strip() != "":
            return _translate_failure_reason(failure_reason)

        output_diff = first_failed_case.get("output_diff")
        if isinstance(output_diff, str) and output_diff.strip() != "":
            return "output_mismatch", "Вывод не совпадает с ожидаемым результатом."

    execution_result = state.get("execution_result")
    if isinstance(execution_result, Mapping):
        if _extract_bool_value(execution_result.get("timed_out")) is True:
            return "timeout", "Превышен лимит времени выполнения (timeout)."

        exit_code = _extract_int_value(execution_result.get("exit_code"))
        if exit_code is not None and exit_code != 0:
            if exit_code < 0:
                return "runtime_error", f"Процесс завершился сигналом {abs(exit_code)}."
            return "runtime_error", f"Программа завершилась с кодом {exit_code}."

    return "failure", "Решение не прошло проверку."


def _truncate_text(value: str, *, max_length: int) -> str:
    if max_length < 4:
        return value[:max_length]

    if len(value) <= max_length:
        return value
    return f"{value[: max_length - 3]}..."


def _truncate_message(value: str, *, max_length: int) -> str:
    if max_length < 4:
        return value[:max_length]

    if len(value) <= max_length:
        return value
    return f"{value[: max_length - 3]}..."


def _format_output_snippet(value: object, *, max_length: int) -> str:
    if not isinstance(value, str):
        return "<empty>"

    normalized = value.replace("\r\n", "\n").strip("\n ")
    if normalized == "":
        return "<empty>"
    single_line = normalized.replace("\n", "\\n")
    return _truncate_text(single_line, max_length=max_length)


def _limit_message(lines: list[str]) -> str:
    non_empty_lines = [line for line in lines if line.strip() != ""]
    if len(non_empty_lines) == 0:
        return ""

    message = "\n".join(non_empty_lines)
    if len(message) <= _TELEGRAM_MESSAGE_TARGET_LENGTH:
        return message

    kept_lines: list[str] = []
    for line in non_empty_lines:
        candidate_message = "\n".join([*kept_lines, line])
        if len(candidate_message) > _TELEGRAM_MESSAGE_TARGET_LENGTH:
            break
        kept_lines.append(line)

    if len(kept_lines) == 0:
        return _truncate_message(message, max_length=_TELEGRAM_MESSAGE_TARGET_LENGTH)

    truncated_message = "\n".join(kept_lines)
    suffix = "…(сообщение сокращено)"
    if len(truncated_message) + 1 + len(suffix) <= _TELEGRAM_MESSAGE_TARGET_LENGTH:
        return f"{truncated_message}\n{suffix}"
    return _truncate_message(truncated_message, max_length=_TELEGRAM_MESSAGE_TARGET_LENGTH)


def _build_failure_context_lines(
    *,
    state: Mapping[str, object],
    first_failed_case: Mapping[str, object] | None,
    failure_kind: str,
) -> list[str]:
    lines: list[str] = []

    first_failed_report = _extract_first_failed_report(state)
    if first_failed_case is not None:
        case_index = _extract_int_value(first_failed_case.get("case_index"))
        if case_index is not None and case_index >= 0:
            lines.append(f"Первый упавший тест: #{case_index + 1}.")

        if failure_kind == "output_mismatch":
            expected_output = _format_output_snippet(
                first_failed_case.get("expected_output"),
                max_length=180,
            )
            actual_output = _format_output_snippet(
                first_failed_case.get("actual_output"),
                max_length=180,
            )
            lines.append("Ожидалось: " f"{expected_output}")
            lines.append("Получено: " f"{actual_output}")

            output_diff = first_failed_case.get("output_diff")
            if isinstance(output_diff, str) and output_diff.strip() != "":
                lines.append("Diff:")
                lines.append(_truncate_message(output_diff.strip(), max_length=700))

        input_data = _format_output_snippet(first_failed_case.get("input_data"), max_length=160)
        if input_data != "<empty>":
            lines.append(f"Вход первого упавшего теста: {input_data}")
    elif first_failed_report is not None:
        summarized_report = first_failed_report.replace("\n", " | ")
        lines.append("Первый упавший тест: " f"{_truncate_text(summarized_report, max_length=350)}")

    if first_failed_report is not None and first_failed_case is not None:
        summarized_report = first_failed_report.replace("\n", " | ")
        lines.append("Краткий отчет: " f"{_truncate_text(summarized_report, max_length=350)}")

    exit_code = _extract_exit_code(state, first_failed_case=first_failed_case)
    if exit_code is not None and exit_code != 0:
        if exit_code < 0:
            lines.append(f"Сигнал завершения: {abs(exit_code)}.")
        else:
            lines.append(f"Код завершения: {exit_code}.")

    stderr_excerpt = _extract_submission_stderr(state, first_failed_case=first_failed_case)
    if stderr_excerpt is not None:
        lines.append(
            f"stderr: {_truncate_text(stderr_excerpt, max_length=_TELEGRAM_SNIPPET_MAX_LENGTH)}"
        )

    return lines


def _build_processing_status(
    state: Mapping[str, object] | None,
    orchestration_error: str | None,
) -> str:
    if orchestration_error is not None or state is None:
        return "orchestrator_fallback"

    llm_review = state.get("llm_review")
    if isinstance(llm_review, Mapping):
        reviewer = llm_review.get("reviewer")
        if reviewer in {"heuristic_fallback", "heuristic_rate_limited"}:
            return "completed_with_fallback"

    execution_result = state.get("execution_result")
    if isinstance(execution_result, Mapping):
        stderr = execution_result.get("stderr")
        if isinstance(stderr, str) and "Sandbox execution failed before completion." in stderr:
            return "completed_with_recovery"

    return "completed"


def _build_submission_result_message(
    *,
    language_label: str,
    submission_id: int,
    state: Mapping[str, object] | None,
    orchestration_error: str | None,
) -> str:
    result_state = state if state is not None else {}
    status = _build_processing_status(state, orchestration_error)
    score = _extract_final_score(result_state)
    next_step = _extract_next_step(result_state)
    next_step_label = _ORCHESTRATION_NEXT_STEP_LABELS.get(
        next_step,
        "повторите отправку после проверки.",
    )

    lines = [
        f"Код сохранен (ID: {submission_id}). Текущий язык: {language_label}.",
    ]

    if orchestration_error is not None or state is None:
        lines.append("Проверка временно недоступна, сохранен только submission.")
        if orchestration_error is not None:
            lines.append(
                "Причина: "
                f"{_truncate_text(orchestration_error, max_length=_TELEGRAM_SNIPPET_MAX_LENGTH)}"
            )
        lines.append(
            "Следующее действие: " f"{_ORCHESTRATION_NEXT_STEP_LABELS['retry_submission']}"
        )
        lines.append("Примечание: оркестратор перешел в детерминированный fallback.")
        return _limit_message(lines)

    has_failure = _has_submission_failure(result_state)
    first_failed_case = _extract_first_failed_case(result_state)
    if has_failure:
        failure_kind, failure_reason = _classify_failure(
            state=result_state,
            first_failed_case=first_failed_case,
        )
        lines.append("Результат: решение не прошло проверку.")
        lines.append(f"Причина: {failure_reason}")
        lines.extend(
            _build_failure_context_lines(
                state=result_state,
                first_failed_case=first_failed_case,
                failure_kind=failure_kind,
            )
        )
        lines.append(f"Текущий score: {score:.2f}/100.")
    else:
        lines.append("Результат: решение прошло проверку.")
        lines.append(f"Итоговый score: {score:.2f}/100.")

    lines.append(f"Следующее действие: {next_step_label}")

    recommended_difficulty = _extract_recommended_difficulty(result_state)
    if recommended_difficulty is not None:
        lines.append(f"Рекомендуемая сложность: {recommended_difficulty}.")

    hints = _extract_actionable_hints(result_state, max_hints=2)
    if len(hints) > 0:
        lines.append("Короткие подсказки:")
        for hint_index, hint in enumerate(hints, start=1):
            lines.append(f"{hint_index}. {hint}")

    if status == "completed_with_fallback":
        lines.append("Примечание: часть проверки выполнена в fallback-режиме.")
    elif status == "completed_with_recovery":
        lines.append("Примечание: проверка завершена после recovery в sandbox.")

    return _limit_message(lines)


def _normalize_json_value(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize_json_value(item) for item in value]
    return str(value)


def _normalize_json_object(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _normalize_json_value(item) for key, item in value.items()}


def _normalize_json_object_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []

    items: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        items.append(_normalize_json_object(item))
    return items


def _extract_int_value(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _extract_float_value(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return None


def _extract_bool_value(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    return None


async def _persist_submission_metrics(
    *,
    user_repository: UserRepository,
    submission_id: int,
    state: Mapping[str, object],
) -> None:
    metrics = state.get("metrics")
    metrics_state = metrics if isinstance(metrics, Mapping) else {}
    execution_result = state.get("execution_result")
    execution_result_state = execution_result if isinstance(execution_result, Mapping) else {}

    runtime_ms = _extract_int_value(metrics_state.get("runtime_ms"))
    if runtime_ms is None or runtime_ms < 0:
        runtime_ms = 0

    memory_usage_kb = _extract_int_value(metrics_state.get("memory_usage_kb"))
    if memory_usage_kb is not None and memory_usage_kb < 0:
        memory_usage_kb = None

    exit_code = _extract_int_value(metrics_state.get("exit_code"))
    if exit_code is None:
        exit_code = _extract_int_value(execution_result_state.get("exit_code"))

    stdout_raw = metrics_state.get("stdout")
    if not isinstance(stdout_raw, str):
        stdout_raw = execution_result_state.get("stdout")
    stdout = stdout_raw if isinstance(stdout_raw, str) else ""

    stderr_raw = metrics_state.get("stderr")
    if not isinstance(stderr_raw, str):
        stderr_raw = execution_result_state.get("stderr")
    stderr = stderr_raw if isinstance(stderr_raw, str) else ""

    timed_out = _extract_bool_value(metrics_state.get("timed_out"))
    if timed_out is None:
        timed_out = _extract_bool_value(execution_result_state.get("timed_out"))
    if timed_out is None:
        timed_out = False

    try:
        is_saved = await user_repository.save_submission_metrics(
            submission_id=submission_id,
            runtime_ms=runtime_ms,
            memory_usage_kb=memory_usage_kb,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
        )
        if not is_saved:
            _ORCHESTRATION_LOGGER.warning(
                "submission.persistence.metrics.not_saved",
                extra={"submission_id": submission_id},
            )
    except Exception:  # noqa: BLE001 - persistence should not crash bot flow
        _ORCHESTRATION_LOGGER.exception(
            "submission.persistence.metrics.failed",
            extra={"submission_id": submission_id},
        )


async def _persist_submission_static_analysis(
    *,
    user_repository: UserRepository,
    submission_id: int,
    state: Mapping[str, object],
) -> None:
    static_analysis = state.get("static_analysis")
    static_analysis_state = static_analysis if isinstance(static_analysis, Mapping) else {}

    language_raw = static_analysis_state.get("language")
    if isinstance(language_raw, str) and language_raw.strip() != "":
        language = language_raw
    else:
        fallback_language = state.get("language")
        language = fallback_language if isinstance(fallback_language, str) else "python"

    pylint_score = _extract_float_value(static_analysis_state.get("pylint_score"))
    complexity_score = _extract_float_value(static_analysis_state.get("complexity_score"))
    security_warnings = _normalize_json_object_list(static_analysis_state.get("security_warnings"))
    pylint_warnings = _normalize_json_object_list(static_analysis_state.get("pylint_warnings"))

    try:
        is_saved = await user_repository.save_submission_static_analysis(
            submission_id=submission_id,
            language=language,
            pylint_score=pylint_score,
            complexity_score=complexity_score,
            security_warnings=security_warnings,
            pylint_warnings=pylint_warnings,
        )
        if not is_saved:
            _ORCHESTRATION_LOGGER.warning(
                "submission.persistence.static_analysis.not_saved",
                extra={"submission_id": submission_id},
            )
    except Exception:  # noqa: BLE001 - persistence should not crash bot flow
        _ORCHESTRATION_LOGGER.exception(
            "submission.persistence.static_analysis.failed",
            extra={"submission_id": submission_id},
        )


async def _persist_submission_orchestration_artifacts(
    *,
    user_repository: UserRepository,
    submission_id: int,
    state: Mapping[str, object],
) -> None:
    final_score = _extract_float_value(state.get("final_score"))
    recommended_difficulty = _extract_int_value(state.get("recommended_difficulty"))
    if recommended_difficulty is not None and recommended_difficulty < 1:
        recommended_difficulty = None

    try:
        is_saved = await user_repository.save_submission_orchestration_artifacts(
            submission_id=submission_id,
            llm_review=_normalize_json_object(state.get("llm_review")),
            score_breakdown=_normalize_json_object(state.get("score_breakdown")),
            final_score=final_score,
            branching=_normalize_json_object(state.get("branching")),
            recommended_difficulty=recommended_difficulty,
        )
        if not is_saved:
            _ORCHESTRATION_LOGGER.warning(
                "submission.persistence.orchestration_artifacts.not_saved",
                extra={"submission_id": submission_id},
            )
    except Exception:  # noqa: BLE001 - persistence should not crash bot flow
        _ORCHESTRATION_LOGGER.exception(
            "submission.persistence.orchestration_artifacts.failed",
            extra={"submission_id": submission_id},
        )


async def _persist_orchestration_outputs(
    *,
    user_repository: UserRepository,
    submission_id: int,
    state: Mapping[str, object],
) -> None:
    await asyncio.gather(
        _persist_submission_metrics(
            user_repository=user_repository,
            submission_id=submission_id,
            state=state,
        ),
        _persist_submission_static_analysis(
            user_repository=user_repository,
            submission_id=submission_id,
            state=state,
        ),
        _persist_submission_orchestration_artifacts(
            user_repository=user_repository,
            submission_id=submission_id,
            state=state,
        ),
    )


async def _process_saved_submission(
    *,
    user_repository: UserRepository,
    telegram_user_id: int,
    preferred_language: str,
    submission_text: str,
) -> tuple[bool, str]:
    submission_id = await user_repository.save_submission_with_id(
        telegram_id=telegram_user_id,
        code=submission_text,
    )
    if submission_id is None:
        return (
            False,
            "Не удалось сохранить решение. Выберите язык и повторите попытку.",
        )

    try:
        skill_profile = await user_repository.get_skill_profile(telegram_user_id)
    except Exception:  # noqa: BLE001 - profile read must not crash processing
        _ORCHESTRATION_LOGGER.exception(
            "submission.context.skill_profile.failed",
            extra={"telegram_id": telegram_user_id, "submission_id": submission_id},
        )
        skill_profile = None

    language_label = LANGUAGE_LABELS.get(preferred_language, preferred_language)
    try:
        initial_state = _build_submission_agent_state(
            telegram_user_id=telegram_user_id,
            submission_id=submission_id,
            preferred_language=preferred_language,
            submission_text=submission_text,
            skill_profile=skill_profile,
        )
    except Exception as error:  # noqa: BLE001 - invalid state should produce fallback reply
        _ORCHESTRATION_LOGGER.exception(
            "submission.context.agent_state.failed",
            extra={"telegram_id": telegram_user_id, "submission_id": submission_id},
        )
        return True, _build_submission_result_message(
            language_label=language_label,
            submission_id=submission_id,
            state=None,
            orchestration_error=f"{type(error).__name__}: {error}",
        )

    final_state, orchestration_error = await _run_full_graph_cycle_for_submission(initial_state)
    if final_state is not None:
        await _persist_orchestration_outputs(
            user_repository=user_repository,
            submission_id=submission_id,
            state=final_state,
        )
    return True, _build_submission_result_message(
        language_label=language_label,
        submission_id=submission_id,
        state=final_state,
        orchestration_error=orchestration_error,
    )


def _build_dispatcher(user_repository: UserRepository) -> Any:
    _, Dispatcher, Router, CommandStart, F, _, _ = _import_aiogram()

    router = Router()
    settings = get_settings()
    submission_rate_limit_count = max(1, settings.submission_rate_limit_count)
    submission_rate_limit_window_seconds = max(
        1,
        settings.submission_rate_limit_window_seconds,
    )

    @router.message(CommandStart())
    async def start_handler(message: Any) -> None:
        from_user = message.from_user
        if from_user is None:
            await message.answer("Не удалось определить ваш Telegram ID.")
            return

        is_new_user = await user_repository.register_user(from_user.id)
        preferred_language = await user_repository.get_preferred_language(from_user.id)
        if preferred_language is None:
            status_text = "Регистрация завершена." if is_new_user else "Профиль найден."
            await message.answer(
                f"{status_text} Чтобы начать, выберите язык программирования:",
                reply_markup=_language_keyboard(),
            )
            return

        selected_language = LANGUAGE_LABELS.get(preferred_language, preferred_language)
        await message.answer(
            f"Текущий язык: {selected_language}. Можно выбрать другой:",
            reply_markup=_language_keyboard(),
        )

    @router.callback_query(F.data.startswith(LANG_CALLBACK_PREFIX))
    async def select_language_handler(callback_query: Any) -> None:
        from_user = callback_query.from_user
        if from_user is None:
            await callback_query.answer(
                "Не удалось определить пользователя.",
                show_alert=True,
            )
            return

        callback_data = callback_query.data
        if callback_data is None:
            await callback_query.answer("Некорректный выбор.", show_alert=True)
            return

        language_code = callback_data.removeprefix(LANG_CALLBACK_PREFIX)
        language_label = LANGUAGE_LABELS.get(language_code)
        if language_label is None:
            await callback_query.answer("Неизвестный язык.", show_alert=True)
            return

        await user_repository.register_user(from_user.id)
        current_language = await user_repository.get_preferred_language(from_user.id)
        if current_language == language_code:
            await callback_query.answer("Этот язык уже выбран.")
            return

        await user_repository.set_preferred_language(from_user.id, language_code)

        if current_language is None:
            response_text = f"Язык сохранен: {language_label}. Теперь можно продолжить."
        else:
            response_text = f"Язык обновлен: {language_label}."

        await callback_query.answer("Готово.")
        if callback_query.message is not None:
            await callback_query.message.answer(response_text)

    @router.message()
    async def language_required_handler(message: Any) -> None:
        from_user = message.from_user
        if from_user is None:
            return

        if message.text is not None and message.text.startswith("/"):
            return

        await user_repository.register_user(from_user.id)
        preferred_language = await user_repository.get_preferred_language(from_user.id)
        if preferred_language is not None:
            submission_text, validation_error = _validate_submission_text(message.text)
            if validation_error is not None:
                await message.answer(validation_error)
                return

            if submission_text is None:
                await message.answer("Не удалось обработать сообщение.")
                return

            recent_submissions = await user_repository.count_recent_submissions(
                from_user.id,
                submission_rate_limit_window_seconds,
            )
            if recent_submissions >= submission_rate_limit_count:
                await message.answer(
                    "Слишком много отправок за короткий интервал. "
                    f"Лимит: {submission_rate_limit_count} за "
                    f"{submission_rate_limit_window_seconds} сек."
                )
                return

            is_saved, response_text = await _process_saved_submission(
                user_repository=user_repository,
                telegram_user_id=from_user.id,
                preferred_language=preferred_language,
                submission_text=submission_text,
            )
            if not is_saved:
                await message.answer(
                    response_text,
                    reply_markup=_language_keyboard(),
                )
                return

            await message.answer(response_text)
            return

        await message.answer(
            "Перед отправкой решений выберите язык программирования.",
            reply_markup=_language_keyboard(),
        )

    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    return dispatcher


async def run_bot() -> None:
    settings = get_settings()
    if settings.bot_token is None:
        print("BOT_TOKEN is not set. Configure .env before running the Telegram bot.")
        return

    try:
        user_repository = build_user_repository(settings.database_url)
    except (RuntimeError, ValueError) as exc:
        print(f"Bot startup failed: {exc}")
        return

    try:
        await user_repository.ensure_schema()
        Bot, _, _, _, _, _, _ = _import_aiogram()
        bot = Bot(token=settings.bot_token)
        dispatcher = _build_dispatcher(user_repository)
    except RuntimeError as exc:
        print(f"Bot startup failed: {exc}")
        return

    print("Bot service started. Polling Telegram updates.")
    try:
        await dispatcher.start_polling(bot)
    finally:
        await bot.session.close()


def main() -> None:
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        print("Bot service interrupted by user.")


if __name__ == "__main__":
    main()
