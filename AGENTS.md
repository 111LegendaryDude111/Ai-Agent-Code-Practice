# AGENTS.md

## Purpose

Use this file as the default operating guide for AI/code agents in this repository.
Goal: implement features quickly without breaking the bot/orchestrator pipeline.

## Project Map

- `apps/bot/src/interview_bot`
  Telegram bot (`aiogram`), user registration, language selection, submission intake.
- `services/orchestrator/src/interview_orchestrator`
  Pipeline/state graph nodes, sandbox execution, predefined tests, static analysis, scoring.
- `libs/common/src/interview_common`
  Shared settings and `.env` loading.
- `sandbox/{python,go,java,cpp}`
  Per-language Docker templates and runtime scripts.
- `tests`
  Unit tests for bot repository, state validation, graph nodes, sandbox/test/static-analysis behavior.

## Core Architecture Constraints

1. Keep `AgentState` as the single contract across graph nodes.
2. Validate state at node boundaries via `validate_agent_state(...)`.
3. Preserve the graph order:
   `GenerateTask -> ExecuteSandbox -> RunTests -> StaticAnalysis -> LLMReview -> ScoreAggregation -> UpdateProfile`.
4. Keep sandbox isolation defaults:
   no network, memory/CPU/time limits, read-only rootfs, container cleanup.
5. Keep user-facing Telegram messages in Russian unless explicitly asked to change locale behavior.

## LLM Review Contract (EPIC 8.2)

1. Keep `llm_review` structured and deterministic after `LLMReview` node.
2. Expected normalized keys in `llm_review`:
   - `summary: str`
   - `strengths: list[str]`
   - `issues: list[str]`
   - `improvement_suggestions: list[str]`
   - `score_template: {correctness, performance, readability, security, final_score}`
   - `score: float`
   - `reviewer: str`
3. Keep score bounds in `[0, 100]` for `score` and every numeric field in `score_template`.
4. If `score_template` is present in reviewer payload, `score` must match `score_template.final_score`.
5. Do not bypass `_normalize_llm_review_payload(...)` when writing `state["llm_review"]`.
6. Keep prompt shape aligned with deterministic templates in `LangChainCloudLLMReviewer`.

## Working Rules

1. Prefer minimal targeted edits in existing modules over creating new abstractions.
2. Update tests in the same change when behavior changes.
3. Preserve strict typing (`mypy` is strict and `disallow_untyped_defs = true`).
4. Keep formatting/lint compatibility (`ruff`, `black`).
5. Avoid introducing new dependencies unless required by the task.

## High-Value Entry Points

- Bot runtime: `apps/bot/src/interview_bot/main.py`
- DB/repository: `apps/bot/src/interview_bot/user_repository.py`
- State schema: `services/orchestrator/src/interview_orchestrator/agent_state.py`
- Graph nodes: `services/orchestrator/src/interview_orchestrator/state_steps.py`
- Sandbox executor: `services/orchestrator/src/interview_orchestrator/sandbox_runner.py`
- Static analysis: `services/orchestrator/src/interview_orchestrator/static_analysis.py`

## Local Commands

1. Setup:
   - `make install`
   - `cp .env.example .env`
2. Full checks:
   - `make ci`
3. Unit tests:
   - `make test`
4. Targeted tests:
   - `PYTHONPATH=libs/common/src:apps/bot/src:services/orchestrator/src .venv/bin/python -m unittest tests.test_graph_nodes -v`
   - `PYTHONPATH=libs/common/src:apps/bot/src:services/orchestrator/src .venv/bin/python -m unittest tests.test_agent_state -v`
   - `PYTHONPATH=libs/common/src:apps/bot/src:services/orchestrator/src .venv/bin/python -m unittest tests.test_user_repository -v`

## Definition of Done for Graph Node Work

1. `run_full_graph_cycle(...)` completes without exceptions.
2. Node outputs are written to expected state keys (`test_results`, `static_analysis`, `llm_review`, `score_breakdown`, `final_score`, `skill_profile`).
3. `llm_review` includes deterministic review fields (`summary`, `strengths`, `issues`, `improvement_suggestions`, `score_template`, `score`, `reviewer`) when `LLMReview` behavior is touched.
4. `tests/test_graph_nodes.py` passes.
5. Related contract tests (`tests/test_agent_state.py`, `tests/test_submission_validation.py`) pass when touched behavior affects them.
