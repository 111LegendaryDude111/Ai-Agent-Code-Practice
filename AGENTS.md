# AGENTS.md

## Purpose

Use this file as the default operating guide for AI/code agents in this repository.
Goal: implement features quickly without breaking the bot/orchestrator pipeline.

## Project Map

- `apps/bot/src/interview_bot`
  Telegram bot (`aiogram`), user registration, language selection, submission intake, rate limiting.
- `services/orchestrator/src/interview_orchestrator`
  Pipeline/state graph nodes, sandbox execution, predefined tests, static analysis, LLM review, scoring, offline evaluation dataset.
- `libs/common/src/interview_common`
  Shared settings and `.env` loading (including rate-limit configuration).
- `sandbox/{python,go,java,cpp}`
  Per-language Docker templates and runtime scripts.
- `sandbox/seccomp/sandbox-seccomp.json`
  Seccomp profile used by sandbox containers.
- `tests`
  Unit tests for bot repository, state validation, graph nodes, sandbox/test/static-analysis behavior, offline dataset consistency.

## Core Architecture Constraints

1. Keep `AgentState` as the single contract across graph nodes.
2. Validate state at node boundaries via `validate_agent_state(...)`.
3. Preserve the graph order:
   `GenerateTask -> ExecuteSandbox -> RunTests -> StaticAnalysis -> LLMReview -> ScoreAggregation -> UpdateProfile -> BranchingLogic`.
4. Keep sandbox isolation defaults:
   no network, memory/CPU/time limits, read-only rootfs, seccomp profile, container cleanup.
5. Keep user-facing Telegram messages in Russian unless explicitly asked to change locale behavior.
6. Keep partial-failure behavior graceful:
   LLM failures/rate limits and sandbox crashes should return deterministic state/result payloads instead of crashing the full cycle.

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
   - `llm_error: str` (optional, fallback path only)
3. Keep score bounds in `[0, 100]` for `score` and every numeric field in `score_template`.
4. If `score_template` is present in reviewer payload, `score` must match `score_template.final_score`.
5. Do not bypass `_normalize_llm_review_payload(...)` when writing `state["llm_review"]`.
6. Keep prompt shape aligned with deterministic templates in `LangChainCloudLLMReviewer`.
7. Preserve graceful degradation:
   - cloud failure path -> `reviewer = "heuristic_fallback"`
   - reviewer exception path in `llm_review_node` -> `reviewer = "heuristic_fallback"` and `llm_error`
   - LLM rate-limit path -> `reviewer = "heuristic_rate_limited"`
8. Keep `observability_metrics` computed in `llm_review_node` (including fallback/rate-limited paths).

## Rate Limiting & Reliability Contract (EPIC 14/15)

1. Keep bot submission throttling via repository-backed recent submission counting.
2. Keep LLM call throttling in orchestrator via rate limiter abstraction (default in-memory limiter).
3. Keep sandbox crash recovery behavior in `DockerSandboxRunner.execute(...)`:
   return `SandboxExecutionResult` with diagnostics and always attempt cleanup in `finally`.
4. Keep `execute_sandbox_step(...)` resilient to injected executor failures:
   write deterministic `execution_result`/`metrics` and continue graph execution.
5. Do not turn recoverable execution/review errors into unhandled exceptions in `run_full_graph_cycle(...)`.
6. Preserve structured logging for failure/recovery paths (`submission.llm_review.*`, `sandbox.execution.*`).

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
- Offline dataset evaluation: `services/orchestrator/src/interview_orchestrator/evaluation_dataset.py`

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
   - `PYTHONPATH=libs/common/src:apps/bot/src:services/orchestrator/src .venv/bin/python -m unittest tests.test_sandbox_runner tests.test_sandbox_templates -v`
   - `PYTHONPATH=libs/common/src:apps/bot/src:services/orchestrator/src .venv/bin/python -m unittest tests.test_offline_dataset -v`

## Definition of Done for Graph Node Work

1. `run_full_graph_cycle(...)` completes without exceptions.
2. Node outputs are written to expected state keys (`test_results`, `static_analysis`, `llm_review`, `observability_metrics`, `score_breakdown`, `final_score`, `skill_profile`, `branching`, `recommended_difficulty`).
3. `llm_review` includes deterministic review fields (`summary`, `strengths`, `issues`, `improvement_suggestions`, `score_template`, `score`, `reviewer`) when `LLMReview` behavior is touched.
4. `tests/test_graph_nodes.py` passes, including fallback and rate-limit scenarios when touched behavior affects them.
5. Related contract tests (`tests/test_agent_state.py`, `tests/test_submission_validation.py`, `tests/test_user_repository.py`) pass when touched behavior affects them.
6. If sandbox behavior is touched, `tests/test_sandbox_runner.py` and `tests/test_sandbox_templates.py` pass.
7. Reliability-specific cases stay covered:
   - `test_llm_review_node_falls_back_when_reviewer_raises`
   - `test_run_full_graph_cycle_recovers_from_partial_failures`
   - `test_execute_recovers_when_command_building_crashes`
