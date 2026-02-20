---
name: interview-assistant-workflow
description: Implement and maintain the Agentic Interview Coding Assistant monorepo (bot, orchestrator graph nodes, sandbox, repository layer, rate limiting, reliability/fallback logic, offline evaluation dataset, and tests). Use this skill when tasks involve changing feature behavior, adding graph nodes, fixing pipeline regressions, updating scoring/profile logic, hardening sandbox/LLM error handling, or stabilizing CI in this specific repository.
---

# Interview Assistant Workflow

## Follow This Scope First

1. Identify affected layer:
   - Bot flow: `apps/bot/src/interview_bot`
   - Orchestrator graph/state: `services/orchestrator/src/interview_orchestrator`
   - Shared settings: `libs/common/src/interview_common`
   - Execution environment: `sandbox/*`
   - Offline evaluation checks: `services/orchestrator/src/interview_orchestrator/evaluation_dataset.py`
2. Prefer editing existing modules before adding new files/classes.
3. Keep changes consistent with current architecture and tests.

## Implement Graph-Node Tasks (EPIC 7.2)

1. Edit node logic in `services/orchestrator/src/interview_orchestrator/state_steps.py`.
2. Keep `AgentState` contract aligned in `services/orchestrator/src/interview_orchestrator/agent_state.py`.
3. Preserve node sequence and data flow:
   `GenerateTask -> ExecuteSandbox -> RunTests -> StaticAnalysis -> LLMReview -> ScoreAggregation -> UpdateProfile -> BranchingLogic`.
4. Inject dependencies through optional executors/runner arguments instead of hard-coding external calls.
5. Return validated state (`validate_agent_state`) after each node update.
6. Preserve expected graph outputs:
   `test_results`, `static_analysis`, `llm_review`, `observability_metrics`, `score_breakdown`, `final_score`, `skill_profile`, `branching`.
7. Add or adjust tests in `tests/test_graph_nodes.py` for new behavior.

## Implement Code Review Prompting (EPIC 8.2)

1. Keep `LLMReview` deterministic in `services/orchestrator/src/interview_orchestrator/state_steps.py`.
2. Preserve prompt-driven structured output in `LangChainCloudLLMReviewer`:
   - review template
   - score template
   - improvement suggestions
3. Keep normalization centralized via `_normalize_llm_review_payload(...)`.
4. Ensure normalized `llm_review` includes:
   - `summary`, `strengths`, `issues`
   - `improvement_suggestions`
   - `score_template` with `correctness/performance/readability/security/final_score`
   - `score`, `reviewer`
5. Keep numeric score fields clamped to `[0, 100]`.
6. If `score_template` exists, keep `score == score_template.final_score`.
7. Keep heuristic fallback deterministic; cloud reviewer failures must degrade gracefully.
8. Keep reviewer exception fallback deterministic inside `llm_review_node`
   (`reviewer = "heuristic_fallback"` and optional `llm_error`).
9. Keep LLM rate-limit degradation deterministic (`reviewer = "heuristic_rate_limited"`).
10. Keep `observability_metrics` computation wired in `llm_review_node`, including fallback/rate-limited branches.
11. Add/adjust tests in `tests/test_graph_nodes.py` for:
   - retry-to-success cloud review behavior
   - malformed payload normalization
   - `score_template.final_score` precedence
   - LLM rate-limit behavior
   - reviewer exception fallback behavior

## Implement Reliability/Rate-Limit Tasks (EPIC 14/15)

1. Bot-side submission throttling should use repository-backed counting (`count_recent_submissions`) and env settings from `interview_common.settings`.
2. LLM throttling should be enforced through limiter abstraction (default in-memory limiter), without breaking graph flow.
3. Preserve graceful fallback on LLM errors/rate limits; avoid propagating recoverable exceptions.
4. Preserve sandbox crash recovery in `DockerSandboxRunner.execute(...)`:
   - return structured `SandboxExecutionResult` on crash/timeout
   - always attempt container cleanup in `finally`
5. Preserve state-step recovery in `execute_sandbox_step(...)` when injected executor raises:
   - write deterministic `execution_result`/`metrics`
   - continue graph flow without unhandled exception
6. Keep structured logs for crash/fallback/rate-limit events.
7. Add/adjust tests in:
   - `tests/test_graph_nodes.py`
   - `tests/test_user_repository.py`
   - `tests/test_sandbox_runner.py`
   - `tests/test_bootstrap.py`

## Implement Offline Evaluation Tasks (EPIC 12)

1. Keep deterministic offline dataset in `evaluation_dataset.py`:
   `correct_solution`, `inefficient_solution`, `security_bad_solution`.
2. Keep scoring consistency checks deterministic.
3. Validate with `tests/test_offline_dataset.py`.

## Implement Bot/Repository Tasks

1. Keep bot handlers async and deterministic in `apps/bot/src/interview_bot/main.py`.
2. Preserve submission validation behavior:
   - Reject empty input
   - Enforce max submission size
   - Require selected language before saving submission
   - Enforce per-user submission rate limit
3. Mirror schema-sensitive changes in repository tests:
   - `tests/test_user_repository.py`
   - `tests/test_submission_validation.py`

## Implement Sandbox/Execution Tasks

1. Keep isolation constraints in `sandbox_runner.py`:
   no network, resource limits, read-only fs, seccomp, container cleanup.
2. Preserve per-language defaults:
   - Python: `main.py`
   - Go: `main.go`
   - Java: `Main.java`
   - C++: `main.cpp`
3. Validate with:
   - `tests/test_sandbox_runner.py`
   - `tests/test_sandbox_templates.py`

## Validate Before Finalizing

1. Run focused tests first.
2. Run full suite if change spans multiple layers.
3. Use these commands:

```bash
make test
make ci
```

For targeted graph-node checks:

```bash
PYTHONPATH=libs/common/src:apps/bot/src:services/orchestrator/src \
.venv/bin/python -m unittest tests.test_graph_nodes tests.test_agent_state -v
```

For targeted reliability/rate-limit checks:

```bash
PYTHONPATH=libs/common/src:apps/bot/src:services/orchestrator/src \
.venv/bin/python -m unittest \
tests.test_user_repository tests.test_bootstrap tests.test_sandbox_runner tests.test_offline_dataset -v
```

## Deliverable Checklist

1. Keep type/lint compatibility (`ruff`, `black`, `mypy`).
2. Keep behavior covered by tests.
3. For EPIC 8.2/14/15 changes, verify LLM review determinism + fallback + rate-limit behavior in tests.
4. For sandbox changes, verify crash/timeout handling and cleanup behavior in tests.
5. Report what changed, why it changed, and which tests were run.
