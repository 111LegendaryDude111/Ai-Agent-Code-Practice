---
name: interview-assistant-workflow
description: Implement and maintain the Agentic Interview Coding Assistant monorepo (bot, orchestrator graph nodes, sandbox, repository layer, and tests). Use this skill when tasks involve changing feature behavior, adding graph nodes, fixing pipeline regressions, updating scoring/profile logic, or stabilizing CI in this specific repository.
---

# Interview Assistant Workflow

## Follow This Scope First

1. Identify affected layer:
   - Bot flow: `apps/bot/src/interview_bot`
   - Orchestrator graph/state: `services/orchestrator/src/interview_orchestrator`
   - Shared settings: `libs/common/src/interview_common`
   - Execution environment: `sandbox/*`
2. Prefer editing existing modules before adding new files/classes.
3. Keep changes consistent with current architecture and tests.

## Implement Graph-Node Tasks (EPIC 7.2)

1. Edit node logic in `services/orchestrator/src/interview_orchestrator/state_steps.py`.
2. Keep `AgentState` contract aligned in `services/orchestrator/src/interview_orchestrator/agent_state.py`.
3. Preserve node sequence and data flow:
   `GenerateTask -> ExecuteSandbox -> RunTests -> StaticAnalysis -> LLMReview -> ScoreAggregation -> UpdateProfile`.
4. Inject dependencies through optional executors/runner arguments instead of hard-coding external calls.
5. Return validated state (`validate_agent_state`) after each node update.
6. Add or adjust tests in `tests/test_graph_nodes.py` for new behavior.

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
8. Add/adjust tests in `tests/test_graph_nodes.py` for:
   - retry-to-success cloud review behavior
   - malformed payload normalization
   - `score_template.final_score` precedence

## Implement Bot/Repository Tasks

1. Keep bot handlers async and deterministic in `apps/bot/src/interview_bot/main.py`.
2. Preserve submission validation behavior:
   - Reject empty input
   - Enforce max submission size
   - Require selected language before saving submission
3. Mirror schema-sensitive changes in repository tests:
   - `tests/test_user_repository.py`
   - `tests/test_submission_validation.py`

## Implement Sandbox/Execution Tasks

1. Keep isolation constraints in `sandbox_runner.py`:
   no network, resource limits, read-only fs, container cleanup.
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

## Deliverable Checklist

1. Keep type/lint compatibility (`ruff`, `black`, `mypy`).
2. Keep behavior covered by tests.
3. For EPIC 8.2 changes, verify `llm_review` payload determinism in tests.
4. Report what changed, why it changed, and which tests were run.
