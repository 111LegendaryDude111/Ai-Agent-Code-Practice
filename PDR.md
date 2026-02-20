# PDR - Agentic Interview Coding Assistant (Current State)

## 1. Scope

Проект реализован как monorepo с Telegram bot, orchestrator pipeline и sandbox execution для задач по программированию.
Цель: детерминированно оценивать submissions и адаптировать сложность задач без падения всего пайплайна при частичных ошибках.

## 2. Stack

- Backend: Python 3.11+
- Bot: `aiogram`
- Orchestration: state-graph workflow (`state_steps.py`)
- LLM integration: LangChain + cloud LLM API + deterministic normalization
- Execution isolation: Docker (per-language templates)
- Storage: SQLite/Postgres via repository layer
- Static analysis: `pylint`, `radon`, `bandit`

## 3. Runtime Architecture

- `apps/bot/src/interview_bot`
  - регистрация пользователей
  - выбор языка
  - прием и валидация submission
  - submission rate limit
- `services/orchestrator/src/interview_orchestrator`
  - graph nodes и `run_full_graph_cycle(...)`
  - sandbox execution
  - predefined tests
  - static analysis
  - llm review + scoring + profile update + branching
- `libs/common/src/interview_common`
  - typed settings + `.env` loading
- `sandbox/{python,go,java,cpp}`
  - изолированные runtime templates
- `sandbox/seccomp/sandbox-seccomp.json`
  - seccomp profile for container hardening

## 4. Graph Order (Contract)

Фактический порядок узлов:

`GenerateTask -> ExecuteSandbox -> RunTests -> StaticAnalysis -> LLMReview -> ScoreAggregation -> UpdateProfile -> BranchingLogic`

`AgentState` валидируется на границах узлов через `validate_agent_state(...)`.

## 5. AgentState Outputs

Ключевые выходы полного цикла:

- `execution_result`
- `metrics`
- `test_results`
- `static_analysis`
- `llm_review`
- `observability_metrics`
- `score_breakdown`
- `final_score`
- `skill_profile`
- `branching`
- `recommended_difficulty`

## 6. Reliability and Rate Limits (EPIC 14/15)

- Bot-side throttling: per-user submission limit via repository counting.
- Orchestrator-side throttling: per-user LLM limiter (`InMemoryLLMRateLimiter` by default).
- LLM graceful degradation:
  - cloud/reviewer failure -> `reviewer = "heuristic_fallback"`
  - rate limit -> `reviewer = "heuristic_rate_limited"`
- Sandbox graceful degradation:
  - `DockerSandboxRunner.execute(...)` returns structured `SandboxExecutionResult` for timeout/crash paths
  - cleanup is attempted in `finally`
  - `execute_sandbox_step(...)` recovers from injected executor exceptions and writes deterministic `execution_result`/`metrics`
- Structured logs are emitted for recovery and completion paths:
  - `submission.llm_review.*`
  - `submission.sandbox.*`
  - `sandbox.execution.*`
  - `sandbox.cleanup.*`

## 7. Sandbox Security Defaults

- `--network none`
- CPU/RAM/timeout limits
- `--read-only`
- `--tmpfs /tmp`
- `--cap-drop ALL`
- `no-new-privileges`
- seccomp profile
- forced container cleanup

## 8. Operational Validation

Рекомендуемые команды:

```bash
make test
make ci
```

Точечная проверка reliability:

```bash
PYTHONPATH=libs/common/src:apps/bot/src:services/orchestrator/src \
.venv/bin/python -m unittest tests.test_graph_nodes tests.test_sandbox_runner tests.test_agent_state -v
```

## 9. Success Criteria

- Полный graph cycle выполняется без unhandled exceptions в recoverable сценариях.
- LLM review остается структурированным и детерминированным (включая fallback/rate-limit).
- Sandbox crash/timeout paths возвращают диагностируемый результат и не валят pipeline.
- Score/profile/branching вычисляются даже при частичных ошибках исполнения или ревью.
