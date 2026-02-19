# Agentic Interview Coding Assistant

Monorepo foundation for the project described in `PDR.md`.

## Repository layout

- `apps/bot` - Telegram bot service (aiogram)
- `services/orchestrator` - LangGraph orchestration service
- `libs/common` - shared settings and core utilities

## Quick start

1. Create virtual environment and install dependencies (prefers `uv`, has pip fallback):

```bash
make install
```

If internet is unavailable, `make install` still creates an offline local source-link setup for running services.

2. Copy env template and set secrets:

```bash
cp .env.example .env
```

3. Run local checks:

```bash
make ci
```

4. Run service bootstrap commands:

```bash
make run-orchestrator
make run-bot
```

## Tooling

- Dependency + workspace management: `uv`
- Formatting/Linting/Types: `black`, `ruff`, `mypy`
- Git hooks: `pre-commit`

## Sandbox Templates

Language-specific Docker sandbox templates are available in:

- `sandbox/python`
- `sandbox/go`
- `sandbox/java`
- `sandbox/cpp`

See `sandbox/README.md` for build and execution examples.

Secure execution wrapper (CPU/RAM limits, timeout, no-network, auto-cleanup):

- `services/orchestrator/src/interview_orchestrator/sandbox_runner.py`
- Execution metrics storage (runtime/memory/exit/stdout/stderr):
  - schema + repository methods in `apps/bot/src/interview_bot/user_repository.py`

## CI/CD

- Workflow: `.github/workflows/ci.yml`
- Jobs:
  - `lint-and-typecheck`
  - `unit-tests`
  - `docker-build` matrix for:
    - `bot`
    - `orchestrator`
    - `sandbox-python`
    - `sandbox-go`
    - `sandbox-java`
    - `sandbox-cpp`
  - `required-ci` (aggregated gate check)

To enforce `DoD` ("PR не мержится без прохождения CI"), enable branch protection for `main` and set `required-ci` as required status check.

## Python version

- Supported runtime: Python `3.11+` (tested config allows `3.11` - `3.13`)
