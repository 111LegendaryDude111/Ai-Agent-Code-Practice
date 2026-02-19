# Agentic Interview Coding Assistant

Monorepo foundation for the project described in `PDR.md`.

## Архитектура

Проект разделен на 3 слоя:

- `apps/bot` - Telegram bot (aiogram), принимает пользователя, язык и код.
- `services/orchestrator` - оркестрация pipeline проверки решения (sandbox, тесты, analysis, scoring).
- `libs/common` - общие настройки и загрузка env.

Поток данных:

1. Пользователь отправляет код в Telegram.
2. Bot сохраняет submission в БД.
3. Orchestrator прогоняет graph steps:
   `GenerateTask -> ExecuteSandbox -> RunTests -> StaticAnalysis -> LLMReview -> ScoreAggregation -> UpdateProfile`.
4. Результаты (метрики, статанализ, score) сохраняются и используются для профиля пользователя.

## Краткое описание модулей

### Bot (`apps/bot/src/interview_bot`)

- `main.py` - aiogram handlers (`/start`, выбор языка, валидация и прием submission).
- `user_repository.py` - слой доступа к БД (SQLite/Postgres), миграции, CRUD для пользователей, задач, submissions, metrics, static analysis.

### Orchestrator (`services/orchestrator/src/interview_orchestrator`)

- `agent_state.py` - единый `TypedDict` state + runtime-валидация полей.
- `state_steps.py` - graph nodes и полный цикл `run_full_graph_cycle`.
- `sandbox_runner.py` - безопасный запуск кода в Docker (CPU/RAM/time limits, no network, cleanup).
- `test_runner.py` - запуск predefined JSON test cases, pass/fail и отчет по первому упавшему тесту.
- `static_analysis.py` - Python static analysis через `pylint`, `radon`, `bandit`.
- `main.py` - bootstrap entrypoint сервиса orchestrator.

### Common (`libs/common/src/interview_common`)

- `settings.py` - загрузка `.env` и `Settings` (`APP_ENV`, `LOG_LEVEL`, `DATABASE_URL`, `BOT_TOKEN`, `LLM_API_KEY`).

### Sandbox templates (`sandbox`)

- `sandbox/python`, `sandbox/go`, `sandbox/java`, `sandbox/cpp` - Dockerfile + `run.sh` для изолированного выполнения по языкам.

## Краткий гайд запуска

### 1. Установка зависимостей

```bash
make install
```

`make install` использует `uv` (если доступен) и fallback на `pip`.

### 2. Настройка окружения

```bash
cp .env.example .env
```

Минимально для локального старта без Postgres можно поставить SQLite:

```env
DATABASE_URL=sqlite:///tmp/bot.db
BOT_TOKEN=<your_telegram_bot_token>
```

### 3. Проверка проекта

```bash
make test
```

Если установлены линтеры/тайпчекер:

```bash
make ci
```

### 4. Запуск сервисов

```bash
make run-orchestrator
make run-bot
```

### 5. (Опционально) Сборка sandbox образов

См. `sandbox/README.md` для команд `docker build` по языкам и примеров запуска.

## Tooling

- Workspace/deps: `uv`
- Lint/format/types: `ruff`, `black`, `mypy`
- Hooks: `pre-commit`

## CI/CD

- Workflow: `.github/workflows/ci.yml`
- Проверки: lint/typecheck, unit tests, docker build matrix
- Gate job: `required-ci`

## Python version

- Python `3.11+` (текущий конфиг: `>=3.11,<3.14`)
