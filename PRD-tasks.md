Отлично. Ниже — **PDR Tasks Breakdown** в production-ориентированном формате: эпики → задачи → критерии готовности (DoD). Это можно напрямую переносить в Jira / Linear / GitHub Projects.

---

# 📄 PDR Tasks

## Project: Agentic Interview Coding Assistant

---

# EPIC 1 — Project Foundation

## 1.1 Repository Setup

**Tasks:**

- Создать monorepo
- Настроить poetry / uv
- Добавить pre-commit (ruff, black, mypy)
- Настроить .env management

**DoD:**

- Проект запускается локально
- CI проверяет lint + type check

---

## 1.2 CI/CD

**Tasks:**

- GitHub Actions pipeline
- Lint + test job
- Docker build job

**DoD:**

- PR не мержится без прохождения CI

---

# EPIC 2 — Telegram Bot (aiogram)

## 2.1 Bot Initialization

**Tasks:**

- Создать aiogram приложение
- Реализовать /start
- Сохранение telegram_id

**DoD:**

- Пользователь регистрируется в БД

---

## 2.2 Language Selection Flow

**Tasks:**

- Inline keyboard с языками
- Сохранение preferred_language
- Обработка повторного выбора

**DoD:**

- Пользователь не может начать без выбора языка
- Язык хранится в БД

---

## 2.3 Submission Handling

**Tasks:**

- Обработка текстового кода
- Ограничение размера
- Проверка на пустой ввод

**DoD:**

- Код сохраняется в submissions table

---

# EPIC 3 — Database Layer (Postgres)

## 3.1 Schema Design

**Tasks:**

- users table
- tasks table
- submissions table
- indexes

**DoD:**

- Миграции применяются
- CRUD работает

---

## 3.2 Skill Profile Storage

**Tasks:**

- JSONB профиль пользователя
- Initial profile bootstrap

**DoD:**

- Профиль создаётся автоматически

---

# EPIC 4 — Sandbox Execution System

## 4.1 Docker Template per Language

**Tasks:**

- Python container
- Go container
- Java container
- C++ container

**DoD:**

- Каждый язык исполняется изолированно

---

## 4.2 Secure Execution Wrapper

**Tasks:**

- CPU limit
- Memory limit
- Timeout
- No network
- Auto container cleanup

**DoD:**

- Infinite loop убивается
- Нет доступа к host FS

---

## 4.3 Metrics Collection

**Tasks:**

- Runtime measurement
- Memory usage
- Exit code
- Stdout / stderr

**DoD:**

- Метрики сохраняются в БД

---

# EPIC 5 — Test Runner

## 5.1 Predefined Test Cases

**Tasks:**

- JSON test format
- Input/output validation
- Edge case tests

**DoD:**

- Pass/fail подсчитывается корректно

---

## 5.2 Failure Reporting

**Tasks:**

- Capture first failed test
- Provide diff output

**DoD:**

- Пользователь видит причину падения

---

# EPIC 6 — Static Analysis Layer

## 6.1 Python Analysis

**Tasks:**

- pylint integration
- radon complexity
- bandit security

**DoD:**

- Complexity score сохраняется
- Security warnings отображаются

---

# EPIC 7 — LangGraph Orchestration

## 7.1 Agent State Definition

**Tasks:**

- TypedDict state
- State validation

**DoD:**

- Все шаги используют единый state

---

## 7.2 Graph Node Implementation

**Tasks:**

- GenerateTask node
- ExecuteSandbox node
- RunTests node
- StaticAnalysis node
- LLMReview node
- ScoreAggregation node
- UpdateProfile node

**DoD:**

- Полный цикл выполняется без падений

---

## 7.3 Branching Logic

**Tasks:**

- Retry on failure
- Adaptive difficulty branch
- Hint branch

**DoD:**

- Граф корректно маршрутизирует state

---

# EPIC 8 — LLM Integration (Cloud API)

## 8.1 LangChain Setup

**Tasks:**

- Cloud LLM wrapper
- Structured output parser
- Retry logic

**DoD:**

- JSON всегда валиден

---

## 8.2 Code Review Prompting

**Tasks:**

- Review template
- Score template
- Improvement suggestions

**DoD:**

- Review структурирован и детерминирован

---

# EPIC 9 — Scoring Engine

## 9.1 Score Aggregation Logic

**Tasks:**

- Correctness weight
- Performance weight
- Readability weight
- Security weight

**DoD:**

- Итоговый score воспроизводим

---

# EPIC 10 — User Profiling & Adaptation

## 10.1 Weakness Detection

**Tasks:**

- Track failed categories
- Track complexity mistakes

**DoD:**

- Профиль изменяется после каждой задачи

---

## 10.2 Difficulty Adjustment

**Tasks:**

- Threshold logic
- Difficulty upgrade/downgrade

**DoD:**

- Сложность меняется автоматически

---

# EPIC 11 — Observability

## 11.1 Logging

**Tasks:**

- Structured logs
- LLM token usage logging
- Sandbox crash logging

**DoD:**

- Можно отследить любой submission

---

## 11.2 Metrics

**Tasks:**

- Avg runtime
- Fail rate
- Cost per submission

**DoD:**

- Метрики считаются автоматически

---

# EPIC 12 — Evaluation Framework

## 12.1 Offline Dataset

**Tasks:**

- Correct solutions
- Inefficient solutions
- Security-bad examples

**DoD:**

- Scoring consistency проверена

---

# EPIC 13 — Security Hardening

## 13.1 Container Hardening

**Tasks:**

- seccomp
- ulimit process cap
- read-only FS

**DoD:**

- Container escape невозможен

---

# EPIC 14 — Rate Limiting & Cost Control

## 14.1 API Rate Limit

**Tasks:**

- Per-user submission limit
- LLM call limit

**DoD:**

- Пользователь не может DDOS систему

---

# EPIC 15 — Production Readiness

## 15.1 Error Handling

**Tasks:**

- Graceful fallback on LLM failure
- Sandbox crash recovery

**DoD:**

- Система не падает при частичных ошибках
