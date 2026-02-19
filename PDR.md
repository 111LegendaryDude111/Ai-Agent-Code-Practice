Project: Agentic Interview Coding Assistant


1. Updated Tech Stack

Core Stack
Backend: Python 3.11+
Agent Framework: LangGraph
LLM Abstraction: LangChain
LLM Provider: Cloud LLM via API
Bot Interface: aiogram (Telegram)
Execution Isolation: Docker
Database: Postgres (JSONB usage)
Queue (optional Phase 2): Redis / Celery
Observability: Logging + Prometheus (optional)

2. Updated System Architecture
Telegram User
     ↓
aiogram Bot Layer
     ↓
API / Service Layer
     ↓
LangGraph Orchestrator
     ↓
 ├── Task Generator Agent (LLM)
 ├── Execution Agent (Sandbox Tool)
 ├── Test Runner Tool
 ├── Static Analyzer Tool
 ├── Performance Profiler Tool
 ├── Reviewer Agent (LLM)
 └── Progress Analyzer Agent (LLM)
     ↓
Postgres

3. Telegram Flow (aiogram Layer)
3.1 On Bot Start

При первом запуске:

Пользователь выбирает:
Python
Go
Java
C++
(расширяемо)

Сохраняем:
preferred_language
interview_language (EN / RU — опционально)
current_skill_profile
User Table (updated)
users:
- id
- telegram_id
- preferred_language
- skill_profile (JSONB)
- created_at

4. Agent Orchestration (LangGraph)
LangGraph используется как state-machine orchestrator.

4.1 Graph Nodes
START
  ↓
GenerateTask
  ↓
WaitForSubmission
  ↓
DetectLanguage
  ↓
ExecuteInSandbox
  ↓
RunTests
  ↓
CollectMetrics
  ↓
StaticAnalysis
  ↓
LLMReview
  ↓
ScoreAggregation
  ↓
UpdateUserProfile
  ↓
END

4.2 Agent State (LangGraph State)
class AgentState(TypedDict):
    user_id: str
    task_id: str
    language: str
    code: str
    execution_result: dict
    test_results: dict
    metrics: dict
    static_analysis: dict
    llm_review: dict
    final_score: float

5. LLM Integration (Cloud API)
    Requirements
    API-based inference
    Structured output enforced (JSON schema)
    Temperature control (deterministic review)
    Retry logic
    Token usage logging

    5.1 LLM Usage Points
    LLM используется для:
    Генерации задач
    Code review
    Difficulty adaptation
    Weakness profiling
    Hint generation

6. Sandbox Design (Updated)

Для каждого языка создаётся Docker template:

/sandbox/python
/sandbox/go
/sandbox/java
/sandbox/cpp


Execution pipeline:
Code injection into container
Compile (если нужно)
Execute with:
CPU limit
Memory limit
Timeout

Collect:
runtime
memory
exit code
stdout
stderr

7. Difficulty & Language Adaptation
7.1 Language-Based Task Filtering

Tasks table updated:

tasks:
- id
- language
- category
- difficulty
- test_cases


При выборе языка:

агент генерирует задачи только под выбранный runtime.

7.2 Adaptive Difficulty Logic

LangGraph branching:

if last_3_scores > threshold:
    increase_difficulty()
elif repeated_failures:
    decrease_difficulty()

8. AI Agent Design (Updated)

Теперь система — это:
    Hybrid Agent System:
    Tool-based deterministic layer
    LLM reasoning layer
    State-driven orchestration
    Profile-aware adaptation

9. Production Concerns
    9.1 LLM Layer

    Rate limiting
    Cost monitoring
    Fallback model
    API error handling
    Circuit breaker

    9.2 Sandbox Layer
    No network
    No mounting host FS
    No privileged mode
    Hard timeout
    Auto container cleanup

    9.3 Bot Layer (aiogram)
    Async handling
    Rate limiting
    Idempotent message processing
    Submission tracking
    Multi-user isolation


10. Updated Success Criteria

    Correct execution isolation
    <3s average response time
    Stable structured LLM outputs
    Language selection working
    Difficulty adaptation working
    Cost per submission controlled