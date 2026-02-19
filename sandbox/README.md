# Sandbox Docker Templates

This folder contains per-language sandbox templates:

- `sandbox/python`
- `sandbox/go`
- `sandbox/java`
- `sandbox/cpp`

Each template runs code inside an isolated container for a single language runtime.
Runner scripts also emit memory usage marker (`__METRIC_MAX_RSS_KB__`) via `/usr/bin/time`
to support metrics collection in the orchestrator wrapper.

## Build images

```bash
docker build -f sandbox/python/Dockerfile -t interview-assistant/sandbox-python:local .
docker build -f sandbox/go/Dockerfile -t interview-assistant/sandbox-go:local .
docker build -f sandbox/java/Dockerfile -t interview-assistant/sandbox-java:local .
docker build -f sandbox/cpp/Dockerfile -t interview-assistant/sandbox-cpp:local .
```

## Execute code

Use strong runtime restrictions when launching containers:

```bash
docker run --rm \
  --network none \
  --cpus 1 \
  --memory 256m \
  --pids-limit 64 \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  -v "$PWD:/workspace:ro" \
  interview-assistant/sandbox-python:local /workspace/main.py
```

Language-specific defaults (when no path is provided):

- Python: `/workspace/main.py`
- Go: `/workspace/main.go`
- Java: `/workspace/Main.java` (optional second arg: main class name)
- C++: `/workspace/main.cpp`
