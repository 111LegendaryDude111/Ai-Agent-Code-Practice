#!/usr/bin/env sh
set -eu

SOURCE_PATH="${1:-/workspace/main.go}"

if [ ! -f "$SOURCE_PATH" ]; then
  echo "source file not found: $SOURCE_PATH" >&2
  exit 2
fi

export GO111MODULE=off
export GOPATH=/tmp/go
export GOCACHE=/tmp/go-cache
mkdir -p "$GOPATH" "$GOCACHE"

if [ -x /usr/bin/time ]; then
  /usr/bin/time -f "__METRIC_MAX_RSS_KB__:%M" go run "$SOURCE_PATH"
else
  go run "$SOURCE_PATH"
fi
