#!/usr/bin/env sh
set -eu

SOURCE_PATH="${1:-/workspace/main.cpp}"
OUTPUT_PATH="/tmp/main"

if [ ! -f "$SOURCE_PATH" ]; then
  echo "source file not found: $SOURCE_PATH" >&2
  exit 2
fi

g++ -O2 -std=c++17 -pipe "$SOURCE_PATH" -o "$OUTPUT_PATH"

if [ -x /usr/bin/time ]; then
  /usr/bin/time -f "__METRIC_MAX_RSS_KB__:%M" "$OUTPUT_PATH"
else
  "$OUTPUT_PATH"
fi
