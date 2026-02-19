#!/usr/bin/env sh
set -eu

SOURCE_PATH="${1:-/workspace/main.py}"

if [ ! -f "$SOURCE_PATH" ]; then
  echo "source file not found: $SOURCE_PATH" >&2
  exit 2
fi

if [ -x /usr/bin/time ]; then
  /usr/bin/time -f "__METRIC_MAX_RSS_KB__:%M" python3 "$SOURCE_PATH"
else
  python3 "$SOURCE_PATH"
fi
