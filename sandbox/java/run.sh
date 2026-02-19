#!/usr/bin/env sh
set -eu

SOURCE_PATH="${1:-/workspace/Main.java}"
MAIN_CLASS="${2:-Main}"
CLASS_DIR="/tmp/java-classes"

if [ ! -f "$SOURCE_PATH" ]; then
  echo "source file not found: $SOURCE_PATH" >&2
  exit 2
fi

mkdir -p "$CLASS_DIR"
javac -d "$CLASS_DIR" "$SOURCE_PATH"

if [ -x /usr/bin/time ]; then
  /usr/bin/time -f "__METRIC_MAX_RSS_KB__:%M" java -cp "$CLASS_DIR" "$MAIN_CLASS"
else
  java -cp "$CLASS_DIR" "$MAIN_CLASS"
fi
