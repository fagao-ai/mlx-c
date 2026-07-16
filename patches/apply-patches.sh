#!/bin/sh

set -eu

for patch in "$@"; do
  if git apply --reverse --check "$patch" >/dev/null 2>&1; then
    continue
  fi
  git apply --whitespace=nowarn "$patch"
done
