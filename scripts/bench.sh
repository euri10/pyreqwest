#!/bin/bash
set -euo pipefail

uv sync
uv run maturin develop --uv --release

libs=(
  "pyreqwest"
  "pyreqwest_sync"
  "aiohttp"
  "urllib3"
  "httpx"
  "rnet"
)
for lib in "${libs[@]}"; do
  uv run python -m tests.bench.latency --lib "$lib"
done

for lib in "${libs[@]}"; do
  if [[ "$lib" != *"pyreqwest"* ]]; then
    uv run python -m tests.bench.plots --lib "$lib"
  fi
done

for lib in "${libs[@]}"; do
  uv run python -m tests.bench.gc --lib "$lib"
done
