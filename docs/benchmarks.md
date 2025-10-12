## Latency

### Compared to urllib3 (sync)

<p align="center">
    <img width="1200" alt="urllib3" src="https://raw.githubusercontent.com/MarkusSintonen/pyreqwest/refs/heads/main/tests/bench/benchmark_urllib3.png" />
</p>

### Compared to aiohttp (async)

<p align="center">
    <img width="1200" alt="aiohttp" src="https://raw.githubusercontent.com/MarkusSintonen/pyreqwest/refs/heads/main/tests/bench/benchmark_aiohttp.png" />
</p>

### Compared to httpx (async)

<p align="center">
    <img width="1200" alt="httpx" src="https://raw.githubusercontent.com/MarkusSintonen/pyreqwest/refs/heads/main/tests/bench/benchmark_httpx.png" />
</p>

---

## GC pressure

| Library (mode)      | Total Collections | Total Collected |
|---------------------|-------------------|-----------------|
| pyreqwest (async)   | 4                 | 6               |
| pyreqwest (sync)    | 4                 | 6               |
| urllib3 (sync)      | 9                 | 19              |
| aiohttp (async)     | 67                | 1,886           |
| httpx (async)       | 578               | 1,588,717       |

---

## Benchmark

```bash
make bench lib=urllib3
make bench lib=aiohttp
make bench lib=httpx
make bench-gc lib=urllib3
make bench-gc lib=aiohttp
make bench-gc lib=httpx
```
Benchmarks run against (concurrency limited) embedded server to minimize any network effects on latency measurements.
These were run on Apple M3 Max machine with 36GB RAM (OS 15.6.1).
