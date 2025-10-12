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

### Benchmark

```bash
make bench lib=urllib3
make bench lib=aiohttp
make bench lib=httpx
```
Benchmarks run against (concurrency limited) embedded server to minimize any network effects on latency measurements.
These were run on Apple M3 Max machine with 36GB RAM (OS 15.6.1).
