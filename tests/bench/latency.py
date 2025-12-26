import argparse
import asyncio
import statistics

from pyreqwest.http import Url

from tests.bench.runner import Runner
from tests.bench.server import server
from tests.bench.utils import StatsCollection, fmt_size, is_sync


class PerformanceLatency:
    def __init__(self, server_url: Url, lib: str, trust_cert_der: bytes) -> None:
        self.lib = lib
        self.body_sizes = [
            5_000_000,  # 5MB
            1_000_000,  # 1MB
            100_000,  # 100KB
            10_000,  # 10KB
        ]
        self.concurrency_levels = [20, 5, 2] if is_sync(lib) else [100, 10, 2]
        self.runner = Runner(
            url=server_url.with_query({"echo_only_body": "1"}),
            trust_cert_der=trust_cert_der,
            big_body_limit=1_000_000,
            big_body_chunk_size=1024 * 1024,
            num_requests=100,
            warmup_iterations=5,
            iterations=50,
        )

    async def run_benchmarks(self) -> None:
        print(f"Starting performance benchmark for {self.lib}...")
        print(f"Body sizes: {[fmt_size(size) for size in self.body_sizes]}")
        print(f"Concurrency levels: {self.concurrency_levels}")
        print(f"Warmup iterations: {self.runner.warmup_iterations}")
        print(f"Benchmark iterations: {self.runner.iterations}")
        print()

        for body_size in self.body_sizes:
            body = b"x" * body_size
            for concurrency in self.concurrency_levels:
                timings = await self.runner.run_lib(self.lib, body, concurrency)
                print(f"{self.lib} average: {statistics.mean(timings):.4f}ms\n")
                StatsCollection.save_result(self.lib, body_size, concurrency, timings)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Performance latency")
    parser.add_argument("--lib", type=str)
    args = parser.parse_args()

    async with server() as (url, trust_cert_der):
        await PerformanceLatency(url, args.lib, trust_cert_der).run_benchmarks()


if __name__ == "__main__":
    asyncio.run(main())
