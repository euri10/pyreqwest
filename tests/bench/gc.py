import argparse
import asyncio
import gc

from pyreqwest.http import Url

from tests.bench.runner import Runner
from tests.bench.server import server
from tests.bench.utils import fmt_size


class PerformanceGcPressure:
    def __init__(self, server_url: Url, lib: str, trust_cert_der: bytes) -> None:
        self.url = server_url.with_query({"echo_only_body": "1"})
        self.lib = lib
        self.body_sizes = [
            10_000,  # 10KB
            100_000,  # 100KB
            1_000_000,  # 1MB
            5_000_000,  # 5MB
        ]
        self.concurrency_levels = [2, 10]
        self.runner = Runner(
            self.url,
            trust_cert_der,
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

        bodies = [b"x" * size for size in self.body_sizes]

        stats_before = gc.get_stats()

        for body in bodies:
            for concurrency in self.concurrency_levels:
                await self.runner.run_lib(self.lib, body, concurrency)

        gc.collect()
        gc.collect()
        gc.collect()
        stats_after = gc.get_stats()

        tot_collections = 0
        tot_collected = 0
        print(f"{self.lib} garbage collection stats:")
        for gen in range(len(stats_after)):
            gen_collections = stats_after[gen]["collections"] - stats_before[gen]["collections"]
            gen_collected = stats_after[gen]["collected"] - stats_before[gen]["collected"]
            tot_collections += gen_collections
            tot_collected += gen_collected
            print(f"Generation {gen}, collections={gen_collections}, collected={gen_collected}")
        print(f"Total collections={tot_collections}, total collected={tot_collected}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Performance benchmark")
    parser.add_argument("--lib", type=str)

    args = parser.parse_args()

    async with server() as (url, trust_cert_der):
        benchmark = PerformanceGcPressure(url, args.lib, trust_cert_der)
        await benchmark.run_benchmarks()


if __name__ == "__main__":
    asyncio.run(main())
