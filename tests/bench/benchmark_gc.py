import argparse
import asyncio
import gc
import inspect
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor

from pyreqwest.client import ClientBuilder, SyncClientBuilder
from pyreqwest.http import Url

from tests.servers.echo_server import EchoServer
from tests.servers.server import ServerConfig, find_free_port
from tests.servers.server_subprocess import SubprocessServer


class PerformanceGcPressure:
    """Benchmark class for comparing HTTP client performance."""

    def __init__(self, server_url: Url, lib: str) -> None:
        """Initialize benchmark."""
        self.url = server_url.with_query({"echo_only_body": "1"})
        self.lib = lib
        self.bodies = [
            b"x" * 10_000,  # 10KB
            b"x" * 100_000,  # 100KB
            b"x" * 1_000_000,  # 1MB
            b"x" * 5_000_000,  # 5MB
        ]
        self.big_body_limit = 1_000_000
        self.big_body_chunk_size = 1024 * 1024
        self.requests = 100
        self.concurrency_levels = [2, 10]
        self.iterations = 50

    async def meas_concurrent_batch(self, fn: Callable[[], Awaitable[None]], concurrency: int) -> None:
        semaphore = asyncio.Semaphore(concurrency)

        async def run() -> None:
            async def sem_fn() -> None:
                async with semaphore:
                    await fn()

            await asyncio.gather(*(sem_fn() for _ in range(self.requests)))

        for _ in range(self.iterations):
            await run()

    def sync_meas_concurrent_batch(self, fn: Callable[[], None], concurrency: int) -> None:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:

            def run() -> None:
                futures = [executor.submit(fn) for _ in range(self.requests)]
                _ = [f.result() for f in futures]

            for _ in range(self.iterations):
                run()

    def body_parts_sync(self, body: bytes) -> Iterator[bytes]:
        chunk_size = self.big_body_chunk_size
        for i in range(0, len(body), chunk_size):
            yield body[i : i + chunk_size]

    async def body_parts(self, body: bytes) -> AsyncGenerator[bytes, None]:
        for part in self.body_parts_sync(body):
            yield part

    async def benchmark_pyreqwest_concurrent(self, body: bytes, concurrency: int) -> None:
        async with ClientBuilder().build() as client:

            async def post_read() -> None:
                if len(body) <= self.big_body_limit:
                    response = await client.post(self.url).body_bytes(body).build().send()
                    assert len(await response.bytes()) == len(body)
                else:
                    async with (
                        client.post(self.url)
                        .body_stream(self.body_parts(body))
                        .streamed_read_buffer_limit(65536 * 2)  # Same as aiohttp read buffer high watermark
                        .build_streamed() as response
                    ):
                        tot = 0
                        while chunk := await response.body_reader.read(self.big_body_chunk_size):
                            assert len(chunk) <= self.big_body_chunk_size
                            tot += len(chunk)
                        assert tot == len(body)

            await self.meas_concurrent_batch(post_read, concurrency)

    def benchmark_sync_pyreqwest_concurrent(self, body: bytes, concurrency: int) -> None:
        with SyncClientBuilder().build() as client:

            def post_read() -> None:
                if len(body) <= self.big_body_limit:
                    response = client.post(self.url).body_bytes(body).build().send()
                    assert len(response.bytes()) == len(body)
                else:
                    with (
                        client.post(self.url)
                        .body_stream(self.body_parts_sync(body))
                        .streamed_read_buffer_limit(65536 * 2)  # Same as aiohttp read buffer high watermark
                        .build_streamed() as response
                    ):
                        tot = 0
                        while chunk := response.body_reader.read(self.big_body_chunk_size):
                            assert len(chunk) <= self.big_body_chunk_size
                            tot += len(chunk)
                        assert tot == len(body)

            self.sync_meas_concurrent_batch(post_read, concurrency)

    async def benchmark_aiohttp_concurrent(self, body: bytes, concurrency: int) -> None:
        import aiohttp

        url_str = str(self.url)

        async with aiohttp.ClientSession() as session:

            async def post_read() -> None:
                if len(body) <= self.big_body_limit:
                    async with session.post(url_str, data=body) as response:
                        assert len(await response.read()) == len(body)
                else:
                    async with session.post(url_str, data=self.body_parts(body)) as response:
                        tot = 0
                        async for chunk in response.content.iter_chunked(self.big_body_chunk_size):
                            assert len(chunk) <= self.big_body_chunk_size
                            tot += len(chunk)
                        assert tot == len(body)

            await self.meas_concurrent_batch(post_read, concurrency)

    async def benchmark_httpx_concurrent(self, body: bytes, concurrency: int) -> None:
        import httpx

        url_str = str(self.url)

        async with httpx.AsyncClient() as client:

            async def post_read() -> None:
                if len(body) <= self.big_body_limit:
                    response = await client.post(url_str, content=body)
                    assert len(await response.aread()) == len(body)
                else:
                    response = await client.post(url_str, content=self.body_parts(body))
                    tot = 0
                    async for chunk in response.aiter_bytes(self.big_body_chunk_size):
                        assert len(chunk) <= self.big_body_chunk_size
                        tot += len(chunk)
                    assert tot == len(body)

            await self.meas_concurrent_batch(post_read, concurrency)

    def benchmark_urllib3_concurrent(self, body: bytes, concurrency: int) -> None:
        import urllib3

        url_str = str(self.url)

        with urllib3.PoolManager(maxsize=concurrency) as pool:
            if len(body) <= self.big_body_limit:

                def post_read() -> None:
                    response = pool.request("POST", url_str, body=body)
                    assert response.status == 200
                    assert len(response.data) == len(body)
            else:

                def post_read() -> None:
                    response = pool.request("POST", url_str, body=self.body_parts_sync(body), preload_content=False)
                    assert response.status == 200
                    tot = 0
                    while chunk := response.read(self.big_body_chunk_size):
                        assert len(chunk) <= self.big_body_chunk_size
                        tot += len(chunk)
                    assert tot == len(body)
                    response.release_conn()

            self.sync_meas_concurrent_batch(post_read, concurrency)

    async def benchmark_lib_concurrent(self, body: bytes, concurrency: int) -> None:
        """Dispatch to the appropriate benchmark method based on comparison library."""
        if self.lib == "pyreqwest":
            await self.benchmark_pyreqwest_concurrent(body, concurrency)
        elif self.lib == "pyreqwest_sync":
            self.benchmark_sync_pyreqwest_concurrent(body, concurrency)
        elif self.lib == "aiohttp":
            await self.benchmark_aiohttp_concurrent(body, concurrency)
        elif self.lib == "httpx":
            await self.benchmark_httpx_concurrent(body, concurrency)
        elif self.lib == "urllib3":
            self.benchmark_urllib3_concurrent(body, concurrency)
        else:
            raise ValueError(f"Unsupported comparison library: {self.lib}")

    async def run_cases(self, fn: Callable[[bytes, int], Awaitable[None] | None]) -> None:
        for body in self.bodies:
            for concurrency in self.concurrency_levels:
                print(".", end="", flush=True)
                if inspect.iscoroutinefunction(fn):
                    await fn(body, concurrency)
                else:
                    fn(body, concurrency)

    async def run_benchmarks(self) -> None:
        """Run all benchmarks."""
        print("Starting performance benchmarks...")
        print(f"Running {self.lib}")
        print(f"Body sizes: {[self.fmt_size(len(body)) for body in self.bodies]}")
        print(f"Concurrency levels: {self.concurrency_levels}")
        print(f"Benchmark iterations: {self.iterations}")
        print(f"Running {self.lib} benchmark")

        stats_before = gc.get_stats()

        await self.run_cases(self.benchmark_lib_concurrent)

        gc.collect()
        gc.collect()
        gc.collect()
        stats_after = gc.get_stats()

        print("\nGarbage collection stats:")
        for gen in range(len(stats_after)):
            gen_collections = stats_after[gen]["collections"] - stats_before[gen]["collections"]
            gen_collected = stats_after[gen]["collected"] - stats_before[gen]["collected"]
            print(f"Generation {gen}, collections={gen_collections}, collected={gen_collected}")

    def fmt_size(self, size: int) -> str:
        return f"{size // 1000}KB" if size < 1_000_000 else f"{size // 1_000_000}MB"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Performance benchmark")
    parser.add_argument(
        "--lib", type=str, choices=["pyreqwest", "pyreqwest_sync", "aiohttp", "httpx", "urllib3"], default="aiohttp"
    )

    args = parser.parse_args()

    server = await SubprocessServer.start(EchoServer, ServerConfig(), find_free_port())
    try:
        benchmark = PerformanceGcPressure(server.url, args.lib)
        await benchmark.run_benchmarks()
    finally:
        await server.kill()


if __name__ == "__main__":
    asyncio.run(main())
