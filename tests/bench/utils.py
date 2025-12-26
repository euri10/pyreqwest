from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field

RESULTS_FILE = Path(__file__).parent / "benchmark_results.json"


class Stats(BaseModel):
    lib: str
    body_size: int
    concurrency: int
    timings: list[float]


class StatsCollection(BaseModel):
    stats: list[Stats] = Field(default_factory=list)

    def find(self, lib: str, body_size: int, concurrency: int) -> Stats | None:
        for stat in self.stats:
            if stat.lib == lib and stat.body_size == body_size and stat.concurrency == concurrency:
                return stat
        return None

    @classmethod
    def load(cls) -> Self:
        return cls.model_validate_json(RESULTS_FILE.read_text())

    @classmethod
    def save_result(cls, lib: str, body_size: int, concurrency: int, timings: list[float]) -> None:
        collection = cls.model_validate_json(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else cls()
        collection.stats = [
            stat
            for stat in collection.stats
            if not (stat.lib == lib and stat.body_size == body_size and stat.concurrency == concurrency)
        ]
        collection.stats.append(Stats(lib=lib, body_size=body_size, concurrency=concurrency, timings=timings))
        RESULTS_FILE.write_text(collection.model_dump_json())


def fmt_size(size: int) -> str:
    return f"{size // 1000}KB" if size < 1_000_000 else f"{size // 1_000_000}MB"


def is_sync(lib: str) -> bool:
    return lib in {"urllib3", "pyreqwest_sync"}
