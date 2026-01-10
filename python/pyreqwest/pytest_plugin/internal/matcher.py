from dataclasses import dataclass
from re import Pattern

try:
    from typing import override

    from dirty_equals import DirtyEquals as _DirtyEqualsBase
except ImportError:
    _DirtyEqualsBase = None  # type: ignore[assignment,misc]
    from typing_extensions import override


@dataclass
class InternalMatcher:
    matcher: str | Pattern[str] | object
    matcher_repr: str = ""

    def matches(self, value: object) -> bool:
        if isinstance(self.matcher, Pattern):
            return self.matcher.search(str(value)) is not None
        return bool(value == self.matcher)

    def __post_init__(self) -> None:
        if _DirtyEqualsBase is not None and isinstance(self.matcher, _DirtyEqualsBase):
            # Need to memoize DirtyEquals repr so it is not messing its repr when doing __eq__:
            # https://dirty-equals.helpmanual.io/latest/usage/#__repr__-and-pytest-compatibility
            self.matcher_repr = repr(self.matcher)
        elif isinstance(self.matcher, str):
            self.matcher_repr = self.matcher
        elif isinstance(self.matcher, Pattern):
            self.matcher_repr = f"{self.matcher.pattern} (regex)"
        else:
            self.matcher_repr = repr(self.matcher)

    @override
    def __repr__(self) -> str:
        return f"Matcher({self.matcher_repr})"
