"""Module providing HTTP request mocking capabilities for pyreqwest clients in tests."""

import json
from collections.abc import AsyncIterable, Awaitable, Callable, Iterable
from functools import cached_property
from re import Pattern
from typing import TYPE_CHECKING, Any, Literal, Self, TypeVar, assert_never, override

import pytest

from pyreqwest.middleware import Next, SyncNext
from pyreqwest.middleware.types import Middleware, SyncMiddleware
from pyreqwest.pytest_plugin.internal.matcher import InternalMatcher
from pyreqwest.pytest_plugin.types import (
    BodyContentMatcher,
    CustomHandler,
    CustomMatcher,
    MethodMatcher,
    PathMatcher,
    QueryMatcher,
    UrlMatcher,
)
from pyreqwest.request import (
    ConsumedRequest,
    OneOffRequestBuilder,
    Request,
    RequestBody,
    RequestBuilder,
    StreamRequest,
    SyncConsumedRequest,
    SyncOneOffRequestBuilder,
    SyncRequestBuilder,
    SyncStreamRequest,
)
from pyreqwest.response import BaseResponse, Response, ResponseBuilder, SyncResponse
from pyreqwest.types import HeadersType

if TYPE_CHECKING:
    MonkeyPatch = pytest.MonkeyPatch
else:
    try:
        pytest_fixture = pytest.fixture
        MonkeyPatch = pytest.MonkeyPatch
    except ImportError:
        pytest_fixture = None  # type: ignore[assignment]
        MonkeyPatch = Any  # type: ignore[assignment,misc]

_R = TypeVar("_R", bound=BaseResponse)


class Mock:
    """Class representing a single mock rule."""

    def __init__(
        self, method: MethodMatcher | None = None, *, path: PathMatcher | None = None, url: UrlMatcher | None = None
    ) -> None:
        """Do not use directly. Instead, use ClientMocker.mock()."""
        self._method_matcher: InternalMatcher | None = InternalMatcher(method) if method is not None else None
        self._path_matcher: InternalMatcher | None = InternalMatcher(path) if path is not None else None
        self._url_matcher: InternalMatcher | None = InternalMatcher(url) if url is not None else None
        self._query_matcher: dict[str, InternalMatcher] | InternalMatcher | None = None
        self._header_matchers: dict[str, InternalMatcher] = {}
        self._body_matcher: tuple[InternalMatcher, Literal["content", "json"]] | None = None
        self._custom_matcher: CustomMatcher | None = None
        self._custom_handler: CustomHandler | None = None

        self._matched_requests: list[Request] = []
        self._unmatched_requests_repr_parts: list[dict[str, str | None]] = []
        self._using_response_builder: bool = False

    def assert_called(
        self,
        *,
        count: int | None = None,
        min_count: int | None = None,
        max_count: int | None = None,
    ) -> None:
        """Assert that this mock was called the expected number of times. By default, exactly once."""
        if count is None and min_count is None and max_count is None:
            count = 1

        if self._assertion_passes(count, min_count, max_count):
            return

        self._assert_fail(count=count, min_count=min_count, max_count=max_count)

    def _assertion_passes(
        self,
        count: int | None,
        min_count: int | None,
        max_count: int | None,
    ) -> bool:
        actual_count = len(self._matched_requests)
        if count is not None:
            return actual_count == count

        min_satisfied = min_count is None or actual_count >= min_count
        max_satisfied = max_count is None or actual_count <= max_count

        return min_satisfied and max_satisfied

    def get_requests(self) -> list[Request]:
        """Get all captured requests by this mock."""
        return [*self._matched_requests]

    def get_call_count(self) -> int:
        """Get the total number of calls to this mock."""
        return len(self._matched_requests)

    def reset_requests(self) -> None:
        """Reset all captured requests for this mock."""
        self._matched_requests.clear()

    def match_query(self, query: QueryMatcher) -> Self:
        """Set a matcher to match the entire query string or query parameters."""
        if isinstance(query, dict):
            self._query_matcher = {k: InternalMatcher(v) for k, v in query.items()}
        else:
            self._query_matcher = InternalMatcher(query)
        return self

    def match_query_param(self, name: str, value: str | Pattern[str] | object) -> Self:
        """Set a matcher to match a specific query parameter."""
        if not isinstance(self._query_matcher, dict):
            self._query_matcher = {}
        self._query_matcher[name] = InternalMatcher(value)
        return self

    def match_header(self, name: str, value: str | Pattern[str] | object) -> Self:
        """Set a matcher to match a specific request header."""
        self._header_matchers[name] = InternalMatcher(value)
        return self

    def match_body(self, matcher: BodyContentMatcher) -> Self:
        """Set a matcher to match request bodies as raw content (text or bytes)."""
        self._body_matcher = (InternalMatcher(matcher), "content")
        return self

    def match_body_json(self, matcher: object) -> Self:
        """Set a matcher to match JSON request bodies."""
        self._body_matcher = (InternalMatcher(matcher), "json")
        return self

    def match_request(self, matcher: Callable[[Request], Awaitable[bool]] | Callable[[Request], bool]) -> Self:
        """Set a custom matcher to match requests."""
        self._custom_matcher = matcher
        return self

    def match_request_with_response(self, handler: CustomHandler) -> Self:
        """Set a custom handler to generate the response for matched requests."""
        assert not self._using_response_builder, "Cannot use response builder and custom handler together"
        self._custom_handler = handler
        return self

    def with_status(self, status: int) -> Self:
        """Set the mocked response status code."""
        _ = self._response_builder.status(status)
        return self

    def with_header(self, name: str, value: str) -> Self:
        """Add a header to the mocked response."""
        _ = self._response_builder.header(name, value)
        return self

    def with_headers(self, headers: HeadersType) -> Self:
        """Add headers to the mocked response."""
        _ = self._response_builder.headers(headers)
        return self

    def with_body_bytes(self, body: bytes | bytearray | memoryview) -> Self:
        """Set the mocked response body to the given bytes."""
        _ = self._response_builder.body_bytes(body)
        return self

    def with_body_text(self, body: str) -> Self:
        """Set the mocked response body to the given text."""
        _ = self._response_builder.body_text(body)
        return self

    def with_body_json(self, json_body: object) -> Self:
        """Set the mocked response body to the given JSON-serializable object."""
        _ = self._response_builder.body_json(json_body)
        return self

    def with_version(self, version: str) -> Self:
        """Set the mocked response HTTP version."""
        _ = self._response_builder.version(version)
        return self

    def _handle_common_matchers(self, request: Request) -> dict[str, bool]:
        return {
            "method": self._matches_method(request),
            "url": self._matches_url(request),
            "path": self._matches_path(request),
            "query": self._match_query(request),
            "headers": self._match_headers(request),
            "body": self._match_body(request),
        }

    async def _handle(self, request: Request) -> Response | None:
        matches = self._handle_common_matchers(request)
        response = await self._handle_callbacks(request, matches)
        return self._check_matched(request, matches, response)

    def _handle_sync(self, request: Request) -> SyncResponse | None:
        matches = self._handle_common_matchers(request)
        response = self._handle_callbacks_sync(request, matches)
        return self._check_matched(request, matches, response)

    async def _handle_callbacks(self, request: Request, matches: dict[str, bool]) -> Response | None:
        matches["custom"] = await self._matches_custom(request)

        if self._custom_handler:
            response = await self._handle_custom_handler(request)
            matches["handler"] = response is not None
        else:
            response = await self._response()
            matches["handler"] = True

        return response if all(matches.values()) else None

    def _handle_callbacks_sync(self, request: Request, matches: dict[str, bool]) -> SyncResponse | None:
        matches["custom"] = self._matches_custom_sync(request)

        if self._custom_handler:
            response = self._handle_custom_handler_sync(request)
            matches["handler"] = response is not None
        else:
            response = self._response_sync()
            matches["handler"] = True

        return response if all(matches.values()) else None

    def _check_matched(self, request: Request, matches: dict[str, bool], response: _R | None) -> _R | None:
        if response is not None:
            self._matched_requests.append(request)
            return response

        from pyreqwest.pytest_plugin.internal.request_formatter import format_unmatched_request_parts

        # Memo the reprs as we may consume the request
        self._unmatched_requests_repr_parts.append(
            format_unmatched_request_parts(request, unmatched={k for k, matched in matches.items() if not matched}),
        )
        return None

    @cached_property
    def _response_builder(self) -> ResponseBuilder:
        assert self._custom_handler is None, "Cannot use response builder and custom handler together"
        self._using_response_builder = True
        return ResponseBuilder()

    async def _response(self) -> Response:
        built_response = await self._response_builder.copy().build()
        assert isinstance(built_response, Response)
        return built_response

    def _response_sync(self) -> SyncResponse:
        built_response = self._response_builder.copy().build_sync()
        assert isinstance(built_response, SyncResponse)
        return built_response

    def _matches_method(self, request: Request) -> bool:
        return self._method_matcher is None or self._method_matcher.matches(request.method)

    def _matches_url(self, request: Request) -> bool:
        return self._url_matcher is None or self._url_matcher.matches(request.url)

    def _matches_path(self, request: Request) -> bool:
        return self._path_matcher is None or self._path_matcher.matches(request.url.path)

    def _match_headers(self, request: Request) -> bool:
        for header_name, expected_value in self._header_matchers.items():
            actual_value = request.headers.get(header_name)
            if actual_value is None or not expected_value.matches(actual_value):
                return False
        return True

    def _match_body(self, request: Request) -> bool:
        if self._body_matcher is None:
            return True

        if request.body is None:
            return False

        assert request.body.get_stream() is None, "Stream should have been consumed into body bytes by mock middleware"
        body_buf = request.body.copy_bytes()
        assert body_buf is not None, "Unknown body type"
        body_bytes = body_buf.to_bytes()

        matcher, kind = self._body_matcher
        if kind == "json":
            try:
                body_json = json.loads(body_bytes)  # pyright: ignore[reportAny]
                return matcher.matches(body_json)  # pyright: ignore[reportAny]
            except json.JSONDecodeError:
                return False
        elif kind == "content":
            if isinstance(matcher.matcher, bytes):
                return matcher.matches(body_bytes)
            return matcher.matches(body_bytes.decode())
        else:
            assert_never(kind)

    def _match_query(self, request: Request) -> bool:
        if self._query_matcher is None:
            return True

        query_str = request.url.query_string or ""
        query_dict = request.url.query_dict_multi_value

        if isinstance(self._query_matcher, dict):
            for key, expected_value in self._query_matcher.items():
                actual_value: str | list[str] | None = query_dict.get(key)
                if actual_value is None or not expected_value.matches(actual_value):
                    return False
            return True
        if isinstance(self._query_matcher.matcher, str | Pattern):
            return self._query_matcher.matches(query_str)
        return self._query_matcher.matches(query_dict)

    async def _matches_custom(self, request: Request) -> bool:
        if self._custom_matcher is None:
            return True
        res = self._custom_matcher(request)
        assert isinstance(res, Awaitable)
        return await res

    def _matches_custom_sync(self, request: Request) -> bool:
        if self._custom_matcher is None:
            return True
        res = self._custom_matcher(request)
        assert isinstance(res, bool)
        return res

    async def _handle_custom_handler(self, request: Request) -> Response | None:
        assert self._custom_handler
        res = self._custom_handler(request)
        assert isinstance(res, Awaitable)
        return await res

    def _handle_custom_handler_sync(self, request: Request) -> SyncResponse | None:
        assert self._custom_handler
        res = self._custom_handler(request)
        assert res is None or isinstance(res, SyncResponse)
        return res

    def _assert_fail(
        self,
        *,
        count: int | None = None,
        min_count: int | None = None,
        max_count: int | None = None,
    ) -> None:
        """Internal method to raise assertion error with formatted message."""
        from pyreqwest.pytest_plugin.internal.assert_eq import assert_eq

        msg = self._format_counts_assert_message(count, min_count, max_count)

        if self._unmatched_requests_repr_parts:
            not_matched = {*self._unmatched_requests_repr_parts[-1].keys()}
            assert not_matched

            msg = f"{msg}. Diff with last unmatched request:"
            assert_eq(self._unmatched_requests_repr_parts[-1], self._format_mock_matchers_parts(not_matched), msg)
        else:
            raise AssertionError(msg)

    def _format_counts_assert_message(
        self,
        count: int | None = None,
        min_count: int | None = None,
        max_count: int | None = None,
    ) -> str:
        """Format the assertion message for call count mismatches."""
        if count is not None:
            expected_desc = f"exactly {count}"
        else:
            expectations = []
            if min_count is not None:
                expectations.append(f"at least {min_count}")
            if max_count is not None:
                expectations.append(f"at most {max_count}")
            expected_desc = " and ".join(expectations)

        method_path = " ".join(
            [
                self._method_matcher.matcher_repr if self._method_matcher is not None else "*",
                self._path_matcher.matcher_repr if self._path_matcher is not None else "*",
            ],
        )
        return f'Expected {expected_desc} request(s) but received {len(self._matched_requests)} to: "{method_path}"'

    def _format_mock_matchers_parts(self, unmatched: set[str] | None) -> dict[str, str | None]:
        """Format mock matcher parts for display in error messages."""
        parts: dict[str, str | None] = {
            "method": self._method_matcher.matcher_repr if self._method_matcher is not None else None,
            "path": self._path_matcher.matcher_repr if self._path_matcher is not None else None,
            "query": self._format_query_matcher() if self._query_matcher is not None else None,
            "headers": self._format_header_matchers() if self._header_matchers is not None else None,
            "body": self._format_body_matcher() if self._body_matcher is not None else None,
            "custom": f"Custom matcher: {self._custom_matcher.__name__}" if self._custom_matcher is not None else None,
            "handler": (
                f"Custom handler: {self._custom_handler.__name__}" if self._custom_handler is not None else None
            ),
        }
        return {k: v for k, v in parts.items() if unmatched is None or k in unmatched}

    def _format_query_matcher(self) -> str:
        """Format query matcher for display."""
        assert self._query_matcher is not None
        if isinstance(self._query_matcher, dict):
            query_parts = [f"{k}={v.matcher_repr}" for k, v in self._query_matcher.items()]
            return ", ".join(query_parts)
        return self._query_matcher.matcher_repr

    def _format_header_matchers(self) -> str:
        """Format header matchers for display."""
        header_parts = [f"{name.title()}: {value.matcher_repr}" for name, value in self._header_matchers.items()]
        return ", ".join(header_parts)

    def _format_body_matcher(self) -> str:
        """Format body matcher for display."""
        assert self._body_matcher is not None
        matcher, kind = self._body_matcher
        if kind == "json":
            try:
                return json.dumps(matcher.matcher, separators=(",", ":"))
            except (TypeError, ValueError):
                return matcher.matcher_repr
        elif kind == "content":
            return matcher.matcher_repr
        else:
            assert_never(kind)

    @override
    def __repr__(self) -> str:
        """Return a string representation of the mock for debugging purposes."""
        parts: list[str] = []
        if self._method_matcher is not None:
            parts.append(f"method={self._method_matcher!r}")
        if self._url_matcher is not None:
            parts.append(f"url={self._url_matcher!r}")
        if self._path_matcher is not None:
            parts.append(f"path={self._path_matcher!r}")
        if self._query_matcher is not None:
            parts.append(f"query={self._query_matcher!r}")
        if self._header_matchers:
            parts.append(f"headers={self._header_matchers!r}")
        if self._body_matcher is not None:
            matcher, kind = self._body_matcher
            parts.append(f"body.{kind}={matcher!r}")
        if self._custom_matcher is not None:
            parts.append(f"custom_matcher={self._custom_matcher!r}")
        if self._custom_handler is not None:
            parts.append(f"custom_handler={self._custom_handler!r}")
        return "<Mock " + ", ".join(parts) + ">"


class ClientMocker:
    """Main class for mocking HTTP requests.
    Use the `client_mocker` fixture or `ClientMocker.create_mocker` to create an instance.
    """

    def __init__(self) -> None:
        """@private"""
        self._mocks: list[Mock] = []
        self._strict: bool = False

    @staticmethod
    def create_mocker(monkeypatch: MonkeyPatch) -> "ClientMocker":
        """Create a ClientMocker for mocking HTTP requests in tests."""
        mocker = ClientMocker()

        def setup_async(klass: type[RequestBuilder]) -> None:
            orig_build_consumed = klass.build
            orig_build_streamed = klass.build_streamed

            def build_patch(
                self: RequestBuilder, orig: Callable[[RequestBuilder], ConsumedRequest | StreamRequest]
            ) -> ConsumedRequest | StreamRequest:
                return orig(self.with_middleware(mocker._create_middleware()))

            monkeypatch.setattr(
                klass,
                "build",
                lambda slf: build_patch(slf, orig_build_consumed),  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
            )
            monkeypatch.setattr(
                klass,
                "build_streamed",
                lambda slf: build_patch(slf, orig_build_streamed),  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
            )

        def setup_sync(klass: type[SyncRequestBuilder]) -> None:
            orig_build_consumed = klass.build
            orig_build_streamed = klass.build_streamed

            def build_patch(
                self: SyncRequestBuilder,
                orig: Callable[[SyncRequestBuilder], SyncConsumedRequest | SyncStreamRequest],
            ) -> SyncConsumedRequest | SyncStreamRequest:
                return orig(self.with_middleware(mocker._create_sync_middleware()))

            monkeypatch.setattr(
                klass,
                "build",
                lambda slf: build_patch(slf, orig_build_consumed),  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
            )
            monkeypatch.setattr(
                klass,
                "build_streamed",
                lambda slf: build_patch(slf, orig_build_streamed),  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
            )

        def setup_oneoff_async(klass: type[OneOffRequestBuilder]) -> None:
            orig_send = klass.send

            def send_patch(self: OneOffRequestBuilder) -> Awaitable[Response]:
                return orig_send(self.with_middleware(mocker._create_middleware()))

            monkeypatch.setattr(klass, "send", send_patch)

        def setup_oneoff_sync(klass: type[SyncOneOffRequestBuilder]) -> None:
            orig_send = klass.send

            def send_patch(self: SyncOneOffRequestBuilder) -> SyncResponse:
                return orig_send(self.with_middleware(mocker._create_sync_middleware()))

            monkeypatch.setattr(klass, "send", send_patch)

        setup_async(RequestBuilder)
        setup_sync(SyncRequestBuilder)
        setup_oneoff_async(OneOffRequestBuilder)
        setup_oneoff_sync(SyncOneOffRequestBuilder)

        return mocker

    def mock(
        self, method: MethodMatcher | None = None, *, path: PathMatcher | None = None, url: UrlMatcher | None = None
    ) -> Mock:
        """Add a mock rule for method and path or URL."""
        mock = Mock(method, path=path, url=url)
        self._mocks.append(mock)
        return mock

    def get(self, *, path: PathMatcher | None = None, url: UrlMatcher | None = None) -> Mock:
        """Mock GET requests to the given path or URL."""
        return self.mock("GET", path=path, url=url)

    def post(self, *, path: PathMatcher | None = None, url: UrlMatcher | None = None) -> Mock:
        """Mock POST requests to the given path or URL."""
        return self.mock("POST", path=path, url=url)

    def put(self, *, path: PathMatcher | None = None, url: UrlMatcher | None = None) -> Mock:
        """Mock PUT requests to the given path or URL."""
        return self.mock("PUT", path=path, url=url)

    def patch(self, *, path: PathMatcher | None = None, url: UrlMatcher | None = None) -> Mock:
        """Mock PATCH requests to the given path or URL."""
        return self.mock("PATCH", path=path, url=url)

    def delete(self, *, path: PathMatcher | None = None, url: UrlMatcher | None = None) -> Mock:
        """Mock DELETE requests to the given path or URL."""
        return self.mock("DELETE", path=path, url=url)

    def head(self, *, path: PathMatcher | None = None, url: UrlMatcher | None = None) -> Mock:
        """Mock HEAD requests to the given path or URL."""
        return self.mock("HEAD", path=path, url=url)

    def options(self, *, path: PathMatcher | None = None, url: UrlMatcher | None = None) -> Mock:
        """Mock OPTIONS requests to the given path or URL."""
        return self.mock("OPTIONS", path=path, url=url)

    def strict(self, enabled: bool = True) -> Self:
        """Enable strict mode - unmatched requests will raise an error."""
        self._strict = enabled
        return self

    def get_requests(self) -> list[Request]:
        """Get all captured requests in all mocks."""
        return [request for mock in self._mocks for request in mock.get_requests()]

    def get_call_count(self) -> int:
        """Get the total number of calls in all mocks."""
        return sum(mock.get_call_count() for mock in self._mocks)

    def clear(self) -> None:
        """Remove all mocks."""
        self._mocks.clear()

    def reset_requests(self) -> None:
        """Reset all captured requests in all mocks."""
        for mock in self._mocks:
            mock.reset_requests()

    def _create_middleware(self) -> Middleware:
        async def mock_middleware(request: Request, next_handler: Next) -> Response:
            if request.body is not None and (stream := request.body.get_stream()) is not None:
                assert isinstance(stream, AsyncIterable)
                body = [bytes(chunk) async for chunk in stream]  # Read the body stream into bytes
                request = request.from_request_and_body(request, RequestBody.from_bytes(b"".join(body)))

            for mock in self._mocks:
                # Call the protected _handle method - this is intentional for internal use
                if (response := await mock._handle(request)) is not None:  # pyright: ignore[reportPrivateUsage]
                    return response

            # No rule matched
            if self._strict:
                msg = f"No mock rule matched request: {request.method} {request.url}"
                raise AssertionError(msg)
            return await next_handler.run(request)  # Proceed normally

        return mock_middleware

    def _create_sync_middleware(self) -> SyncMiddleware:
        def mock_middleware(request: Request, next_handler: SyncNext) -> SyncResponse:
            if request.body is not None and (stream := request.body.get_stream()) is not None:
                assert isinstance(stream, Iterable)
                body = [bytes(chunk) for chunk in stream]  # Read the body stream into bytes
                request = request.from_request_and_body(request, RequestBody.from_bytes(b"".join(body)))

            for mock in self._mocks:
                # Call the protected _handle_sync method - this is intentional for internal use
                if (response := mock._handle_sync(request)) is not None:  # pyright: ignore[reportPrivateUsage]
                    return response

            # No rule matched
            if self._strict:
                msg = f"No mock rule matched request: {request.method} {request.url}"
                raise AssertionError(msg)
            return next_handler.run(request)  # Proceed normally

        return mock_middleware


if "pytest" in globals():

    @pytest.fixture
    def client_mocker(monkeypatch: pytest.MonkeyPatch) -> "ClientMocker":
        """Fixture that provides a ClientMocker for mocking HTTP requests in tests."""
        return ClientMocker.create_mocker(monkeypatch)
