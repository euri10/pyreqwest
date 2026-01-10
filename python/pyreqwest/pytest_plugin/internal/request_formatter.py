"""Utilities for formatting request information for error messages."""

from pyreqwest.request import Request


def format_unmatched_request_parts(request: Request, unmatched: set[str]) -> dict[str, str | None]:
    """Format request parts for display in error messages."""
    req_parts: dict[str, str | None] = {
        "method": request.method,
        "url": str(request.url),
        "path": request.url.path,
        "query": None,
        "headers": None,
        "body": None,
    }

    if request.url.query_pairs:
        query_parts = [f"{k}={v}" for k, v in request.url.query_pairs]
        req_parts["query"] = ", ".join(query_parts)

    if request.headers:
        header_parts = [f"{name.title()}: {value}" for name, value in request.headers.items()]
        req_parts["headers"] = ", ".join(header_parts)

    if request.body:
        if (bytes_body := request.body.copy_bytes()) is not None:
            req_parts["body"] = bytes_body.to_bytes().decode("utf8", errors="replace")
        elif (stream_body := request.body.get_stream()) is not None:
            req_parts["body"] = repr(stream_body)
        else:
            req_parts["body"] = repr(request.body)

    fmt_parts: dict[str, str | None] = {
        "custom": f"No match with request {req_parts}",
        "handler": f"No match with request {req_parts}",
    }
    fmt_parts = {**req_parts, **fmt_parts}

    return {k: v for k, v in fmt_parts.items() if k in unmatched}
