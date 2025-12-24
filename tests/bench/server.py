import ssl
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import trustme
from granian.constants import HTTPModes
from pyreqwest.http import Url

from tests.servers.echo_server import EchoServer
from tests.servers.server import EmbeddedServer, ServerConfig, find_free_port


@asynccontextmanager
async def server() -> AsyncGenerator[tuple[Url, bytes], None]:
    ca = trustme.CA()
    cert_der = ssl.PEM_cert_to_DER_cert(ca.cert_pem.bytes().decode())
    cert = ca.issue_cert("127.0.0.1", "localhost")
    with (
        cert.cert_chain_pems[0].tempfile() as cert_tmp,
        cert.private_key_pem.tempfile() as pk_tmp,
        ca.cert_pem.tempfile() as ca_tmp,
    ):
        config = ServerConfig(ssl_cert=Path(cert_tmp), ssl_key=Path(pk_tmp), ssl_ca=Path(ca_tmp), http=HTTPModes.http1)
        port = find_free_port()

        async with EmbeddedServer(EchoServer(), port, config).serve_context() as echo_server:
            yield echo_server.url, cert_der
