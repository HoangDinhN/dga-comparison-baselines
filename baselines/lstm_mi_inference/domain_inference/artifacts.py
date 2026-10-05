from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import unquote, urlparse

from domain_inference.config import nested, require

LOGGER = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ArtifactRegistry:
    """Allow-listed model files served by the controller."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.public_base_url = str(require(config, "artifact_server.public_base_url")).rstrip("/")
        self.token = str(require(config, "artifact_server.token"))
        self.files: dict[tuple[str, str], Path] = {}
        self._metadata: dict[tuple[str, str], dict[str, Any]] = {}
        model_artifacts = nested(config, "model_artifacts", {})
        for role, entries in model_artifacts.items():
            if not isinstance(entries, dict):
                continue
            for logical_name, raw_path in entries.items():
                if raw_path in (None, ""):
                    continue
                path = Path(str(raw_path)).expanduser().resolve()
                self.files[(str(role), str(logical_name))] = path

    def validate_role(self, role: str) -> None:
        files = [(name, path) for (item_role, name), path in self.files.items() if item_role == role]
        if not files:
            raise FileNotFoundError(f"No model_artifacts configured for role {role!r}")
        missing = [str(path) for _, path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing artifacts for {role}: {', '.join(missing)}")

    def manifest(self, role: str) -> list[dict[str, Any]]:
        self.validate_role(role)
        result = []
        for (item_role, name), path in sorted(self.files.items()):
            if item_role != role:
                continue
            key = (role, name)
            metadata = self._metadata.get(key)
            stat = path.stat()
            if metadata is None or metadata["mtime_ns"] != stat.st_mtime_ns or metadata["size"] != stat.st_size:
                metadata = {
                    "name": name,
                    "size": stat.st_size,
                    "sha256": sha256_file(path),
                    "mtime_ns": stat.st_mtime_ns,
                }
                self._metadata[key] = metadata
            result.append(
                {
                    "name": name,
                    "size": metadata["size"],
                    "sha256": metadata["sha256"],
                    "url": f"{self.public_base_url}/artifacts/{role}/{name}",
                }
            )
        return result

    def resolve(self, role: str, name: str) -> Path | None:
        return self.files.get((role, name))


class ArtifactServer:
    def __init__(self, registry: ArtifactRegistry, host: str, port: int):
        self.registry = registry
        self.host = host
        self.port = port
        self.httpd: ThreadingHTTPServer | None = None
        self.thread: Thread | None = None

    def start(self) -> None:
        registry = self.registry

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                if self.headers.get("Authorization") != f"Bearer {registry.token}":
                    self.send_error(HTTPStatus.UNAUTHORIZED)
                    return
                parts = [unquote(part) for part in urlparse(self.path).path.split("/") if part]
                if len(parts) != 3 or parts[0] != "artifacts":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                path = registry.resolve(parts[1], parts[2])
                if path is None or not path.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                stat = path.stat()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(stat.st_size))
                self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
                self.end_headers()
                with path.open("rb") as handle:
                    shutil.copyfileobj(handle, self.wfile, length=1024 * 1024)

            def log_message(self, fmt, *args):
                LOGGER.debug("artifact-http: " + fmt, *args)

        self.httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self.thread = Thread(target=self.httpd.serve_forever, name="artifact-server", daemon=True)
        self.thread.start()
        LOGGER.info("Artifact server listening on %s:%d", self.host, self.port)

    def stop(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)


@dataclass
class DownloadedArtifacts:
    temporary_directory: tempfile.TemporaryDirectory
    paths: dict[str, str]

    def cleanup(self) -> None:
        self.temporary_directory.cleanup()


def download_artifacts(
    manifest: list[dict[str, Any]],
    token: str,
    timeout_seconds: float = 120,
) -> DownloadedArtifacts:
    temporary = tempfile.TemporaryDirectory(prefix="dga-model-")
    paths: dict[str, str] = {}
    try:
        for item in manifest:
            name = str(item["name"])
            destination = Path(temporary.name, Path(name).name)
            request = urllib.request.Request(
                str(item["url"]),
                headers={"Authorization": f"Bearer {token}"},
            )
            LOGGER.info("Downloading artifact %s (%s bytes)", name, item.get("size", "?"))
            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response, destination.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            except urllib.error.URLError as exc:
                raise RuntimeError(f"Cannot download artifact {name}: {exc}") from exc
            expected_size = int(item["size"])
            if destination.stat().st_size != expected_size:
                raise RuntimeError(f"Artifact size mismatch for {name}")
            if sha256_file(destination) != item["sha256"]:
                raise RuntimeError(f"Artifact checksum mismatch for {name}")
            paths[name] = str(destination)
        return DownloadedArtifacts(temporary, paths)
    except Exception:
        temporary.cleanup()
        raise

