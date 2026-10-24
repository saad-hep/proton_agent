"""Expose proton_agent as an OpenAI-compatible service for Open WebUI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

# This service renders files from worker threads and never owns a GUI event
# loop. Select the non-interactive backend before importing proton_agent;
# doing this in main() is too late once Tk/pyplot has been initialized.
os.environ["PROTON_AGENT_SERVER_MODE"] = "1"
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg", force=True)

from langchain_core.messages import AIMessage, HumanMessage

from proton_agent import OUTPUT_DIR, ask, _parse_mcmc_request, _code_free_answer
from proton_agent import (_requested_bounds_filepath, _requested_matrix_elements_filepath,
                          _explicit_filepath_from_text, _resolve_project_path)


FULL_MODEL_ID = "proton-agent"
MODEL_IDS = (FULL_MODEL_ID,)


def _initial_progress_content(question: str) -> str:
    """Describe a sampling request without claiming a plot started sampling."""
    if _parse_mcmc_request(question) is not None:
        return "Processing the MCMC request; the numerical report will follow when sampling completes.\n\n"
    return ""

BACKGROUND_TASK_RESPONSES = {
    "title_generation": {"title": "Proton Agent"},
    "follow_up_generation": {"follow_ups": []},
    "tags_generation": {"tags": ["Science", "Particle Physics"]},
    "query_generation": {"queries": []},
    "autocomplete_generation": {"completion": ""},
}


def _ollama_running_models_payload() -> dict[str, Any]:
    """Return the small Ollama-compatible status payload Open WebUI probes."""
    return {
        "models": [{
            "name": FULL_MODEL_ID,
            "model": FULL_MODEL_ID,
            "size": 0,
            "digest": "proton-agent",
            "details": {
                "family": "proton-agent",
                "parameter_size": "",
                "quantization_level": "",
            },
        }]
    }


def _background_task_answer(task: Any) -> str | None:
    """Return inert metadata-task output so Open WebUI cannot invoke physics tools."""
    normalized = str(task or "").strip().lower()
    if not normalized:
        return None
    payload = BACKGROUND_TASK_RESPONSES.get(normalized, {})
    return json.dumps(payload, separators=(",", ":"))


def _background_task_from_messages(messages: list[dict[str, Any]]) -> str | None:
    """Recognize Open WebUI task templates when provider metadata is stripped."""
    content = "\n".join(
        _message_text(message.get("content"))
        for message in messages
        if message.get("role") == "user"
    ).lower()
    signatures = {
        "generate a concise title summarizing the chat history": "title_generation",
        "suggest 3-5 relevant follow-up questions": "follow_up_generation",
        "generate 1-3 broad tags categorizing the main themes": "tags_generation",
    }
    for signature, task in signatures.items():
        if signature in content:
            return task
    return None


def _request_fingerprint(model: str, messages: list[dict[str, Any]], scope: str | None = None) -> str:
    """Deduplicate retries while preserving prior turns and selected input data.

    Ignore assistant placeholders appended *after* the latest user message.
    Earlier messages identify the actual calculation or scan being discussed.
    """
    latest_user = max(
        (index for index, message in enumerate(messages) if message.get("role") == "user"),
        default=len(messages) - 1,
    )
    latest_question = (
        _message_text(messages[latest_user].get("content")).strip()
        if messages and latest_user >= 0
        else ""
    )
    identity = {"model": model, "latest_user_question": latest_question}
    identity["history"] = [
        {"role": message.get("role"), "content": _message_text(message.get("content")).strip()}
        for message in messages[:latest_user]
        if message.get("role") in {"user", "assistant"}
    ]
    if scope:
        identity["scope"] = scope
    optional_files = {}
    for select in (_requested_bounds_filepath, _requested_matrix_elements_filepath,
                   _explicit_filepath_from_text):
        filepath = select(latest_question)
        if filepath:
            try:
                path = _resolve_project_path(filepath, must_exist=True)
                if path.suffix.lower() in {".txt", ".json", ".yaml", ".yml"}:
                    optional_files[filepath] = hashlib.sha256(path.read_bytes()).hexdigest()
                else:
                    status = path.stat()
                    optional_files[filepath] = [str(path), status.st_size, status.st_mtime_ns]
            except OSError as error:
                optional_files[filepath] = f"unreadable:{type(error).__name__}"
    if optional_files:
        identity["optional_input_files"] = optional_files
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _message_text(content: Any) -> str:
    """Extract text from OpenAI string or multipart message content."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")

    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") in {"text", "input_text"}:
            text = item.get("text")
            if text:
                parts.append(str(text))
    return "\n".join(parts)


_PLOT_PATH_RE = re.compile(
    r"`(?P<quoted>[^`\r\n]+\.png)`|"
    r"(?P<bare>(?:/[^\s`'\"]+|output/[^\s`'\"]+)\.png)",
    flags=re.IGNORECASE,
)


def _plot_path_from_url_path(url_path: str) -> Path | None:
    """Resolve a /plots URL beneath OUTPUT_DIR without allowing traversal."""
    prefix = "/plots/"
    if not url_path.startswith(prefix):
        return None
    relative = Path(unquote(url_path[len(prefix):]))
    if relative.is_absolute():
        return None
    output_root = OUTPUT_DIR.resolve()
    candidate = (output_root / relative).resolve()
    if candidate.suffix.lower() != ".png" or not candidate.is_relative_to(output_root):
        return None
    return candidate


def _answer_with_inline_plot_links(answer: str, host: str) -> str:
    """Append renderable Markdown images for verified local PNG paths."""
    safe_host = host.strip()
    if not re.fullmatch(r"[A-Za-z0-9.\-:\[\]]+", safe_host):
        safe_host = "127.0.0.1:8001"
    images: list[str] = []
    seen: set[Path] = set()
    output_root = OUTPUT_DIR.resolve()
    for match in _PLOT_PATH_RE.finditer(answer):
        raw = match.group("quoted") or match.group("bare")
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if candidate in seen or not candidate.exists() or not candidate.is_relative_to(output_root):
            continue
        seen.add(candidate)
        relative = candidate.relative_to(output_root)
        url = f"http://{safe_host}/plots/{quote(relative.as_posix(), safe='/')}"
        images.append(f"![Generated plot]({url})")
    if not images:
        return answer
    return answer.rstrip() + "\n\n" + "\n\n".join(images)


def _request_context(messages: list[dict[str, Any]]) -> tuple[str, list[Any]]:
    """Return the latest user question and preceding user/assistant history."""
    latest_user = -1
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            latest_user = index
            break

    if latest_user < 0:
        raise ValueError("The request must contain at least one user message.")

    question = _message_text(messages[latest_user].get("content")).strip()
    if not question:
        raise ValueError("The latest user message is empty or unsupported.")

    history: list[Any] = []
    for message in messages[:latest_user]:
        role = message.get("role")
        content = _message_text(message.get("content")).strip()
        if not content:
            continue
        if role == "user":
            history.append(HumanMessage(content=content))
        elif role == "assistant":
            history.append(AIMessage(content=content))

    return question, history


class ProtonAgentHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        upstream_model: str,
        api_key: str | None,
        upstream_api_base: str | None = None,
        upstream_api_key: str | None = None,
        upstream_api_email: str | None = None,
        upstream_api_password: str | None = None,
    ):
        super().__init__(server_address, ProtonAgentRequestHandler)
        self.upstream_model = upstream_model
        self.api_key = api_key
        self.upstream_api_base = upstream_api_base
        self.upstream_api_key = upstream_api_key
        self.upstream_api_email = upstream_api_email
        self.upstream_api_password = upstream_api_password
        # proton_agent currently stores graph and physics result state globally.
        self.agent_lock = threading.Lock()
        # A streaming client can retry a slow local-model request before the
        # first response finishes.  Share the first result instead of running
        # the same side-effecting tool workflow again.
        self.request_cache_lock = threading.Lock()
        self.request_cache: dict[str, dict[str, Any]] = {}
        self.request_cache_ttl = float(
            os.environ.get("PROTON_AGENT_REQUEST_CACHE_TTL", "600")
        )

    def request_entry(self, fingerprint: str, *, reuse_completed: bool = False) -> tuple[dict[str, Any], bool]:
        now = time.monotonic()
        with self.request_cache_lock:
            expired = [
                key
                for key, entry in self.request_cache.items()
                if entry.get("finished_at") is not None
                and now - float(entry["finished_at"]) > self.request_cache_ttl
            ]
            for key in expired:
                del self.request_cache[key]

            entry = self.request_cache.get(fingerprint)
            if entry is not None and entry.get("finished_at") is not None and (
                not reuse_completed or entry.get("error") is not None
            ):
                # Without a client conversation identity, identical text may
                # be a new chat or a deliberate fresh request, not a retry.
                entry = None
            if entry is not None:
                return entry, False

            entry = {
                "event": threading.Event(),
                "answer": None,
                "error": None,
                "finished_at": None,
            }
            self.request_cache[fingerprint] = entry
            return entry, True

    def complete_request(
        self,
        entry: dict[str, Any],
        *,
        answer: str | None = None,
        error: BaseException | None = None,
    ) -> None:
        with self.request_cache_lock:
            entry["answer"] = _code_free_answer(answer) if answer is not None else None
            entry["error"] = error
            entry["finished_at"] = time.monotonic()
            entry["event"].set()


class ProtonAgentRequestHandler(BaseHTTPRequestHandler):
    server: ProtonAgentHTTPServer
    protocol_version = "HTTP/1.1"

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Open WebUI may cancel or time out while the local LLM is still
            # working. The completed request must release agent_lock cleanly
            # without emitting a socketserver traceback.
            self.close_connection = True

    def _write_stream_bytes(self, *chunks: bytes) -> bool:
        """Write SSE chunks, returning False when the client has disconnected."""
        try:
            for chunk in chunks:
                self.wfile.write(chunk)
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True
            return False

    def _error(self, status: int, message: str) -> None:
        self._send_json(
            status,
            {"error": {"message": message, "type": "proton_agent_error"}},
        )

    def _authorized(self) -> bool:
        if not self.server.api_key:
            return True
        header = self.headers.get("Authorization", "")
        supplied = header[7:] if header.lower().startswith("bearer ") else ""
        return secrets.compare_digest(supplied, self.server.api_key)

    def _require_auth(self) -> bool:
        if self._authorized():
            return True
        self._error(HTTPStatus.UNAUTHORIZED, "Missing or invalid bearer token.")
        return False

    def do_GET(self) -> None:
        path = urlsplit(self.path).path.rstrip("/")
        if path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        if path.startswith("/plots/"):
            plot_path = _plot_path_from_url_path(path)
            if plot_path is None or not plot_path.is_file():
                self._error(HTTPStatus.NOT_FOUND, "Plot file not found.")
                return
            data = plot_path.read_bytes()
            try:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Disposition", f'inline; filename="{plot_path.name}"')
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                self.close_connection = True
            return
        if path in {"/models", "/v1/models"}:
            if not self._require_auth():
                return
            self._send_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": model_id,
                            "object": "model",
                            "created": 0,
                            "owned_by": "local",
                        }
                        for model_id in MODEL_IDS
                    ],
                },
            )
            return
        if path in {"/api/ps", "/v1/api/ps"}:
            if not self._require_auth():
                return
            self._send_json(HTTPStatus.OK, _ollama_running_models_payload())
            return
        self._error(HTTPStatus.NOT_FOUND, f"Unknown endpoint: {self.path}")

    def do_POST(self) -> None:
        path = self.path.rstrip("/")
        if path not in {"/chat/completions", "/v1/chat/completions"}:
            self._error(HTTPStatus.NOT_FOUND, f"Unknown endpoint: {self.path}")
            return
        if not self._require_auth():
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(content_length) or b"{}")
            messages = body.get("messages")
            if not isinstance(messages, list):
                raise ValueError("The 'messages' field must be a list.")
            requested_model = str(body.get("model") or FULL_MODEL_ID)
            if requested_model not in MODEL_IDS:
                raise ValueError(
                    f"Unknown model {requested_model!r}; choose one of: "
                    f"{', '.join(MODEL_IDS)}."
                )
            metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
            background_task = metadata.get("task") or _background_task_from_messages(messages)
            background_answer = _background_task_answer(background_task)
            if background_answer is None:
                question, history = _request_context(messages)
            else:
                question, history = "", []
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
            return

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        stream = bool(body.get("stream", False))

        if background_answer is not None:
            print(f"[BACKGROUND TASK] {background_task}: physics agent bypassed")
            if stream:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": requested_model,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": background_answer},
                        "finish_reason": "stop",
                    }],
                }
                self._write_stream_bytes(
                    f"data: {json.dumps(chunk)}\n\n".encode(),
                    b"data: [DONE]\n\n",
                )
                return
            self._send_json(
                HTTPStatus.OK,
                {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": requested_model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": background_answer},
                        "finish_reason": "stop",
                    }],
                },
            )
            return

        if stream:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

        scope_values = {key: body.get(key) or metadata.get(key)
                        for key in ("chat_id", "conversation_id", "session_id", "message_id")}
        scope_values = {key: value for key, value in scope_values.items() if value}
        idempotency_key = self.headers.get("Idempotency-Key")
        if idempotency_key:
            scope_values["idempotency_key"] = idempotency_key
        scope = json.dumps(scope_values, sort_keys=True, default=str) if scope_values else None
        fingerprint = _request_fingerprint(requested_model, messages, scope)
        reuse_completed = any(key in scope_values for key in (
            "chat_id", "conversation_id", "message_id", "idempotency_key"))
        entry, leader = self.server.request_entry(fingerprint, reuse_completed=reuse_completed)

        if leader:
            def execute_request() -> None:
                try:
                    with self.server.agent_lock:
                        answer, _ = ask(
                            question,
                            conversation_history=history,
                            verbose=False,
                            model=self.server.upstream_model,
                            api_base=self.server.upstream_api_base,
                            api_key=self.server.upstream_api_key,
                            api_email=self.server.upstream_api_email,
                            api_password=self.server.upstream_api_password,
                        )
                    self.server.complete_request(entry, answer=str(answer))
                except BaseException as error:
                    self.server.complete_request(entry, error=error)

            threading.Thread(target=execute_request, daemon=True).start()

        if stream:
            initial_content = _initial_progress_content(question)
            initial_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": requested_model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": initial_content},
                        "finish_reason": None,
                    }
                ],
            }
            if not self._write_stream_bytes(
                f"data: {json.dumps(initial_chunk)}\n\n".encode()
            ):
                return

        while not entry["event"].wait(timeout=10.0):
            if stream and not self._write_stream_bytes(b": keep-alive\n\n"):
                return

        error = entry.get("error")
        if error is not None:
            safe_error = _code_free_answer(str(error))
            if stream:
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": requested_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": f"[Proton Agent error] {safe_error}"},
                            "finish_reason": "stop",
                        }
                    ],
                }
                self._write_stream_bytes(
                    f"data: {json.dumps(chunk)}\n\n".encode(),
                    b"data: [DONE]\n\n",
                )
                return
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, safe_error)
            return

        answer = _code_free_answer(str(entry.get("answer") or ""))
        answer = _answer_with_inline_plot_links(
            answer,
            self.headers.get("Host") or f"127.0.0.1:{self.server.server_port}",
        )

        if stream:
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": requested_model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": answer},
                        "finish_reason": None,
                    }
                ],
            }
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": requested_model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            self._write_stream_bytes(
                f"data: {json.dumps(chunk)}\n\n".encode(),
                f"data: {json.dumps(final_chunk)}\n\n".encode(),
                b"data: [DONE]\n\n",
            )
            return

        self._send_json(
            HTTPStatus.OK,
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": requested_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[HTTP] {self.address_string()} - {fmt % args}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="proton_agent_server",
        description="OpenAI-compatible HTTP wrapper for proton_agent.",
    )
    parser.add_argument("--host", default=os.environ.get("PROTON_AGENT_SERVER_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PROTON_AGENT_SERVER_PORT", "8001")),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("PROTON_AGENT_SERVER_API_KEY"),
        help="Bearer token Open WebUI must send to this service.",
    )
    parser.add_argument(
        "--upstream-model",
        default=os.environ.get("PROTON_AGENT_UPSTREAM_MODEL", "ollama:gpt-oss:20b"),
        help="Model used internally by proton_agent.",
    )
    parser.add_argument("--upstream-api-base", default=os.environ.get("PROTON_AGENT_UPSTREAM_API_BASE"))
    parser.add_argument("--upstream-api-key", default=os.environ.get("PROTON_AGENT_UPSTREAM_API_KEY"))
    parser.add_argument("--upstream-api-email", default=os.environ.get("PROTON_AGENT_UPSTREAM_API_EMAIL"))
    parser.add_argument("--upstream-api-password", default=os.environ.get("PROTON_AGENT_UPSTREAM_API_PASSWORD"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    os.environ["PROTON_AGENT_SERVER_MODE"] = "1"
    server = ProtonAgentHTTPServer(
        (args.host, args.port),
        upstream_model=args.upstream_model,
        api_key=args.api_key,
        upstream_api_base=args.upstream_api_base,
        upstream_api_key=args.upstream_api_key,
        upstream_api_email=args.upstream_api_email,
        upstream_api_password=args.upstream_api_password,
    )

    print(f"Proton Agent API listening on http://{args.host}:{args.port}/v1")
    print(f"Published models: {', '.join(MODEL_IDS)}")
    print(f"Upstream model: {args.upstream_model}")
    if not args.api_key:
        print("WARNING: API authentication is disabled. Set PROTON_AGENT_SERVER_API_KEY.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Proton Agent API.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
