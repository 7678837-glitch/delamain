"""
delamain Gateway - query-string redaction for log records (SAF-90).

Credentials must never end up in the gateway's logs, but two loggers write full
request URLs, query string included:

* ``uvicorn.access`` — one line per HTTP request, e.g.
  ``"GET /transfer/status?token=... HTTP/1.1" 400``. Rejecting ``?token=`` with a
  4xx (issue #8) does not help here: the access line is written for the rejected
  request too.
* ``httpx`` — at INFO it logs every outgoing request URL
  (``HTTP Request: GET http://127.0.0.1:8650/...?token=...``), i.e. whatever the
  gateway forwards to the Java backend.

``QueryParamRedactionFilter`` rewrites the URL-ish arguments of a record so that
the *value* of every sensitive URL parameter is replaced by ``***`` — including the
odd spellings that the 400 ``token_in_query_not_allowed`` check does not catch
(``TOKEN=``, ``token[]=``, ``;token=``, ``%74oken=``, repeated parameters; see
``redact_query_params``); the path,
the parameter names, their order and every other parameter stay byte-for-byte
as they were, so the line is still useful for troubleshooting. It is attached by
``install_log_redaction()`` (root handlers + ``uvicorn.access``) and by
``uvicorn_log_config()`` (the dictConfig uvicorn applies when it starts), both
called from ``gateway/main.py``.
"""

from __future__ import annotations

import copy
import logging
import re
from typing import Any
from urllib.parse import unquote_plus

# Query parameter names (compared case-insensitively, after percent-decoding)
# whose values are masked. Deliberately short:
#   token          - the transfer token (issue #8) and the usual name clients pick
#                    for a bearer credential in a URL.
#   access_token   - RFC 6750 §2.3 / OAuth bearer token in a URI query.
#   api_key, apikey- common API-key parameter spellings.
#   password       - never legitimate in a URL, mask if someone does it anyway.
#   sig, signature - pre-signed URL signatures (S3/Azure SAS style) that grant
#                    access on their own.
#   client_secret  - OAuth client secret sent as a query parameter.
# Bare ``key`` is intentionally NOT in the list: the Java backend's resource
# routes take ``key=<Android resource name>`` (ResourceRoutes.java), which is not
# a secret and is useful in the gateway->backend httpx log lines.
SENSITIVE_QUERY_PARAMS = frozenset(
    {
        "token",
        "access_token",
        "api_key",
        "apikey",
        "password",
        "sig",
        "signature",
        "client_secret",
    }
)

MASK = "***"


# "token[]" / "token[0]" (PHP/Rails-style array params) count as "token".
_ARRAY_SUFFIX_RE = re.compile(r"\[[^\]]*\]$")
# Separators between query parameters: "&" and, for servers/clients that honour
# it, ";" (W3C HTML 4 recommendation / older urllib behaviour).
_QUERY_SPLIT_RE = re.compile(r"([&;])")
# Matrix parameters inside the path: "/a;token=x/b" or "/a;token=x?..." (RFC 3986
# §3.3 allows ";" in segments; some frameworks parse them as parameters).
# uvicorn's access log re-quotes the path (urllib.parse.quote(scope["path"])), so
# there the same parameter shows up as "/a%3Btoken%3Dx/b": both the literal and
# the percent-encoded ";" / "=" are recognised.
_MATRIX_PARAM_RE = re.compile(
    r"(;|%3[Bb])"                          # 1: separator
    r"((?:(?!%3[BbDd])[^;/?#=])*)"         # 2: name
    r"(=|%3[Dd])"                          # 3: equals
    r"((?:(?!%3[Bb])[^;/?#])*)"            # 4: value, up to the next ; / ? #
)


def _normalise_name(name: str) -> str:
    # Decode until stable (bounded): uvicorn's access log re-quotes the path, so a
    # matrix parameter name sent as "%74oken" is logged as "%2574oken".
    decoded = name
    for _ in range(3):
        nxt = unquote_plus(decoded)
        if nxt == decoded:
            break
        decoded = nxt
    return _ARRAY_SUFFIX_RE.sub("", decoded.strip().lower()).strip()


def _is_sensitive(name: str) -> bool:
    return _normalise_name(name) in SENSITIVE_QUERY_PARAMS


def _redact_query(query: str) -> str:
    parts = _QUERY_SPLIT_RE.split(query)
    # even indexes are params, odd indexes are the separators that were between them
    for i in range(0, len(parts), 2):
        name, eq, value = parts[i].partition("=")
        if eq and value and _is_sensitive(name):
            parts[i] = f"{name}={MASK}"
    return "".join(parts)


def _redact_matrix(path: str) -> str:
    def repl(m: re.Match[str]) -> str:
        sep, name, eq, value = m.groups()
        if value and _is_sensitive(name):
            return f"{sep}{name}{eq}{MASK}"
        return m.group(0)

    return _MATRIX_PARAM_RE.sub(repl, path)


def redact_query_params(text: str) -> str:
    """Return ``text`` with the values of sensitive URL parameters replaced by
    ``***``.

    ``text`` is a request target (``/path?a=1&token=x``) or a full URL. Parameter
    names are matched case-insensitively after URL-decoding, with an array suffix
    (``token[]``, ``token[0]``) ignored, so ``TOKEN=``, ``Token=``, ``%74oken=``,
    ``token%5B%5D=`` all match. Covered forms:

    * query parameters separated by ``&`` or ``;`` — every occurrence of a repeated
      parameter is masked;
    * matrix parameters in the path (``/a;token=x/b``), also in the re-quoted
      form uvicorn's access log prints (``/a%3Btoken%3Dx/b``).

    Only ``name=value`` pairs with a non-empty value are rewritten (to
    ``name=***``, keeping the name exactly as it was written); the path, other
    parameters, separators and any ``#fragment`` are kept byte-for-byte. A string
    with nothing to mask is returned unchanged (the same object).
    """
    if "?" not in text and ";" not in text and "%3" not in text:
        return text
    head, qmark, rest = text.partition("?")
    query, hash_sep, fragment = rest.partition("#")
    if not qmark:
        # no query string: a "#fragment" may still follow the path
        head, hash_sep, fragment = text.partition("#")
    new_head = _redact_matrix(head) if (";" in head or "%3" in head) else head
    new_query = _redact_query(query) if query else query
    if new_head == head and new_query == query:
        return text
    return new_head + qmark + new_query + hash_sep + fragment


def _redact_arg(arg: Any) -> Any:
    if isinstance(arg, str):
        return redact_query_params(arg)
    # httpx.URL / starlette.datastructures.URL: format them as their string form
    # (exactly what "%s" would print) so the query can be redacted. Other types
    # are left alone — calling str() on arbitrary objects is not free.
    if type(arg).__name__ == "URL":
        try:
            as_text = str(arg)
        except Exception:
            return arg
        redacted = redact_query_params(as_text)
        return arg if redacted == as_text else redacted
    return arg


class QueryParamRedactionFilter(logging.Filter):
    """logging.Filter that masks sensitive query-parameter values in a record.

    * ``record.args`` (tuple): every ``str``/URL argument is redacted; the tuple
      keeps its length and the position of every element, which uvicorn's
      AccessFormatter relies on (it unpacks exactly five args).
    * ``record.msg`` is redacted only when the record has no args (a pre-formatted
      message such as an f-string). With args, ``msg`` is a format string and
      rewriting it could break %-formatting.

    Never drops a record and never raises: on any unexpected error the record is
    passed through as it is.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            args = record.args
            if isinstance(args, tuple) and args:
                new_args = tuple(_redact_arg(a) for a in args)
                if any(n is not o for n, o in zip(new_args, args)):
                    record.args = new_args
            elif not args and isinstance(record.msg, str):
                record.msg = redact_query_params(record.msg)
        except Exception:  # pragma: no cover - defensive, logging must not fail
            pass
        return True


def _has_redaction_filter(filterer: logging.Filterer) -> bool:
    return any(isinstance(f, QueryParamRedactionFilter) for f in filterer.filters)


def install_log_redaction() -> None:
    """Attach the redaction filter to every handler on the root logger (covers
    ``httpx`` and all ``delamain.*`` loggers, which propagate to root) and to the
    ``uvicorn.access`` logger itself (covers any handler a deployment attaches to
    it). Idempotent."""
    for handler in logging.getLogger().handlers:
        if not _has_redaction_filter(handler):
            handler.addFilter(QueryParamRedactionFilter())
    access = logging.getLogger("uvicorn.access")
    if not _has_redaction_filter(access):
        access.addFilter(QueryParamRedactionFilter())


def uvicorn_log_config() -> dict:
    """uvicorn's own default ``LOGGING_CONFIG`` (same handlers, formatters, streams
    and levels) with the redaction filter added to the ``access`` and ``default``
    handlers and to the ``uvicorn.access`` logger.

    Passed to uvicorn as ``log_config`` so the filter survives uvicorn's
    ``dictConfig()`` at server start instead of depending on it leaving
    pre-existing filters alone."""
    from uvicorn.config import LOGGING_CONFIG

    config = copy.deepcopy(LOGGING_CONFIG)
    config.setdefault("filters", {})["delamain_redact_query"] = {"()": QueryParamRedactionFilter}
    for handler_name in ("default", "access"):
        handler = config["handlers"][handler_name]
        handler["filters"] = [*handler.get("filters", []), "delamain_redact_query"]
    access_logger = config["loggers"]["uvicorn.access"]
    access_logger["filters"] = [*access_logger.get("filters", []), "delamain_redact_query"]
    return config
