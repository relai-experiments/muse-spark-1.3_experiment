# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Semantic safety classification for API requests."""

from __future__ import annotations

import json
import re
from urllib.parse import unquote, urlsplit

_READ_ONLY_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_QUICKBOOKS_HOSTS = frozenset(
    {
        "quickbooks.api.intuit.com",
        "sandbox-quickbooks.api.intuit.com",
    }
)
_QUICKBOOKS_QUERY_PATH = re.compile(r"^/v3/company/[^/]+/query/?$", re.IGNORECASE)
_GRAPHQL_LEADING_TRIVIA = re.compile(r"^(?:\ufeff|\s|#[^\r\n]*(?:\r?\n|$))*")
_GRAPHQL_MUTATION_TOKEN = re.compile(r"\bmutation\b", re.IGNORECASE)
_GRAPHQL_OPERATION = re.compile(r"^(query|mutation|subscription)\b|^(\{)", re.IGNORECASE)


def _is_read_only_post(url: object) -> bool:
    if not isinstance(url, str):
        return False

    parsed = urlsplit(url.strip())
    host = (parsed.hostname or "").lower()
    path = unquote(parsed.path)
    return host in _QUICKBOOKS_HOSTS and bool(_QUICKBOOKS_QUERY_PATH.fullmatch(path))


def _graphql_operation_is_mutating(body: object) -> bool | None:
    payload = body
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(payload, dict):
        return None

    document = payload.get("query")
    if not isinstance(document, str):
        return None
    document = _GRAPHQL_LEADING_TRIVIA.sub("", document)
    if not document or _GRAPHQL_MUTATION_TOKEN.search(document):
        return True if document else None

    operation_match = _GRAPHQL_OPERATION.match(document)
    if operation_match is None:
        return None
    operation = (operation_match.group(1) or operation_match.group(2)).lower()
    if operation in {"query", "{"}:
        return False
    if operation == "mutation":
        return True
    return None


def is_api_request_mutating(method: object, url: object, body: object = None) -> bool:
    """Return whether an API request should pass through mutation review."""
    if not isinstance(method, str):
        return False

    normalized_method = method.strip().upper()
    if normalized_method in _READ_ONLY_METHODS:
        return False
    if normalized_method == "POST" and _is_read_only_post(url):
        return False
    if normalized_method == "POST":
        graphql_mutation = _graphql_operation_is_mutating(body)
        if graphql_mutation is not None:
            return graphql_mutation
    return normalized_method in _MUTATING_METHODS