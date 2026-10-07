"""ToolRet tool catalog adapter for retrieval benchmarking.

The catalog entries are metadata-only benchmark records. Their executor does
not call an external service; it returns a clear placeholder if selected.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from openjiuwen.core.foundation.tool import LocalFunction, ToolCard, ToolExposure
from openjiuwen.core.foundation.tool.schema import ToolOutput


_CATEGORIES = ("code", "web", "customized")
_TYPE_MAP = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "dict": "object",
    "object": "object",
    "list": "array",
    "array": "array",
}


def _safe_name(value: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
    if not name or not name[0].isalpha():
        name = f"tool_{name}"
    return name[:64]


def _parameter_schema(parameters: Any) -> dict[str, Any]:
    """Convert ToolRet's parameter shorthand or a JSON schema to JSON schema."""
    if isinstance(parameters, str):
        try:
            parameters = json.loads(parameters)
        except (TypeError, ValueError):
            return {"type": "object", "properties": {}}
    if not isinstance(parameters, dict):
        return {"type": "object", "properties": {}}

    # Some datasets already contain a standard JSON Schema object.
    if parameters.get("type") == "object" or "properties" in parameters:
        schema = dict(parameters)
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        schema["properties"] = {
            str(key): dict(value) if isinstance(value, dict) else {"type": "string"}
            for key, value in schema["properties"].items()
        }
        return schema

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, raw in parameters.items():
        spec = raw if isinstance(raw, dict) else {"type": raw}
        raw_type = str(spec.get("type", "string")).lower()
        nullable = "optional" in raw_type or "none" in raw_type
        base_type = raw_type.replace("optional[", "").replace("]", "")
        base_type = base_type.split("|")[0].strip()
        prop: dict[str, Any] = {"type": _TYPE_MAP.get(base_type, "string")}
        description = spec.get("description")
        if description:
            prop["description"] = str(description)
        for key in ("default", "enum", "items", "format"):
            if key in spec:
                prop[key] = spec[key]
        properties[str(name)] = prop
        if not nullable and "default" not in spec and spec.get("required", True):
            required.append(str(name))

    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _placeholder_executor(**kwargs: Any) -> ToolOutput:
    return ToolOutput(
        success=False,
        error=(
            "This ToolRet entry is a retrieval benchmark record, not an "
            "executable service. No external action was performed."
        ),
    )


def register_toolret_tools(categories: str = "all") -> list[LocalFunction]:
    """Load ToolRet's public catalog as deferred tools.

    Requires the optional ``datasets`` package. Categories accepts ``all`` or
    a comma-separated subset of ``code``, ``web``, and ``customized``. Catalog
    loading and tool construction are cached per category selection.
    """
    selected = categories.strip().lower()
    if selected == "all":
        normalized = ",".join(_CATEGORIES)
    else:
        normalized = ",".join(
            sorted({item.strip().lower() for item in selected.split(",") if item.strip()})
        )
    return list(_load_toolret_tools(normalized))


@lru_cache(maxsize=3)
def _load_toolret_tools(categories: str) -> tuple[LocalFunction, ...]:
    try:
        from datasets import concatenate_datasets, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "ToolRet benchmark tools require the optional 'datasets' package. "
            "Install JiuwenSwarm with its toolret-benchmark extra."
        ) from exc

    selected_categories = categories.split(",")
    invalid = sorted(set(selected_categories) - set(_CATEGORIES))
    if not selected_categories or invalid:
        raise ValueError(
            f"Invalid ToolRet categories {invalid or selected_categories}; "
            f"choose from {', '.join(_CATEGORIES)} or 'all'."
        )

    records = concatenate_datasets(
        [
            load_dataset("mangopy/ToolRet-Tools", category)["tools"]
            for category in selected_categories
        ]
    )
    tools: list[LocalFunction] = []
    used_names: set[str] = set()
    for record in records:
        record_id = str(record.get("id") or "").strip()
        doc = record.get("doc") or {}
        if not record_id or not isinstance(doc, dict):
            continue
        name = _safe_name(f"toolret_{record_id}")
        if name in used_names:
            continue
        used_names.add(name)
        original_name = str(doc.get("name") or record_id).strip()
        description = str(doc.get("description") or "").strip()
        card = ToolCard(
            id=name,
            name=name,
            description=f"{original_name}: {description}".strip(": "),
            input_params=_parameter_schema(doc.get("parameters")),
            exposure=ToolExposure.DEFERRED,
            parallel_safe=True,
            stateless=True,
            idempotent=True,
        )
        tools.append(LocalFunction(card=card, func=_placeholder_executor))
    return tuple(tools)


__all__ = ["register_toolret_tools"]
