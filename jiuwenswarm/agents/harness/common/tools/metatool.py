"""MetaTool catalog adapter for retrieval benchmarking.

The catalog entries are metadata-only benchmark records. Their executor does
not call an external service; it returns a clear placeholder if selected.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from openjiuwen.core.foundation.tool import LocalFunction, ToolCard, ToolExposure
from openjiuwen.core.foundation.tool.schema import ToolOutput


_EXPECTED_TOOL_COUNT = 199
_TOOL_DOCUMENTS_FILENAME = "plugin_des.json"
_GROUPED_TOOL_DOCUMENTS_FILENAME = "big_tool_des.json"


def _default_data_dir() -> Path:
    # The benchmark repository is expected beside the JiuwenSwarm checkout.
    return Path(__file__).resolve().parents[5].parent / "MetaTool" / "dataset"


def _resolve_data_dir(data_dir: str | Path | None) -> Path:
    if data_dir is None or not str(data_dir).strip():
        return _default_data_dir()
    path = Path(data_dir).expanduser()
    if not path.is_absolute():
        # Relative config paths are rooted at the JiuwenSwarm repository.
        path = Path(__file__).resolve().parents[5] / path
    return path.resolve()


def _placeholder_executor(**kwargs: Any) -> ToolOutput:
    return ToolOutput(
        success=False,
        error=(
            "This MetaTool entry is a retrieval benchmark record, not an "
            "executable service. No external action was performed."
        ),
    )


def register_metatool_tools(
    data_dir: str | Path | None = None,
) -> list[LocalFunction]:
    """Load MetaTool's 199 tool descriptions as deferred tools.

    MetaTool provides descriptions but no parameter schemas, so each card
    receives an empty object schema rather than invented parameters. Grouped
    descriptions from ``big_tool_des.json`` take precedence for matching names;
    all other names use ``plugin_des.json``.
    """
    return list(_load_metatool_tools(str(_resolve_data_dir(data_dir))))


@lru_cache(maxsize=4)
def _load_metatool_tools(data_dir: str) -> tuple[LocalFunction, ...]:
    tool_path = Path(data_dir) / _TOOL_DOCUMENTS_FILENAME
    if not tool_path.is_file():
        raise FileNotFoundError(
            f"MetaTool catalog not found: {tool_path}. "
            "Set metatool_benchmark.data_dir to the repository's dataset directory."
        )

    try:
        records = json.loads(tool_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read MetaTool catalog {tool_path}: {exc}") from exc
    if not isinstance(records, dict):
        raise ValueError("MetaTool catalog must be a JSON object of name to description")
    if len(records) != _EXPECTED_TOOL_COUNT:
        raise ValueError(
            f"MetaTool catalog has {len(records)} documents; "
            f"expected {_EXPECTED_TOOL_COUNT}: {tool_path}"
        )

    grouped_path = Path(data_dir) / _GROUPED_TOOL_DOCUMENTS_FILENAME
    try:
        grouped_records = json.loads(grouped_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not read MetaTool grouped descriptions {grouped_path}: {exc}"
        ) from exc
    if not isinstance(grouped_records, dict):
        raise ValueError(
            "MetaTool grouped descriptions must be a JSON object of name to description"
        )

    tools: list[LocalFunction] = []
    seen_names: set[str] = set()
    for index, (raw_name, raw_description) in enumerate(records.items()):
        name = str(raw_name).strip()
        if not name:
            raise ValueError(f"MetaTool document at index {index} has an empty name")
        if name in seen_names:
            raise ValueError(f"MetaTool catalog contains duplicate name: {name}")
        description = grouped_records.get(name, raw_description)
        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                f"MetaTool document {name!r} has no usable description"
            )
        seen_names.add(name)

        card = ToolCard(
            id=f"metatool_{index}",
            name=name,
            description=description.strip(),
            input_params={"type": "object", "properties": {}},
            exposure=ToolExposure.DEFERRED,
            parallel_safe=True,
            stateless=True,
            idempotent=True,
        )
        tools.append(LocalFunction(card=card, func=_placeholder_executor))

    if len(tools) != _EXPECTED_TOOL_COUNT:
        raise RuntimeError(
            f"Loaded {len(tools)} MetaTool cards; expected {_EXPECTED_TOOL_COUNT}"
        )
    return tuple(tools)


__all__ = ["register_metatool_tools"]
