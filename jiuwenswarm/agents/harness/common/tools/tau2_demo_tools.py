"""tau2 domain tools exposed as deferred JiuWenSwarm demo tools."""

from __future__ import annotations

from typing import Any

from openjiuwen.core.foundation.tool import LocalFunction, ToolCard, ToolExposure
from openjiuwen.core.foundation.tool.schema import ToolOutput


def register_tau2_demo_tools(
    domain: str,
    *,
    owner_id: str,
) -> tuple[list[LocalFunction], str]:
    """Build tau2-backed tools and return them with the domain policy.

    Each domain gets a fresh tau2 environment. The returned tool executors
    share their domain environment so mutations persist for the lifetime of
    the owning JiuWenSwarm session adapter. ``domain="all"`` combines airline,
    retail, telecom, and mock, prefixing names to avoid cross-domain clashes.
    """
    try:
        from tau2.data_model.message import ToolCall as Tau2ToolCall
        from tau2.environment.tool import as_tool
        from tau2.runner import build_environment
    except ModuleNotFoundError as exc:
        if exc.name != "tau2":
            raise RuntimeError(
                "tau2 is on PYTHONPATH, but one of its dependencies is missing: "
                f"{exc.name}. Run `uv sync` from the JiuWenSwarm checkout."
            ) from exc
        raise RuntimeError(
            "tau2_demo is enabled, but tau2-bench is not importable. Add "
            "tau2-bench/src to PYTHONPATH or run `uv sync` from the "
            "JiuWenSwarm checkout."
        ) from exc

    domains = ["airline", "retail", "telecom", "mock"] if domain == "all" else [domain]
    tools: list[LocalFunction] = []
    policies: list[str] = []
    for domain_name in domains:
        environment = build_environment(domain_name)
        policies.append(f"## {domain_name}\n{environment.get_policy()}")
        # tau2 separates ordinary tools from discoverable tools. In JiuWenSwarm
        # both groups should be deferred and available through tool search.
        tau2_tools = environment.get_tools()
        tau2_tools.extend(
            as_tool(tool)
            for tool in environment.tools.get_discoverable_tools().values()
        )
        for index, tau_tool in enumerate(tau2_tools):
            schema = tau_tool.openai_schema["function"]
            native_name = str(schema["name"])
            exposed_name = (
                f"{domain_name}_{native_name}" if domain == "all" else native_name
            )
            description = str(schema.get("description") or native_name)
            if domain == "all":
                description = f"[{domain_name}] {description}"
            card = ToolCard(
                id=f"tau2_demo_{owner_id}_{domain_name}_{index}",
                name=exposed_name,
                description=description,
                input_params=schema.get("parameters") or {"type": "object", "properties": {}},
                exposure=ToolExposure.DEFERRED,
                parallel_safe=False,
                stateless=False,
                idempotent=False,
                properties={"catalog": "tau2_demo"},
            )

            def execute(
                _environment=environment,
                _native_name: str = native_name,
                **kwargs: Any,
            ) -> ToolOutput:
                response = _environment.get_response(
                    Tau2ToolCall(
                        id=f"tau2-demo-{_native_name}",
                        name=_native_name,
                        arguments=kwargs,
                        requestor="assistant",
                    )
                )
                if response.error:
                    return ToolOutput(
                        success=False,
                        error=response.content or "tau2 tool failed",
                    )
                return ToolOutput(
                    success=True,
                    data={"content": response.content or "", "tool_name": _native_name},
                )

            execute.__name__ = exposed_name
            execute.__doc__ = description
            tools.append(LocalFunction(card=card, func=execute))

    return tools, "\n\n".join(policies)


__all__ = ["register_tau2_demo_tools"]
