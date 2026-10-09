from jiuwenswarm.common.config import get_tool_discovery_config


def test_tool_discovery_defaults_to_bm25_and_jev_model() -> None:
    assert get_tool_discovery_config({}) == {
        "tool_discovery_backend": "bm25",
        "tool_discovery_api_key": None,
        "tool_discovery_model": "typesafe/jev-1.13",
        "tool_discovery_max_tools": 10,
        "tool_discovery_min_score": 0.0,
        "tool_discovery_api_base": None,
    }


def test_tool_discovery_accepts_hosted_model_settings() -> None:
    assert get_tool_discovery_config(
        {
            "tool_discovery_backend": " JEV ",
            "tool_discovery_model": " custom/model ",
            "tool_discovery_max_tools": 100,
            "tool_discovery_api_base": " https://provider.example/decisions ",
        }
    ) == {
        "tool_discovery_backend": "jev",
        "tool_discovery_api_key": None,
        "tool_discovery_model": "custom/model",
        "tool_discovery_max_tools": 10,
        "tool_discovery_min_score": 0.0,
        "tool_discovery_api_base": "https://provider.example/decisions",
    }


def test_tool_discovery_rejects_local_embedding_backend() -> None:
    config = get_tool_discovery_config(
        {"tool_discovery_backend": "sentence_transformers"}
    )
    assert config["tool_discovery_backend"] == "bm25"
    assert config["tool_discovery_model"] == "typesafe/jev-1.13"


def test_tool_discovery_ignores_unsupported_legacy_keys() -> None:
    config = get_tool_discovery_config(
        {
            "tool_discovery_backend": "jev",
            "jev_model": "typesafe/old-model",
            "jev_max_tools": 6,
            "tool_discovery_embedding_model": "old-embedding-model",
        }
    )
    assert config == {
        "tool_discovery_backend": "jev",
        "tool_discovery_api_key": None,
        "tool_discovery_model": "typesafe/jev-1.13",
        "tool_discovery_max_tools": 10,
        "tool_discovery_min_score": 0.0,
        "tool_discovery_api_base": None,
    }


def test_tool_discovery_invalid_values_fall_back_safely() -> None:
    assert get_tool_discovery_config(
        {
            "tool_discovery_backend": "unknown",
            "tool_discovery_model": " ",
            "tool_discovery_max_tools": "not-a-number",
        }
    ) == {
        "tool_discovery_backend": "bm25",
        "tool_discovery_api_key": None,
        "tool_discovery_model": "typesafe/jev-1.13",
        "tool_discovery_max_tools": 10,
        "tool_discovery_min_score": 0.0,
        "tool_discovery_api_base": None,
    }
