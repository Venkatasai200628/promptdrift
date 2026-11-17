"""Offline coverage for Ollama response normalization and provenance safety."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from promptdrift.engine.baseline import write_baseline
from promptdrift.engine.monitor import monitor_suite
from promptdrift.engine.runner import run_suite
from promptdrift.errors import ProviderError
from promptdrift.models import Config
from promptdrift.models.config import ProviderConfig
from promptdrift.providers.ollama import OllamaProvider
from promptdrift.storage.history import history_path, load_monitor_history, save_monitor_report

FIXTURES = Path(__file__).parents[1] / "fixtures" / "ollama"


def _payload(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _install_payload(monkeypatch, name: str) -> None:
    payload = _payload(name)

    def fake_post(url: str, **kwargs):
        return httpx.Response(
            200,
            json=payload,
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("promptdrift.providers.ollama.httpx.post", fake_post)


def _install_payload_sequence(monkeypatch, *fixture_names: str) -> None:
    payloads = iter(_payload(name) for name in fixture_names)

    def fake_post(url: str, **kwargs):
        try:
            payload = next(payloads)
        except StopIteration:
            pytest.fail("Unexpected additional Ollama request")

        return httpx.Response(
            200,
            json=payload,
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("promptdrift.providers.ollama.httpx.post", fake_post)


def _monitor_config(tmp_path: Path) -> tuple[Config, Path]:
    config_path = tmp_path / "promptdrift.yaml"
    config = Config.model_validate(
        {
            "provider": {"type": "ollama", "model": "llama3.2"},
            "tests": [
                {
                    "id": "refund-window",
                    "prompt": "policy.txt",
                    "assertions": [{"type": "contains", "value": "30 days"}],
                }
            ],
        }
    )
    (tmp_path / "policy.txt").write_text("Refunds: 30 days", encoding="utf-8")
    config_path.write_text(json.dumps(config.model_dump(mode="json")), encoding="utf-8")
    return config, config_path


def _write_baseline(config: Config, config_path: Path, monkeypatch) -> None:
    _install_payload(monkeypatch, "generate_with_metadata.json")
    baseline = run_suite(config, config_path)
    write_baseline(config_path.parent / config.baseline.path, baseline)


def test_generate_fixture_normalizes_supported_metadata(monkeypatch):
    _install_payload(monkeypatch, "generate_with_metadata.json")
    response = OllamaProvider(ProviderConfig(type="ollama", model="alias")).complete(
        "test", temperature=0, max_output_tokens=100
    )

    assert response.output == "30 days PRIVATE_OLLAMA_OUTPUT_SENTINEL"
    assert response.model == "alias"
    assert response.resolved_model == "llama3.2:latest"
    assert response.input_tokens == 7
    assert response.output_tokens == 4
    assert response.system_fingerprint is None


@pytest.mark.parametrize(
    ("current_fixture", "expected_evidence"),
    [
        (
            "generate_metadata_changed.json",
            "Provider-reported response_model_hash changed (supporting signal only).",
        ),
        (
            "generate_metadata_absent.json",
            "Provider-reported response_model_hash is unavailable on one side; no equality claim is possible.",
        ),
    ],
)
def test_ollama_metadata_change_or_absence_stays_supporting_or_uncertain(
    tmp_path, monkeypatch, current_fixture, expected_evidence
):
    config, config_path = _monitor_config(tmp_path)
    _write_baseline(config, config_path, monkeypatch)
    _install_payload(monkeypatch, current_fixture)

    report = monitor_suite(config, config_path, samples=1)

    assert report.tests[0].diagnosis == "stable"
    assert any(expected_evidence in item for item in report.tests[0].evidence)


def test_monitor_detects_ollama_model_change_across_two_samples(tmp_path, monkeypatch):
    config, config_path = _monitor_config(tmp_path)
    _write_baseline(config, config_path, monkeypatch)

    _install_payload_sequence(
        monkeypatch,
        "generate_with_metadata.json",
        "generate_metadata_changed.json",
    )

    report = monitor_suite(config, config_path, samples=2)

    expected = "Provider-reported response_model_hash changed (supporting signal only)."

    assert report.tests[0].samples == 2
    assert expected in report.tests[0].evidence
    assert report.tests[0].diagnosis == "stable"


def test_monitor_does_not_report_model_change_when_model_is_unchanged(tmp_path, monkeypatch):
    config, config_path = _monitor_config(tmp_path)
    _write_baseline(config, config_path, monkeypatch)

    _install_payload_sequence(
        monkeypatch,
        "generate_with_metadata.json",
        "generate_with_metadata.json",
    )

    report = monitor_suite(config, config_path, samples=2)

    assert report.tests[0].samples == 2
    assert not any(
        "Provider-reported response_model_hash changed" in item for item in report.tests[0].evidence
    )
    assert report.tests[0].diagnosis == "stable"


def test_monitor_history_does_not_persist_raw_ollama_payload(tmp_path, monkeypatch):
    config, config_path = _monitor_config(tmp_path)
    _write_baseline(config, config_path, monkeypatch)
    _install_payload(monkeypatch, "generate_with_metadata.json")

    report = monitor_suite(config, config_path, samples=1)
    path = history_path(config_path)
    save_monitor_report(report, path)
    stored = json.dumps(load_monitor_history(path))

    assert "PRIVATE_OLLAMA_OUTPUT_SENTINEL" not in stored
    assert "314159" not in stored


@pytest.mark.parametrize(
    ("fixture_name", "private_value"),
    [
        ("generate_malformed_metadata.json", "PRIVATE_MALFORMED_OLLAMA_OUTPUT"),
        ("chat_response_unsupported.json", "PRIVATE_CHAT_RESPONSE"),
    ],
)
def test_malformed_or_unsupported_ollama_shape_fails_safely(
    monkeypatch, fixture_name, private_value
):
    _install_payload(monkeypatch, fixture_name)

    with pytest.raises(ProviderError, match="Ollama request failed") as error:
        OllamaProvider(ProviderConfig(type="ollama", model="llama3.2")).complete(
            "private prompt", temperature=0, max_output_tokens=100
        )

    assert private_value not in str(error.value)
