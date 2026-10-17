"""Unit tests for the generation launch-default computation.

These tests pin the built-in defaults applied when ``uv run acestep`` is
started without CLI flags: CPU offload enabled (except Apple Silicon), the
PyTorch ("pt") 5Hz LM backend, and a batch size of ``min(8, max)``. They also
cover the .env-based default model resolution used by the service controls.
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


from acestep.ui.gradio.interfaces.generation_defaults import (
    compute_init_defaults,
    resolve_default_config_path,
    resolve_default_lm_model,
)


def _gpu_config(**overrides) -> SimpleNamespace:
    """Return a representative GPU config with the given overrides."""
    base = dict(
        tier="tier6b",
        gpu_memory_gb=24.0,
        max_duration_with_lm=480,
        max_duration_without_lm=480,
        max_batch_size_with_lm=8,
        max_batch_size_without_lm=8,
        init_lm_default=True,
        offload_to_cpu_default=False,
        offload_dit_to_cpu_default=False,
        quantization_default=False,
        compile_model_default=False,
        lm_backend_restriction="all",
        recommended_backend="vllm",
        recommended_lm_model="acestep-5Hz-lm-1.7B",
        available_lm_models=["acestep-5Hz-lm-1.7B"],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class ComputeInitDefaultsTests(unittest.TestCase):
    """Verify compute_init_defaults produces the built-in launch defaults."""

    def setUp(self) -> None:
        # Ensure the XPU override path never fires so tests are deterministic
        # regardless of the host's accelerator stack.
        self._xpu_patcher = None
        try:
            import torch

            if hasattr(torch, "xpu"):
                self._xpu_patcher = patch.object(torch.xpu, "is_available", return_value=False)
                self._xpu_patcher.start()
        except ImportError:
            pass

    def tearDown(self) -> None:
        if self._xpu_patcher is not None:
            self._xpu_patcher.stop()

    def test_default_offload_enabled_on_non_darwin(self) -> None:
        """CPU offload should be enabled by default on the primary (Linux) path."""
        with patch.object(sys, "platform", "linux"):
            result = compute_init_defaults({"gpu_config": _gpu_config()}, "en")
        self.assertTrue(result["default_offload"])

    def test_default_offload_disabled_on_darwin(self) -> None:
        """Apple Silicon keeps CPU offload disabled (non-target path preserved)."""
        with patch.object(sys, "platform", "darwin"):
            result = compute_init_defaults({"gpu_config": _gpu_config()}, "en")
        self.assertFalse(result["default_offload"])

    def test_recommended_backend_is_pt(self) -> None:
        """The 5Hz LM backend default should be 'pt' when it is available."""
        result = compute_init_defaults({"gpu_config": _gpu_config()}, "en")
        self.assertEqual(result["recommended_backend"], "pt")

    def test_recommended_backend_pt_only_restriction(self) -> None:
        """With a pt_only restriction the backend should still resolve to 'pt'."""
        config = _gpu_config(lm_backend_restriction="pt_only")
        result = compute_init_defaults({"gpu_config": config}, "en")
        self.assertEqual(result["recommended_backend"], "pt")
        self.assertEqual(result["available_backends"], ["pt"])

    def test_default_batch_size_is_min_8_max(self) -> None:
        """Batch default should be min(8, max_batch_size) when no CLI value given."""
        result = compute_init_defaults({"gpu_config": _gpu_config()}, "en")
        self.assertEqual(result["default_batch_size"], 8)

    def test_default_batch_size_capped_by_max(self) -> None:
        """Batch default should be capped by max_batch_size when max < 8."""
        config = _gpu_config(max_batch_size_without_lm=4)
        result = compute_init_defaults({"gpu_config": config}, "en")
        self.assertEqual(result["default_batch_size"], 4)

    def test_cli_batch_size_overrides_default(self) -> None:
        """A CLI-provided batch size should take precedence over the default."""
        config = _gpu_config(max_batch_size_without_lm=8)
        result = compute_init_defaults(
            {"gpu_config": config, "default_batch_size": 3}, "en"
        )
        self.assertEqual(result["default_batch_size"], 3)

    def test_cli_batch_size_capped_by_max(self) -> None:
        """A CLI batch size above the tier max should be clamped to the max."""
        config = _gpu_config(max_batch_size_without_lm=4)
        result = compute_init_defaults(
            {"gpu_config": config, "default_batch_size": 8}, "en"
        )
        self.assertEqual(result["default_batch_size"], 4)


class ResolveDefaultConfigPathTests(unittest.TestCase):
    """Verify ACESTEP_CONFIG_PATH seeds the UI default DiT model."""

    def test_env_model_wins_when_available(self) -> None:
        """A .env DiT model that is available on disk should become the default."""
        with patch.dict(
            os.environ, {"ACESTEP_CONFIG_PATH": "acestep-v15-xl-sft"}, clear=True
        ):
            result = resolve_default_config_path(
                ["acestep-v15-turbo", "acestep-v15-xl-sft"]
            )
        self.assertEqual("acestep-v15-xl-sft", result)

    def test_env_model_ignored_when_unavailable(self) -> None:
        """A .env DiT model missing from disk should fall back to the built-ins."""
        with patch.dict(
            os.environ, {"ACESTEP_CONFIG_PATH": "acestep-v15-missing"}, clear=True
        ):
            result = resolve_default_config_path(["acestep-v15-turbo"])
        self.assertEqual("acestep-v15-turbo", result)

    def test_whitespace_env_falls_back_to_builtin_preference(self) -> None:
        """A set-but-blank env var must not break the built-in preference order."""
        with patch.dict(os.environ, {"ACESTEP_CONFIG_PATH": "   "}, clear=True):
            result = resolve_default_config_path(
                ["acestep-v15-turbo", "acestep-v15-xl-turbo"]
            )
        self.assertEqual("acestep-v15-xl-turbo", result)

    def test_no_env_prefers_turbo_over_first_available(self) -> None:
        """Without the env var, turbo still wins over the first model on disk."""
        with patch.dict(os.environ, clear=True):
            result = resolve_default_config_path(["custom-model", "acestep-v15-turbo"])
        self.assertEqual("acestep-v15-turbo", result)

    def test_no_models_returns_none(self) -> None:
        """With no models on disk the resolver should return None."""
        with patch.dict(os.environ, clear=True):
            self.assertIsNone(resolve_default_config_path([]))


class ResolveDefaultLmModelTests(unittest.TestCase):
    """Verify ACESTEP_LM_MODEL_PATH seeds the UI default 5Hz LM model."""

    def test_env_model_wins_when_available(self) -> None:
        """A .env LM model that is available on disk should become the default."""
        with patch.dict(
            os.environ, {"ACESTEP_LM_MODEL_PATH": "acestep-5Hz-lm-4B"}, clear=True
        ):
            result = resolve_default_lm_model(
                ["acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-4B"], "acestep-5Hz-lm-1.7B"
            )
        self.assertEqual("acestep-5Hz-lm-4B", result)

    def test_env_model_ignored_when_unavailable(self) -> None:
        """A .env LM model missing from disk should fall back to the tier model."""
        with patch.dict(
            os.environ, {"ACESTEP_LM_MODEL_PATH": "acestep-5Hz-lm-4B"}, clear=True
        ):
            result = resolve_default_lm_model(
                ["acestep-5Hz-lm-1.7B"], "acestep-5Hz-lm-1.7B"
            )
        self.assertEqual("acestep-5Hz-lm-1.7B", result)

    def test_no_env_uses_tier_recommendation(self) -> None:
        """Without the env var, the tier-recommended model should be used."""
        with patch.dict(os.environ, clear=True):
            result = resolve_default_lm_model(
                ["acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-4B"], "acestep-5Hz-lm-1.7B"
            )
        self.assertEqual("acestep-5Hz-lm-1.7B", result)

    def test_no_models_returns_none(self) -> None:
        """With no models on disk the resolver should return None."""
        with patch.dict(os.environ, clear=True):
            self.assertIsNone(resolve_default_lm_model([], "acestep-5Hz-lm-1.7B"))


if __name__ == "__main__":
    unittest.main()
