"""Contract tests for the built-in generation launch defaults.

These tests pin the initial UI values shown when ``uv run acestep`` is started
without CLI flags.  They parse the interface builders' source (AST) rather than
booting the full app, matching the repo's existing contract-test convention.
"""

import ast
from pathlib import Path
import unittest


_DIR = Path(__file__).resolve().parent
_PRIMARY = _DIR / "generation_tab_primary_controls.py"
_OPTIONAL = _DIR / "generation_tab_optional_controls.py"
_SIMPLE = _DIR / "generation_tab_simple_controls.py"
_ADV_PRIMARY = _DIR / "generation_advanced_primary_controls.py"
_ADV_OUTPUT = _DIR / "generation_advanced_output_controls.py"


def _component_call(path: Path, func_name: str, target: str) -> ast.Call:
    """Return the Call node assigned to *target* inside *func_name*."""
    module = ast.parse(path.read_text(encoding="utf-8"))
    func = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == func_name
    )
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if node.targets[0].id == target and isinstance(node.value, ast.Call):
            return node.value
    raise AssertionError(f"Assignment {target!r} not found in {func_name!r}")


def _kw(call: ast.Call, name: str) -> ast.expr:
    """Return the value of keyword *name* from a component Call."""
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    raise AssertionError(f"kwarg {name!r} not found on component call")


class LaunchModeDefaultsTests(unittest.TestCase):
    """The generation mode should start in Simple (with matching state)."""

    def test_mode_radio_defaults_to_simple(self) -> None:
        call = _component_call(_PRIMARY, "build_mode_selector_controls", "generation_mode")
        self.assertEqual(_kw(call, "value").value, "Simple")

    def test_previous_mode_state_defaults_to_simple(self) -> None:
        call = _component_call(_PRIMARY, "build_hidden_generation_state", "previous_generation_mode")
        self.assertEqual(_kw(call, "value").value, "Simple")


class LaunchOptionalDefaultsTests(unittest.TestCase):
    """Duration should start at a fixed 120s with the auto toggle off."""

    def test_audio_duration_defaults_to_120(self) -> None:
        call = _component_call(_OPTIONAL, "build_optional_parameter_controls", "audio_duration")
        self.assertEqual(_kw(call, "value").value, 120)

    def test_duration_auto_defaults_to_off(self) -> None:
        call = _component_call(_OPTIONAL, "build_optional_parameter_controls", "duration_auto")
        self.assertIs(_kw(call, "value").value, False)


class LaunchSimpleControlsDefaultsTests(unittest.TestCase):
    """Simple mode should start instrumental with the language lock engaged."""

    def test_instrumental_checkbox_defaults_on(self) -> None:
        call = _component_call(_SIMPLE, "build_simple_input_controls", "simple_instrumental_checkbox")
        self.assertIs(_kw(call, "value").value, True)

    def test_vocal_language_defaults_locked(self) -> None:
        call = _component_call(_SIMPLE, "build_simple_input_controls", "simple_vocal_language")
        self.assertIs(_kw(call, "interactive").value, False)


class LaunchAdvancedDefaultsTests(unittest.TestCase):
    """Caption rewrite on, and FLAC as the default output format."""

    def test_caption_rewrite_defaults_on(self) -> None:
        call = _component_call(_ADV_PRIMARY, "build_lm_controls", "use_cot_caption")
        self.assertIs(_kw(call, "value").value, True)

    def test_audio_format_defaults_to_flac(self) -> None:
        call = _component_call(_ADV_OUTPUT, "build_output_controls", "initial_audio_format")
        # ``params.get("audio_format", "<default>")`` — the default is args[1].
        self.assertEqual(call.args[1].value, "flac")


if __name__ == "__main__":
    unittest.main()
