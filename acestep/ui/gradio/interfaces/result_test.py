"""Unit tests for the results-section initial audio-column visibility.

Regression coverage for: with the built-in default batch size of 8, the
results view rendered only 2 of the 8 sample columns visible on first
render, so 6 generated samples were hidden in the browser.  The initial
visibility must match the initial ``batch_size_input`` value.
"""

import unittest
from unittest.mock import MagicMock

import gradio as gr

from acestep.ui.gradio.interfaces.result import create_results_section


def _build_section(initial_batch_size: int | None = 8) -> dict:
    """Build the results section inside a Blocks context, as in production."""
    with gr.Blocks():
        return create_results_section(MagicMock(), initial_batch_size)


class CreateResultsSectionVisibilityTests(unittest.TestCase):
    """Initial per-sample column visibility follows the initial batch size."""

    def _column_visibility(self, section: dict) -> list[bool]:
        """Return the ``visible`` flag for audio columns 1-8 in order."""
        return [section[f"audio_col_{i}"].visible for i in range(1, 9)]

    def test_batch_size_8_shows_all_columns(self) -> None:
        """With the default batch size of 8, all 8 columns and both rows show.

        Regression: the old static initial state (2 columns visible) hid
        6 of the 8 generated samples in the browser.
        """
        section = _build_section(initial_batch_size=8)
        self.assertEqual(
            self._column_visibility(section),
            [True] * 8,
            "batch size 8 must show all 8 sample columns",
        )
        self.assertTrue(section["audio_row_5_8"].visible)

    def test_batch_size_2_shows_only_first_two_columns(self) -> None:
        """A batch size of 2 keeps the legacy layout: 2 visible columns."""
        section = _build_section(initial_batch_size=2)
        self.assertEqual(
            self._column_visibility(section),
            [True, True, False, False, False, False, False, False],
        )
        self.assertFalse(section["audio_row_5_8"].visible)

    def test_batch_size_5_shows_second_row(self) -> None:
        """A batch size of 5 shows the second row with its first column."""
        section = _build_section(initial_batch_size=5)
        self.assertEqual(
            self._column_visibility(section),
            [True, True, True, True, True, False, False, False],
        )
        self.assertTrue(section["audio_row_5_8"].visible)

    def test_batch_size_1_shows_single_column(self) -> None:
        """A batch size of 1 shows exactly one column."""
        section = _build_section(initial_batch_size=1)
        self.assertEqual(
            self._column_visibility(section),
            [True, False, False, False, False, False, False, False],
        )
        self.assertFalse(section["audio_row_5_8"].visible)

    def test_missing_batch_size_defaults_to_all_visible(self) -> None:
        """Without a batch size the view shows every column (never hides)."""
        section = _build_section(initial_batch_size=None)
        self.assertEqual(self._column_visibility(section), [True] * 8)
        self.assertTrue(section["audio_row_5_8"].visible)

    def test_out_of_range_batch_size_is_clamped(self) -> None:
        """Batch sizes outside 1-8 are clamped like the change-event handler."""
        section = _build_section(initial_batch_size=16)
        self.assertEqual(self._column_visibility(section), [True] * 8)

        section = _build_section(initial_batch_size=0)
        self.assertEqual(
            self._column_visibility(section),
            [True, False, False, False, False, False, False, False],
        )


class CreateResultsSectionShapeTests(unittest.TestCase):
    """Non-target behavior: the component map keeps its wiring keys."""

    def test_expected_wiring_keys_present(self) -> None:
        section = _build_section(initial_batch_size=4)
        expected_keys = [
            "lm_metadata_state",
            "is_format_caption_state",
            "current_batch_index",
            "total_batches",
            "batch_queue",
            "generation_params_state",
            "is_generating_background",
            "status_output",
            "prev_batch_btn",
            "batch_indicator",
            "next_batch_btn",
            "next_batch_status",
            "restore_params_btn",
            "audio_row_5_8",
            "generated_audio_batch",
            "generation_info",
        ]
        for key in expected_keys:
            self.assertIn(key, section)
        for i in range(1, 9):
            for suffix in (
                "generated_audio", "audio_col", "send_to_remix_btn",
                "send_to_repaint_btn", "save_btn", "score_btn",
                "score_display", "codes_display", "convert_to_codes_btn",
                "lrc_btn", "lrc_display", "save_lrc_btn",
                "lrc_download_file", "details_accordion",
            ):
                self.assertIn(f"{suffix}_{i}", section)


if __name__ == "__main__":
    unittest.main()
