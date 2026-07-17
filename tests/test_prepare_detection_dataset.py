from __future__ import annotations

import pytest

from scripts.prepare_detection_dataset import convert_row


def test_converts_polygon_to_minimal_box() -> None:
    converted = convert_row("2 0.1 0.2 0.5 0.2 0.5 0.8 0.1 0.8")
    values = converted.split()
    assert values[0] == "2"
    assert list(map(float, values[1:])) == pytest.approx([0.3, 0.5, 0.4, 0.6])


def test_can_collapse_annotations_to_single_class() -> None:
    assert convert_row("6 0.5 0.5 0.2 0.3", single_class=True) == "0 0.5 0.5 0.2 0.3"


def test_rejects_malformed_annotation() -> None:
    with pytest.raises(ValueError, match="neither"):
        convert_row("1 0.1 0.2 0.3")
