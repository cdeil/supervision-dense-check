"""Verify the detector and the published before/after evidence."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from scene import CAPS, COUNTS, detect, render


@pytest.mark.parametrize("count", COUNTS)
@pytest.mark.parametrize("classes", [1, 10])
def test_detector_recovers_rendered_geometry(count: int, classes: int) -> None:
    """Boxes and categories are recovered from pixels at every tested density."""
    image, expected_boxes, expected_classes = render(count, classes)
    boxes, labels, scores = detect(image)
    np.testing.assert_array_equal(boxes, expected_boxes)
    np.testing.assert_array_equal(labels, expected_classes)
    assert len(scores) == count
    assert np.all(np.diff(scores) < 0)


def test_frozen_measurements_match_reference_and_preserve_inputs() -> None:
    """Every published score matches COCO and uses identical before/after inputs."""
    report = json.loads(Path("site/results.json").read_text())
    assert report["before"]["versions"]["supervision"] == "0.30.9"
    assert len(report["after"]["cases"]) == 20
    for before, after in zip(
        report["before"]["cases"], report["after"]["cases"], strict=True
    ):
        assert before["image_pixel_sha256"] == after["image_pixel_sha256"]
        assert before["predictions"] == after["predictions"]
        assert (
            after["image_pixel_sha256"]
            == hashlib.sha256(
                Image.open(Path("site", after["image"])).convert("RGB").tobytes()
            ).hexdigest()
        )
        assert [item["cap"] for item in after["measurements"]] == list(CAPS)
        np.testing.assert_allclose(
            before["measurements"][0]["ap50"],
            after["measurements"][0]["ap50"],
            atol=1e-7,
            rtol=0,
        )
        for item in before["measurements"] + after["measurements"]:
            np.testing.assert_allclose(
                [item["ap50"], item["ap50_95"]],
                [item["reference"]["ap50"], item["reference"]["ap50_95"]],
                atol=1e-7,
            )
        if not after["missing"]:
            assert after["detected"] == after["count"]
            assert after["measurements"][-1]["ap50"] == pytest.approx(1.0)
        else:
            assert after["detected"] < after["count"]
            assert after["measurements"][-1]["ap50"] < 1.0


def test_dense_single_class_exposes_cap_and_multiclass_control() -> None:
    """Dense one-class input loses recall; ten balanced categories avoid that cap."""
    cases = json.loads(Path("site/results.json").read_text())["after"]["cases"]
    single = next(
        case
        for case in cases
        if case["count"] == 1000 and case["classes"] == 1 and not case["missing"]
    )
    multi = next(
        case
        for case in cases
        if case["count"] == 1000 and case["classes"] == 10 and not case["missing"]
    )
    assert single["measurements"][0]["ap50"] == pytest.approx(11 / 101)
    assert single["measurements"][-1]["ap50"] == pytest.approx(1.0)
    assert multi["measurements"][0]["ap50"] == pytest.approx(1.0)
