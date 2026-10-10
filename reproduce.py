"""Measure released or patched Supervision against a configured COCO reference."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import numpy as np
import supervision as sv
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from supervision.metrics import MeanAveragePrecision
from scene import CAPS, COUNTS, detect, render


def reference(boxes, labels, predicted, predicted_labels, scores, cap):
    """Select AP from COCO's actual cap axis, avoiding its default summary."""
    categories = sorted(set(labels.tolist()))
    annotations = []
    for index, (box, category) in enumerate(zip(boxes, labels, strict=True)):
        x0, y0, x1, y1 = box
        annotations.append(
            {
                "id": index + 1,
                "image_id": 0,
                "category_id": int(category),
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "area": (x1 - x0) * (y1 - y0),
                "iscrowd": 0,
            }
        )
    predictions = []
    for box, category, score in zip(predicted, predicted_labels, scores, strict=True):
        x0, y0, x1, y1 = box
        predictions.append(
            {
                "image_id": 0,
                "category_id": int(category),
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "score": float(score),
            }
        )
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {
            "info": {},
            "images": [{"id": 0}],
            "categories": [{"id": c, "name": str(c)} for c in categories],
            "annotations": annotations,
        }
        gt.createIndex()
        evaluator = COCOeval(gt, gt.loadRes(predictions), "bbox")
        evaluator.params.maxDets = [1, 10, cap]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"][:, :, :, 0, -1]
    return {
        "ap50": round(float(precision[0][precision[0] >= 0].mean()), 12),
        "ap50_95": round(float(precision[precision >= 0].mean()), 12),
    }


def run(patched: bool, assets: Path) -> dict:
    """Re-evaluate identical pixel detections for every count/class/control case."""
    assets.mkdir(parents=True, exist_ok=True)
    cases = []
    for count in COUNTS:
        for class_count in (1, 10):
            image, boxes, labels = render(count, class_count)
            detected, categories, confidence = detect(image)
            np.testing.assert_array_equal(detected, boxes)
            np.testing.assert_array_equal(categories, labels)
            image_path = assets / f"stars-{count}-{class_count}.png"
            image.save(image_path, optimize=True)
            for missing in (False, True):
                keep = np.ones(count, dtype=bool)
                if missing:
                    keep[::10] = False
                pred = sv.Detections(
                    xyxy=detected[keep],
                    class_id=categories[keep],
                    confidence=confidence[keep],
                )
                target = sv.Detections(xyxy=boxes, class_id=labels)
                measurements = []
                for cap in CAPS if patched else (100,):
                    options = (
                        {"max_detection_thresholds": (1, 10, cap)} if patched else {}
                    )
                    result = (
                        MeanAveragePrecision(**options).update(pred, target).compute()
                    )
                    coco = reference(
                        boxes,
                        labels,
                        detected[keep],
                        categories[keep],
                        confidence[keep],
                        cap,
                    )
                    np.testing.assert_allclose(
                        [result.map50, result.map50_95],
                        [coco["ap50"], coco["ap50_95"]],
                        atol=1e-7,
                    )
                    measurements.append(
                        {
                            "cap": cap,
                            "ap50": round(float(result.map50), 12),
                            "ap50_95": round(float(result.map50_95), 12),
                            "reference": coco,
                        }
                    )
                cases.append(
                    {
                        "count": count,
                        "classes": class_count,
                        "missing": missing,
                        "detected": int(keep.sum()),
                        "boxes": boxes.astype(int).tolist(),
                        "predictions": detected[keep].astype(int).tolist(),
                        "image": f"assets/{image_path.name}",
                        "image_pixel_sha256": hashlib.sha256(
                            image.tobytes()
                        ).hexdigest(),
                        "prediction_class_ids": categories[keep].tolist(),
                        "prediction_scores": np.round(confidence[keep], 12).tolist(),
                        "measurements": measurements,
                    }
                )
    return {
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("supervision", "numpy", "pillow", "scipy", "pycocotools")
        },
        "mode": "after" if patched else "before",
        "cases": cases,
    }


def main() -> None:
    """Write a frozen report, optionally joining a separately measured baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.after, Path("site/assets"))
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        for before, after in zip(baseline["cases"], report["cases"], strict=True):
            assert before["image_pixel_sha256"] == after["image_pixel_sha256"]
            assert before["predictions"] == after["predictions"]
            assert before["prediction_class_ids"] == after["prediction_class_ids"]
            assert before["prediction_scores"] == after["prediction_scores"]
            np.testing.assert_allclose(
                before["measurements"][0]["ap50"],
                after["measurements"][0]["ap50"],
                atol=1e-12,
            )
        report = {"before": baseline, "after": report}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, separators=(",", ":")) + "\n")
    print(f"Verified {len(report.get('after', report)['cases'])} cases → {args.output}")


if __name__ == "__main__":
    main()
