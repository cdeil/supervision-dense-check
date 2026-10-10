## Description

Allow callers to opt into a larger prediction limit for dense-scene mAP, while retaining COCO's `(1, 10, 100)` default.

On an image with 1,000 perfect predictions of one category, `MeanAveragePrecision()` reports AP ≈ 0.108911 because only the 100 highest-confidence predictions enter matching. The targets stay uncapped. This is correct COCO AP@100, but callers cannot currently choose an evaluation cap suitable for dense scenes through the public API.

This PR adds a keyword-only option:

```python
MeanAveragePrecision(max_detection_thresholds=(1, 10, 1000))
```

With the same predictions and targets, AP becomes 1.0. Default results remain identical.

**[Interactive before/after check](https://cdeil.github.io/supervision-dense-check/)** · **[Standalone uv-locked reproducer](https://github.com/cdeil/supervision-dense-check)**

Minimal reproduction (boxes, no model or dataset required):

```python
import numpy as np
import supervision as sv
from supervision.metrics import MeanAveragePrecision

n = 1000
x = np.arange(n, dtype=float) * 2
boxes = np.column_stack((x, np.zeros(n), x + 1, np.ones(n)))
targets = sv.Detections(xyxy=boxes, class_id=np.zeros(n, dtype=int))
predictions = sv.Detections(
    xyxy=boxes.copy(),
    class_id=np.zeros(n, dtype=int),
    confidence=np.linspace(0.99, 0.5, n),
)
print(MeanAveragePrecision().update(predictions, targets).compute().map50)
# approximately 0.108911 on released 0.30.9 and this branch
print(
    MeanAveragePrecision(max_detection_thresholds=(1, 10, 1000))
    .update(predictions, targets)
    .compute()
    .map50
)
# 1.0 with this PR; unsupported keyword on released 0.30.9
```

## Motivation and Context

Dense scenes can contain more than 100 instances of a single class. Callers should be able to evaluate supplied predictions under an explicit larger cap, without replacing the evaluator or changing the standard default.

The limit is **per image and category after class mapping or class-agnostic merging**, not 100 objects across an entire dataset. Ten categories of 100 objects each can score 1.0 at the default cap. COCO's 101-point interpolation explains why 1,000 perfect detections score 11/101 rather than exactly 0.1.

There is precedent for this configuration in [pycocotools `params.maxDets`](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py), [TorchMetrics `max_detection_thresholds`](https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/detection/mean_ap.py), and [RF-DETR `eval_max_dets`](https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/config.py).

## Changes Made

- Add `max_detection_thresholds: Sequence[int] = (1, 10, 100)` as a keyword-only constructor option. Validate exactly three strictly increasing positive integers, reject booleans/floats, and copy the sequence to prevent caller mutation. AP uses the last value; the three-limit structure follows the existing COCO evaluator and TorchMetrics convention.
- Pass the limits to the existing evaluator for boxes, masks and oriented boxes. Fix its internal headline summary to select the actual last cap instead of literal 100.
- Retain limits in overall and size-specific results. Custom text summaries and plot titles report the cap. DataFrame `attrs` retain configuration without changing score columns; docs explicitly state that CSV export drops those attributes.
- Document cap scope, custom metric comparability, and increased matching work/memory at higher limits. Add a changelog entry.

## Testing

- Baseline: `uv run --no-sync pytest --cov=supervision -q` → **5,672 passed, 1 skipped**, 38 warnings.
- Initial new tests on unchanged `develop`: **34 failed, 2 passed**; unsupported keyword and hardcoded headline selection reproduce the missing feature.
- Final branch: the same full-suite command → **5,717 passed, 1 skipped**, the same 38 warnings, coverage 93%. Includes **45 new tests** for boundary counts 99/100/101/350/1,000; caps 100/300/1,000; pycocotools agreement; false positives/missed predictions; categories, class-agnostic merging and explicit mapping; independent image caps; masks/oriented boxes; all size buckets; ties, empty inputs, reset, validation and result metadata.
- `uv run --no-sync pre-commit run --all-files` → **passed**, including mypy. Documentation hooks passed after adding the dense-scene guide.
- Standalone check: **20 deterministic scenes/controls, 80 baseline/candidate evaluations**, each checked against pycocotools at the same cap. The toy detector reads rendered pixels independently of ground-truth geometry. Few-object, ten-category and missed-object controls prevent interpreting a larger cap as an automatic accuracy improvement.

The local upstream test environment uses Python 3.11; the independently locked demo uses Python 3.14. Both exercise the actual package source, without monkeypatching.

- [x] Added tests and full suite passes locally
- [x] Updated docstrings and metric guide
- [x] Added an Unreleased changelog entry

## Additional Notes

Custom-cap AP must be labelled and compared at matching caps; it is not standard COCO AP@100. Increasing this cap cannot recover predictions removed by inference, NMS, decoding or model query limits. Larger matching matrices also cost more memory/work.

Keypoint AP retains cap 20 and its existing tests pass. MeanAverageRecall result naming is already discussed in [#2689](https://github.com/roboflow/supervision/issues/2689) / [#2701](https://github.com/roboflow/supervision/pull/2701); this PR does not duplicate that work. Hosted evaluation configuration/provenance is a separate platform question.

Once a PR number exists, its link can be added to the changelog entry.
