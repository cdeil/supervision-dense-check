# Supervision dense check

**[Interactive before/after demo](https://cdeil.github.io/supervision-dense-check/)** · [Supervision PR #2716](https://github.com/roboflow/supervision/pull/2716)

Perfect detections can receive low COCO AP in dense scenes: evaluation retains at most **100 predictions per image and category**, even when every supplied box is correct. That default is appropriate for the standard COCO benchmark. This check demonstrates an optional larger cap for counting/dense-scene evaluation, keeping the default unchanged.

| Perfect detections, one class | Released AP@100 | Proposed AP@1000 |
| --- | ---: | ---: |
| 25 | 1.000000 | 1.000000 |
| 100 | 1.000000 | 1.000000 |
| 101 | 0.990099 | 1.000000 |
| 350 | 0.287129 | 1.000000 |
| 1,000 | 0.108911 | 1.000000 |

AP50 and AP50–95 are equal here because the boxes match exactly. Scores use COCO's 101 recall samples, so they are not exactly `100 / object_count`.

```python
from supervision.metrics import MeanAveragePrecision

metric = MeanAveragePrecision(max_detection_thresholds=(1, 10, 1000))
result = metric.update(predictions, targets).compute()
```

Custom-cap AP is a different metric from standard COCO AP@100. Always identify the cap when comparing scores. A larger evaluation cap cannot recover predictions lost at inference, NMS, decoding or a model's query limit.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/). uv selects Python 3.14 and locks all dependencies in `uv.lock`. The `before` and `after` extras conflict intentionally: the baseline comes from PyPI (`supervision==0.30.9`), and the candidate comes from the immutable Git commit `23343ef871a9c89aba0c9df02a318ea8e5429f38`. No import-time patch or vendored evaluator is used.

```bash
git clone https://github.com/cdeil/supervision-dense-check.git
cd supervision-dense-check
export DENSE_CHECK_DATA="${DENSE_CHECK_DATA:-$HOME/data/supervision-dense-check}"
export UV_CACHE_DIR="$DENSE_CHECK_DATA/cache/uv"
export UV_PROJECT_ENVIRONMENT="$DENSE_CHECK_DATA/venv"
uv run --locked --extra before python reproduce.py --output results/before.json
uv run --locked --extra after python reproduce.py --after \
  --baseline results/before.json --output site/results.json
uv run --locked --extra after pytest tests.py -q
uv run --locked --extra after ruff check .
uv run --locked --extra after ruff format --check .
uv run --locked --extra after python -m http.server 8000 --directory site
```

Open `http://localhost:8000`. Switching extras replaces the installed evaluator; it does not combine both packages. CI runs both evaluations afresh and checks that the published JSON matches the regenerated results before deploying Pages.

`scene.py` draws stars and records ground-truth geometry. Its independent, deterministic toy detector reads the **image pixels**, using connected components and color lookup. It does not consume the ground-truth boxes. Scores decrease in raster order to make the cap selection unambiguous. This isolates metric behavior without model quality, weights, network inference or an inference cap. It is not a learned detection model or a claim about real-world model accuracy.

The matrix covers 25/100/101/350/1,000 stars, one/ten categories, and perfect/missed-object controls: **20 scenes/controls and 80 evaluator runs** (20 baseline; 60 candidate at caps 100/300/1,000). Every result is checked against pycocotools configured with the same `maxDets`. The reference reads the precision tensor's actual cap slice, avoiding pycocotools' standard headline summary, which selects literal 100.

The missing-object control removes every tenth prediction. With ten categories this removes one complete category, deliberately making a simple, deterministic imperfect control. Raising the evaluation cap does not make these predictions perfect.

The browser displays precomputed Python scores and overlays exactly which predictions enter matching. It does not implement AP in JavaScript. JSON records versions, image pixel hashes, boxes, category IDs, scores, reference values and evaluation caps. Scores are rounded to 12 decimal places for portable artifacts after numerical agreement is checked. Hashes cover decoded RGB pixels, since PNG compression may differ between platforms. Released and development evaluators can differ slightly in floating-point precision; the default-cap and reference comparisons use a numerical tolerance of 1e-7.

## Scope and precedent

The proposed upstream change is confined to Supervision's `MeanAveragePrecision`: expose validated limits, pass them to its existing COCO evaluator, and label custom-cap results. Boxes, masks and oriented boxes share this path. Keypoint AP keeps its existing cap 20. Precision/recall/F1 and inference/data loaders are unaffected. MeanAverageRecall's custom-limit reporting is a separate existing upstream discussion.

Other evaluators already expose comparable configuration:

- [pycocotools](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py): `params.maxDets`.
- [TorchMetrics](https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/detection/mean_ap.py): `max_detection_thresholds`, default `[1, 10, 100]` and length three.
- [RF-DETR](https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/config.py): `eval_max_dets` for its native training evaluator. This is independent of its inference/query capacity.

A caller wrapping Supervision would need to expose and forward this option to make it user-selectable. No data-loader change is needed. A hosted platform would need its own supported configuration and clear score labels; changing the open-source evaluator alone cannot change a separate hosted evaluation service.

The proposed API, implementation and discussion are in [Supervision PR #2716](https://github.com/roboflow/supervision/pull/2716).
