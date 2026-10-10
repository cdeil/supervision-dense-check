"""Render labelled stars and independently recover boxes from their pixels."""

import math
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

COUNTS = (25, 100, 101, 350, 1000)
CAPS = (100, 300, 1000)
COLORS = (
    (107, 72, 190),
    (23, 133, 118),
    (212, 97, 39),
    (42, 111, 182),
    (180, 57, 101),
    (132, 123, 29),
    (69, 137, 54),
    (166, 83, 174),
    (59, 118, 131),
    (145, 99, 54),
)


def render(count: int, classes: int = 1) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Draw isolated filled stars; labels follow the integer drawing geometry."""
    if count < 1 or not 1 <= classes <= len(COLORS):
        raise ValueError("Positive count and one to ten classes required")
    columns = math.ceil(math.sqrt(count * 4 / 3))
    rows = math.ceil(count / columns)
    cell = min(960 / columns, 704 / rows)
    left = (1024 - columns * cell) / 2
    top = (768 - rows * cell) / 2
    image = Image.new("RGB", (1024, 768), (251, 250, 246))
    draw = ImageDraw.Draw(image)
    boxes = []
    labels = np.arange(count, dtype=int) % classes
    for index in range(count):
        center = (
            left + (index % columns + 0.5) * cell,
            top + (index // columns + 0.5) * cell,
        )
        points = []
        for point in range(10):
            radius = cell * (0.38 if point % 2 == 0 else 0.17)
            angle = -math.pi / 2 + point * math.pi / 5
            points.append(
                (
                    round(center[0] + radius * math.cos(angle)),
                    round(center[1] + radius * math.sin(angle)),
                )
            )
        draw.polygon(points, fill=COLORS[int(labels[index])])
        xs, ys = zip(*points, strict=False)
        boxes.append([min(xs), min(ys), max(xs) + 1, max(ys) + 1])
    return image, np.asarray(boxes, dtype=float), labels


def detect(image: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find colored connected components without reading the rendering labels."""
    pixels = np.asarray(image)
    foreground = np.min(pixels, axis=2) < 220
    components, _ = ndimage.label(foreground)
    boxes, classes = [], []
    for label, slices in enumerate(ndimage.find_objects(components), start=1):
        if slices is None:
            continue
        y, x = slices
        boxes.append([x.start, y.start, x.stop, y.stop])
        positions = np.argwhere(components[y, x] == label)
        py, px = positions[0]
        color = tuple(int(v) for v in pixels[y.start + py, x.start + px])
        classes.append(COLORS.index(color))
    confidence = np.linspace(0.99, 0.5, len(boxes))
    return np.asarray(boxes, dtype=float), np.asarray(classes, dtype=int), confidence
