"""Build one YOLO training set for WildTrack from public aerial wildlife datasets.

Each source is optional; pass the ones you have downloaded (see training/README.md):

    python training/prepare_dataset.py --out data/wildtrack \
        --waid  /data/WAID/WAID \
        --bucktales /data/BuckTales/detection \
        --aed-images /data/AED/training_images --aed-points /data/AED/training_elephants.csv \
        --custom /data/my_leopards

What it does:
  * reads YOLO-format sources (WAID, the BuckTales detection subset, your own data) and
    point-labelled AED, and renames every label through training/class_map.yaml;
  * cuts large frames (drone stills, 5K video frames) into overlapping tiles, so small animals
    keep enough pixels at training resolution, the same reason the app tiles in Manual search;
  * keeps each source's own train/val/test split when it has one, otherwise splits at random;
  * writes <out>/images/<split>, <out>/labels/<split> and <out>/wildtrack.yaml for training.

Only standard YOLO txt labels are written: "<class> <x_c> <y_c> <w> <h>", normalised to [0, 1].
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val",
                 "validation": "val", "test": "test", "testing": "test"}


@dataclass
class Box:
    name: str          # already mapped to the WildTrack class name
    x0: float          # pixel coordinates in the source image
    y0: float
    x1: float
    y1: float


@dataclass
class Sample:
    source: str
    image: Path
    boxes: list[Box] = field(default_factory=list)
    split: str | None = None   # None = assign at random later


# --------------------------------------------------------------------------- class mapping

def load_class_map(path: Path) -> dict[str, dict[str, str | None]]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {src: (m or {}) for src, m in data.items()}


def map_name(class_map: dict[str, dict[str, str | None]], source: str, name: str) -> str | None:
    m = class_map.get(source, {})
    if name in m:
        return m[name]
    if "*" in m:
        return m["*"]
    return name


# --------------------------------------------------------------------------- readers

def split_from_path(path: Path) -> str | None:
    for part in reversed(path.parts[:-1]):
        s = SPLIT_ALIASES.get(part.lower())
        if s:
            return s
    return None


def find_names(root: Path, override: list[str] | None) -> list[str]:
    """Class names for a YOLO source: --names, else classes.txt / data.yaml near the root."""
    if override:
        return override
    for base in [root, *root.parents][:3]:
        txt = base / "classes.txt"
        if txt.is_file():
            return [ln.strip() for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for name in ("data.yaml", "dataset.yaml"):
            y = base / name
            if y.is_file():
                names = (yaml.safe_load(y.read_text(encoding="utf-8")) or {}).get("names")
                if isinstance(names, dict):
                    return [names[k] for k in sorted(names)]
                if isinstance(names, list):
                    return names
    sys.exit(f"{root}: no classes.txt or data.yaml found; pass the class names with --names")


def label_path_for(image: Path) -> Path:
    """Ultralytics convention: .../images/<split>/x.jpg -> .../labels/<split>/x.txt."""
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")   # labels next to the images


def read_yolo_source(source: str, root: Path, names: list[str],
                     class_map: dict[str, dict[str, str | None]]) -> list[Sample]:
    samples = []
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        sys.exit(f"{source}: no images found under {root}")
    for img in images:
        lbl = label_path_for(img)
        with Image.open(img) as im:
            w, h = im.size
        s = Sample(source, img, split=split_from_path(img))
        if lbl.is_file():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls, xc, yc, bw, bh = int(float(parts[0])), *map(float, parts[1:5])
                if not 0 <= cls < len(names):
                    continue
                name = map_name(class_map, source, names[cls])
                if name is None:
                    continue
                s.boxes.append(Box(name, (xc - bw / 2) * w, (yc - bh / 2) * h,
                                   (xc + bw / 2) * w, (yc + bh / 2) * h))
        samples.append(s)
    return samples


def read_point_source(source: str, images_dir: Path, points_csv: Path, box_px: float,
                      cols: tuple[int, int, int], class_map: dict[str, dict[str, str | None]]
                      ) -> list[Sample]:
    """Point labels (one animal centre per row) -> fixed-size boxes of box_px pixels.

    AED gives elephant centres, not boxes; a fixed box around each centre is the usual
    approximation (100 px in Naudé & Joubert's follow-up work). Rows: image id, x, y; a header
    row is skipped automatically. Use --aed-cols if your copy orders the columns differently.
    """
    by_stem: dict[str, list[tuple[float, float]]] = {}
    with open(points_csv, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) <= max(cols):
                continue
            try:
                x, y = float(row[cols[1]]), float(row[cols[2]])
            except ValueError:
                continue                     # header row
            by_stem.setdefault(Path(row[cols[0]].strip()).stem, []).append((x, y))
    name = map_name(class_map, source, "elephant")
    samples = []
    for img in sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
        s = Sample(source, img, split=split_from_path(img))
        if name is not None:
            r = box_px / 2
            s.boxes = [Box(name, x - r, y - r, x + r, y + r) for x, y in by_stem.get(img.stem, [])]
        samples.append(s)
    if not any(s.boxes for s in samples):
        sys.exit(f"{source}: no point labels matched any image; check --aed-cols and file names")
    return samples


# --------------------------------------------------------------------------- tiling

def tile_origins(n: int, tile: int, step: int) -> list[int]:
    if n <= tile:
        return [0]
    xs = list(range(0, n - tile, step)) + [n - tile]
    return sorted(set(xs))


def clip_boxes(boxes: list[Box], x: int, y: int, tw: int, th: int, min_keep: float) -> list[Box]:
    out = []
    for b in boxes:
        area = max(0.0, b.x1 - b.x0) * max(0.0, b.y1 - b.y0)
        cx0, cy0 = max(b.x0, x), max(b.y0, y)
        cx1, cy1 = min(b.x1, x + tw), min(b.y1, y + th)
        if cx1 <= cx0 or cy1 <= cy0 or area <= 0:
            continue
        if (cx1 - cx0) * (cy1 - cy0) / area < min_keep:
            continue
        out.append(Box(b.name, cx0 - x, cy0 - y, cx1 - x, cy1 - y))
    return out


# --------------------------------------------------------------------------- writing

def to_yolo_lines(boxes: list[Box], w: int, h: int, index: dict[str, int]) -> list[str]:
    lines = []
    for b in boxes:
        x0, y0 = max(0.0, b.x0), max(0.0, b.y0)
        x1, y1 = min(float(w), b.x1), min(float(h), b.y1)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        lines.append(f"{index[b.name]} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                     f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    return lines


def write_dataset(samples: list[Sample], out: Path, tile: int, overlap: float, min_keep: float,
                  empty_keep: float, rng: random.Random) -> dict[str, dict[str, int]]:
    names = sorted({b.name for s in samples for b in s.boxes})
    if not names:
        sys.exit("no labelled boxes left after class mapping; check training/class_map.yaml")
    index = {n: i for i, n in enumerate(names)}
    stats: dict[str, dict[str, int]] = {}
    step = max(1, int(tile * (1.0 - overlap)))
    for k, s in enumerate(samples):
        img_dir, lbl_dir = out / "images" / s.split, out / "labels" / s.split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{s.source}_{k:06d}"
        with Image.open(s.image) as im:
            w, h = im.size
            if max(w, h) <= tile * 1.5:          # small enough: copy the frame as it is
                shutil.copy2(s.image, img_dir / f"{stem}{s.image.suffix.lower()}")
                lines = to_yolo_lines(s.boxes, w, h, index)
                (lbl_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
                st = stats.setdefault(s.split, {"images": 0, "boxes": 0})
                st["images"] += 1
                st["boxes"] += len(lines)
                continue
            rgb = im.convert("RGB")
            for ty in tile_origins(h, tile, step):
                for tx in tile_origins(w, tile, step):
                    tw, th = min(tile, w - tx), min(tile, h - ty)
                    boxes = clip_boxes(s.boxes, tx, ty, tw, th, min_keep)
                    if not boxes and rng.random() > empty_keep:
                        continue                 # keep only a few empty tiles as negatives
                    name = f"{stem}_{tx}_{ty}"
                    rgb.crop((tx, ty, tx + tw, ty + th)).save(img_dir / f"{name}.jpg", quality=95)
                    lines = to_yolo_lines(boxes, tw, th, index)
                    (lbl_dir / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
                    st = stats.setdefault(s.split, {"images": 0, "boxes": 0})
                    st["images"] += 1
                    st["boxes"] += len(lines)
    splits = sorted(stats)
    cfg = {"path": str(out.resolve()), "names": {i: n for i, n in enumerate(names)}}
    for sp in ("train", "val", "test"):
        if sp in splits:
            cfg[sp] = f"images/{sp}"
    if "val" not in cfg:
        sys.exit("no validation images; raise --val-frac")
    with open(out / "wildtrack.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return stats


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="output folder for the merged dataset")
    ap.add_argument("--class-map", type=Path, default=here / "class_map.yaml")
    ap.add_argument("--waid", type=Path, help="WAID folder (the one holding classes.txt, images/, labels/)")
    ap.add_argument("--bucktales", type=Path, help="BuckTales detection subset, YOLO format")
    ap.add_argument("--bucktales-names", nargs="+", help="class names, if the subset has no classes.txt")
    ap.add_argument("--aed-images", type=Path, help="AED image folder")
    ap.add_argument("--aed-points", type=Path, help="AED CSV of elephant centre points")
    ap.add_argument("--aed-cols", type=int, nargs=3, default=(0, 1, 2), metavar=("IMG", "X", "Y"),
                    help="column indices of image id, x and y in the AED CSV (default 0 1 2)")
    ap.add_argument("--aed-box", type=float, default=100.0, help="box size in pixels around each AED point")
    ap.add_argument("--custom", type=Path, help="your own YOLO-format data (for example leopards)")
    ap.add_argument("--custom-names", nargs="+", help="class names, if --custom has no classes.txt")
    ap.add_argument("--tile", type=int, default=1280, help="tile size for large frames (match detector.imgsz)")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--min-keep", type=float, default=0.5,
                    help="keep a box cut by a tile edge only if this fraction of it is inside")
    ap.add_argument("--empty-keep", type=float, default=0.1, help="fraction of empty tiles kept as negatives")
    ap.add_argument("--val-frac", type=float, default=0.15, help="for sources without their own split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} is not empty; choose a new --out folder")
    class_map = load_class_map(args.class_map)
    rng = random.Random(args.seed)

    samples: list[Sample] = []
    if args.waid:
        samples += read_yolo_source("waid", args.waid, find_names(args.waid, None), class_map)
    if args.bucktales:
        samples += read_yolo_source("bucktales", args.bucktales,
                                    find_names(args.bucktales, args.bucktales_names), class_map)
    if args.aed_images or args.aed_points:
        if not (args.aed_images and args.aed_points):
            sys.exit("AED needs both --aed-images and --aed-points")
        samples += read_point_source("aed", args.aed_images, args.aed_points, args.aed_box,
                                     tuple(args.aed_cols), class_map)
    if args.custom:
        samples += read_yolo_source("custom", args.custom,
                                    find_names(args.custom, args.custom_names), class_map)
    if not samples:
        sys.exit("pass at least one source: --waid, --bucktales, --aed-images/--aed-points or --custom")

    for s in samples:                          # unsplit sources: random train/val
        if s.split is None:
            s.split = "val" if rng.random() < args.val_frac else "train"
    for src in {s.source for s in samples}:    # a source with train/test but no val: carve one out
        own = [s for s in samples if s.source == src]
        if not any(s.split == "val" for s in own):
            for s in own:
                if s.split == "train" and rng.random() < args.val_frac:
                    s.split = "val"

    stats = write_dataset(samples, args.out, args.tile, args.overlap, args.min_keep,
                          args.empty_keep, rng)
    print(f"wrote {args.out / 'wildtrack.yaml'}")
    for sp, st in sorted(stats.items()):
        print(f"  {sp:>5}: {st['images']:>6} images  {st['boxes']:>7} boxes")
    with open(args.out / "wildtrack.yaml", encoding="utf-8") as f:
        print("  classes:", ", ".join((yaml.safe_load(f) or {})["names"].values()))


if __name__ == "__main__":
    main()
