# Training the aerial-wildlife detector

WildTrack needs a detector that recognises animals **from above**. COCO weights know what a dog or a horse looks like
from the side, not from a drone, so the app only uses them as a stand-in until you train your own. This folder
rebuilds the detector the original project used, a YOLO model fine-tuned on public aerial wildlife datasets, in three
commands. Trained weights are not shipped with the repository: each dataset has its own licence, and you should
train on the data you are allowed to use.

## Base model

**YOLO26s** (`yolo26s.pt`, COCO-pretrained, Ultralytics, January 2026) is the default starting point:

- it is the model the app is built and tested around (`detect.py`, YOLO26 + ByteTrack);
- YOLO26 adds small-target-aware label assignment (STAL), which keeps small objects from losing their positive
  labels during training, the main difficulty when an animal is a few dozen pixels from the air;
- the `s` size is a sensible balance of speed and accuracy on a laptop GPU. Use `yolo26n.pt` on a weak GPU or CPU,
  `yolo26m.pt` if you have the headroom.

Ultralytics also publishes a `yolo26-p2.yaml` architecture with an extra high-resolution head for very small objects,
but no pretrained weights for it; it is an option for advanced users, not the default.

## Public datasets

| Dataset | What it has | Labels | Licence | Get it |
| --- | --- | --- | --- | --- |
| **WAID** (Applied Sciences, 2023) | about 14,400 drone images: sheep, cattle, seal, camelus, kiang, zebra | YOLO boxes | none stated in the repository; the paper is CC BY 3.0 and some images come from Roboflow Universe, so check before redistributing | [github.com/xiaohuicui/WAID](https://github.com/xiaohuicui/WAID) |
| **BuckTales**, detection subset (Naik et al., NeurIPS 2024) | 320 drone images (5.4K) of blackbuck antelope, a protected Indian species | YOLO and COCO boxes | CC BY-SA 4.0 | [doi:10.17617/3.jcz9wk](https://doi.org/10.17617/3.jcz9wk) |
| **Aerial Elephant Dataset** (Naudé & Joubert, 2019) | 2,074 aerial images, 15,581 elephants | centre points, not boxes | CC0 1.0 | [doi:10.5281/zenodo.3234780](https://doi.org/10.5281/zenodo.3234780) (16.3 GB) |

Also worth a look: **MMLA** (2025), low-altitude drone footage of zebras, giraffes, onagers and African
wild dogs, published on Hugging Face by Imageomics with its own YOLO preparation script. Convert it with their script
and pass it in with `--custom`.

**What public data does not cover.** We found no public aerial dataset of leopards or tigers. For those targets, record
your own footage (with the permits and permissions in the main README), label it in YOLO format with any labelling
tool, and pass it with `--custom`. A few hundred labelled frames of your own animals, at your own altitudes, will help
more than any extra public data.

## Steps

From the repository root, with the laptop's virtual environment active (`pip install -r laptop/requirements.txt`
already installs Ultralytics and PyYAML):

**1. Download** the datasets you want into a folder outside the repository, for example `~/data`.

**2. Build one training set.** Every source is optional; pass the ones you have:

```bash
python training/prepare_dataset.py --out ~/data/wildtrack \
    --waid ~/data/WAID/WAID \
    --bucktales ~/data/BuckTales/detection --bucktales-names blackbuck \
    --aed-images ~/data/AED/training_images --aed-points ~/data/AED/training_elephants.csv
```

- Labels are renamed through [`class_map.yaml`](class_map.yaml) (for example WAID's `camelus` becomes `camel`, and
  `seal` is dropped). Edit it to merge or drop classes.
- Large frames are cut into overlapping 1280 px tiles, so a small animal keeps its pixels at training resolution.
- The paths above are examples; use the folder and file names of your own downloads.
- AED gives elephant centres only; each becomes a 100 px box (`--aed-box`). If your copy of the AED CSV orders its
  columns differently from image id, x, y, set them with `--aed-cols`.
- A source's own train/val/test split is kept when it has one; otherwise 15% goes to validation (`--val-frac`).
- `--bucktales-names` (and `--custom-names`) are only needed when the folder has no `classes.txt` or `data.yaml`.

**3. Train and install.**

```bash
python training/train.py --data ~/data/wildtrack/wildtrack.yaml          # first GPU, 100 epochs, 1280 px
python training/train.py --data ~/data/wildtrack/wildtrack.yaml --epochs 3 --device cpu   # quick check only
```

The best checkpoint is copied to `laptop/models/wildtrack-yolo26s.pt`, the path `laptop/config.yaml` already uses,
and the script prints the class names the model learned. Training curves and validation results are kept in
`training/runs/`.

**4. Point the app at it.**

- In `laptop/config.yaml`, check that `classes.targets` (animals that may be locked) and `classes.confusers` (never
  locked) use the printed names. Anything in neither list is shown but ignored.
- For the Tello profile, set in `laptop/config.tello.yaml`:

  ```yaml
  detector:
    weights: "models/wildtrack-yolo26s.pt"
    imgsz: 960            # the Tello streams 960x720
  classes:
    targets: [elephant, blackbuck]       # whatever you trained on
    confusers: [person, cattle, sheep]
  ```

## A note on altitude

The public datasets above were mostly flown much higher than a Tello's ceiling of about ten metres. A model trained
only on them sees animals smaller and flatter than a Tello will. If you fly a Tello, add some of your own low-altitude
frames through `--custom`; this gap between training data and deployment is exactly what decides how well the detector
holds up in the field.

## Licences

The training code here is MIT like the rest of the repository. Ultralytics YOLO is AGPL-3.0. Weights you train inherit
whatever obligations the datasets carry (BuckTales is share-alike), so check them before you distribute a model.
